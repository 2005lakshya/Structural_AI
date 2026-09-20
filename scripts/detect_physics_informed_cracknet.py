"""
detect_physics_informed_cracknet.py — Sliding-window detector using PI-CrackNet.

Same sliding-window + NMS strategy as detect_cracknet.py / detect_swin_cracknet.py,
but for each candidate window this also:
  1. Runs TinyUNet to get a mask for that patch.
  2. Extracts the morphology descriptor from the mask.
  3. Runs PhysicsInformedCrackNet's dual-branch forward pass to get
     P(crack) AND a predicted fracture_ratio (K_I / K_IC) for that patch.

The fracture_ratio is exposed per-detection — this is the only detector in
the project whose confidence score is trained to also stay consistent with
the fracture-mechanics engine, so it's the preferred detector when its
weights are present (see ensure_crack_detector_loaded in backend/main.py).

Output per image (same interface as the other detectors so the backend can
swap detectors transparently):
    detections : list of {class, confidence, bbox:[x1,y1,x2,y2], fracture_ratio}
    annotated  : numpy array (RGB) with drawn bounding boxes
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scripts.extract_features import TinyUNet  # noqa: E402
from scripts.physics_informed_cracknet import (  # noqa: E402
    PhysicsInformedCrackNet,
    extract_morphology_descriptor,
)

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
DESCRIPTOR_DIM = 6 + 8
MASK_SIZE = 224


class PhysicsInformedCrackNetDetector:
    """Wraps PhysicsInformedCrackNet with a sliding-window approach to produce bounding boxes."""

    DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "models", "physics_informed_cracknet", "pi_cracknet_best.pth")
    DEFAULT_UNET_WEIGHTS = os.path.join(BASE_DIR, "models", "unet", "best_unet.pth")

    def __init__(
        self,
        weights_path: str | None = None,
        unet_weights_path: str | None = None,
        device: str | None = None,
        conf_threshold: float = 0.55,
        nms_iou_threshold: float = 0.40,
        tile_sizes: Tuple[int, ...] = (224, 112),
        stride_ratio: float = 0.5,
    ):
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.tile_sizes = tile_sizes
        self.stride_ratio = stride_ratio

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        wp = weights_path or self.DEFAULT_WEIGHTS
        if not os.path.exists(wp):
            raise FileNotFoundError(
                f"PhysicsInformedCrackNet weights not found at: {wp}\n"
                "Run  python scripts/train_physics_informed_cracknet.py  first."
            )
        ckpt = torch.load(wp, map_location=device, weights_only=True)
        descriptor_dim = ckpt.get("descriptor_dim", DESCRIPTOR_DIM)
        self.model = PhysicsInformedCrackNet(morphology_dim=descriptor_dim)
        self.model.load_state_dict(ckpt.get("model_state", ckpt))
        self.model.to(device)
        self.model.eval()

        unet_wp = unet_weights_path or self.DEFAULT_UNET_WEIGHTS
        if not os.path.exists(unet_wp):
            raise FileNotFoundError(
                f"TinyUNet weights not found at: {unet_wp}\n"
                "Run  python scripts/train_unet.py  first."
            )
        self.unet = TinyUNet()
        self.unet.load_state_dict(torch.load(unet_wp, map_location=device, weights_only=True))
        self.unet.to(device)
        self.unet.eval()

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Parameters
        ----------
        image_rgb : H×W×3 numpy array, uint8, RGB colour space.

        Returns
        -------
        dict with keys:
            detections  : list[{class, confidence, bbox:[x1,y1,x2,y2], fracture_ratio}]
            count       : int
            annotated   : H×W×3 numpy array (RGB) with drawn boxes
        """
        detections = self._sliding_window(image_rgb)
        detections = self._nms(detections)
        annotated = self._draw(image_rgb.copy(), detections)

        return {
            "detections": [
                {
                    "class": 0,
                    "confidence": round(float(d["conf"]), 4),
                    "bbox": [round(float(v), 2) for v in d["bbox"]],
                    "fracture_ratio": round(float(d["fracture_ratio"]), 4),
                }
                for d in detections
            ],
            "count": len(detections),
            "annotated": annotated,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_mask_batch(self, patches: List[np.ndarray]) -> List[np.ndarray]:
        tensors = []
        for patch in patches:
            resized = cv2.resize(patch, (128, 128))
            tensors.append(torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0)
        batch = torch.stack(tensors).to(self.device)
        outputs = self.unet(batch).squeeze(1).cpu().numpy()  # B×128×128

        masks = []
        for out, patch in zip(outputs, patches):
            mask = (out > 0.5).astype(np.uint8) * 255
            masks.append(cv2.resize(mask, (patch.shape[1], patch.shape[0])))
        return masks

    @torch.no_grad()
    def _score_patches(self, patches: List[np.ndarray]) -> Tuple[List[float], List[float]]:
        """Score a list of HxWx3 uint8 RGB patches; return (P(crack), fracture_ratio) per patch."""
        masks = self._infer_mask_batch(patches)

        image_tensors = []
        descriptors = []
        for patch, mask in zip(patches, masks):
            pil = Image.fromarray(patch).resize((224, 224), Image.BILINEAR)
            t = TF.normalize(TF.to_tensor(pil), _MEAN, _STD)
            image_tensors.append(t)

            resized_mask = cv2.resize(mask, (MASK_SIZE, MASK_SIZE))
            descriptors.append(extract_morphology_descriptor(resized_mask))

        image_batch = torch.stack(image_tensors).to(self.device)
        descriptor_batch = torch.tensor(np.stack(descriptors), dtype=torch.float32).to(self.device)

        class_logits, physics_pred = self.model(image_batch, descriptor_batch)
        probs = torch.softmax(class_logits, dim=-1)[:, 1].cpu().tolist()
        fracture_ratios = physics_pred.cpu().tolist()
        return probs, fracture_ratios

    def _sliding_window(self, img: np.ndarray) -> List[Dict]:
        h, w = img.shape[:2]
        raw_boxes: List[Dict] = []

        for tile_sz in self.tile_sizes:
            stride = max(1, int(tile_sz * self.stride_ratio))

            patches, coords = [], []
            y = 0
            while y + tile_sz <= h:
                x = 0
                while x + tile_sz <= w:
                    patches.append(img[y: y + tile_sz, x: x + tile_sz])
                    coords.append((x, y, x + tile_sz, y + tile_sz))
                    x += stride
                y += stride

            if not patches:
                continue

            scores, fracture_ratios = [], []
            for i in range(0, len(patches), 32):
                s, f = self._score_patches(patches[i: i + 32])
                scores.extend(s)
                fracture_ratios.extend(f)

            for (x1, y1, x2, y2), conf, fr in zip(coords, scores, fracture_ratios):
                if conf >= self.conf_threshold:
                    raw_boxes.append({"bbox": [x1, y1, x2, y2], "conf": conf, "fracture_ratio": fr})

        return raw_boxes

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, boxes: List[Dict]) -> List[Dict]:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
        keep = []
        suppressed = [False] * len(boxes)
        for i, b in enumerate(boxes):
            if suppressed[i]:
                continue
            keep.append(b)
            for j in range(i + 1, len(boxes)):
                if not suppressed[j]:
                    if self._iou(b["bbox"], boxes[j]["bbox"]) > self.nms_iou_threshold:
                        suppressed[j] = True
        return keep

    @staticmethod
    def _draw(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["conf"]
            fr = det["fracture_ratio"]
            color = (255, 60, 60)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"crack {conf:.2f} | Kr={fr:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PhysicsInformedCrackNet sliding-window detector")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--weights", default=None, help="Override model weights path")
    ap.add_argument("--conf", type=float, default=0.55, help="Confidence threshold")
    ap.add_argument("--out", default="pi_cracknet_result.jpg", help="Output image path")
    args = ap.parse_args()

    detector = PhysicsInformedCrackNetDetector(weights_path=args.weights, conf_threshold=args.conf)

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"Error: cannot read {args.image}")
        sys.exit(1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = detector.predict(rgb)

    print(f"Detections : {result['count']}")
    for d in result["detections"]:
        print(f"  conf={d['confidence']:.3f}  fracture_ratio={d['fracture_ratio']:.3f}  bbox={d['bbox']}")

    out_rgb = result["annotated"]
    cv2.imwrite(args.out, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
    print(f"Annotated image saved to: {args.out}")
