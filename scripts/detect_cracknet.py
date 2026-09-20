"""
detect_cracknet.py — Sliding-window crack detector using the trained CrackNet classifier.

Produces the same output interface as the YOLO pipeline so the backend can
swap in this detector with minimal changes.

Output per image:
    detections : list of {class, confidence, bbox:[x1,y1,x2,y2]}
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

from scripts.cracknet import CrackNet, load_cracknet  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
#  Detector class
# ─────────────────────────────────────────────────────────────────────────────

class CrackNetDetector:
    """
    Wraps CrackNet with a sliding-window approach to produce bounding boxes.

    Strategy
    --------
    1. Run CrackNet on the full resized image → early-exit if P(crack) < threshold.
    2. Tile the image with overlapping windows at two scales.
    3. Score each window patch; keep windows where P(crack) ≥ conf_threshold.
    4. Merge overlapping detections with NMS.
    5. Return bbox list + annotated image (matching the YOLO output format used in
       backend/main.py).
    """

    DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "models", "cracknet", "cracknet_best.pth")

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | None = None,
        conf_threshold: float = 0.55,
        nms_iou_threshold: float = 0.40,
        tile_sizes: Tuple[int, ...] = (224, 112),
        stride_ratio: float = 0.5,
    ):
        self.conf_threshold  = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.tile_sizes      = tile_sizes
        self.stride_ratio    = stride_ratio

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        wp = weights_path or self.DEFAULT_WEIGHTS
        if not os.path.exists(wp):
            raise FileNotFoundError(
                f"CrackNet weights not found at: {wp}\n"
                "Run  python scripts/train_cracknet.py  first."
            )
        self.model = load_cracknet(wp, device=device)

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Parameters
        ----------
        image_rgb : H×W×3 numpy array, uint8, RGB colour space.

        Returns
        -------
        dict with keys:
            detections  : list[{class, confidence, bbox:[x1,y1,x2,y2]}]
            count       : int
            annotated   : H×W×3 numpy array (RGB) with drawn boxes
        """
        h, w = image_rgb.shape[:2]
        detections = self._sliding_window(image_rgb)
        detections = self._nms(detections)

        annotated = self._draw(image_rgb.copy(), detections)

        return {
            "detections": [
                {
                    "class": 0,
                    "confidence": round(float(d["conf"]), 4),
                    "bbox": [round(float(v), 2) for v in d["bbox"]],
                }
                for d in detections
            ],
            "count": len(detections),
            "annotated": annotated,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _score_patches(self, patches: List[np.ndarray]) -> List[float]:
        """Score a list of HxWx3 uint8 RGB patches; return P(crack) per patch."""
        tensors = []
        for patch in patches:
            pil = Image.fromarray(patch).resize((224, 224), Image.BILINEAR)
            t   = TF.to_tensor(pil)
            t   = TF.normalize(t, _MEAN, _STD)
            tensors.append(t)

        batch = torch.stack(tensors).to(self.device)     # B×3×224×224
        probs = self.model.predict_proba(batch)          # B×2
        return probs[:, 1].cpu().tolist()                # P(crack)

    def _sliding_window(self, img: np.ndarray) -> List[Dict]:
        h, w = img.shape[:2]
        raw_boxes: List[Dict] = []

        for tile_sz in self.tile_sizes:
            stride = max(1, int(tile_sz * self.stride_ratio))

            # collect all patches for this scale → batch-score them
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

            # batch in chunks of 64 to keep VRAM / RAM reasonable
            scores: List[float] = []
            for i in range(0, len(patches), 64):
                scores.extend(self._score_patches(patches[i: i + 64]))

            for (x1, y1, x2, y2), conf in zip(coords, scores):
                if conf >= self.conf_threshold:
                    raw_boxes.append({"bbox": [x1, y1, x2, y2], "conf": conf})

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
        """Draw bounding boxes and confidence scores on a copy of the image."""
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["conf"]
            color = (255, 60, 60)   # red in RGB
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"crack {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            # filled background for text
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone CLI  (python scripts/detect_cracknet.py --image path/to/img.jpg)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CrackNet sliding-window detector")
    ap.add_argument("--image",    required=True, help="Path to input image")
    ap.add_argument("--weights",  default=None,  help="Override model weights path")
    ap.add_argument("--conf",     type=float, default=0.55, help="Confidence threshold")
    ap.add_argument("--out",      default="cracknet_result.jpg", help="Output image path")
    args = ap.parse_args()

    detector = CrackNetDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
    )

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"Error: cannot read {args.image}")
        sys.exit(1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = detector.predict(rgb)

    print(f"Detections : {result['count']}")
    for d in result["detections"]:
        print(f"  conf={d['confidence']:.3f}  bbox={d['bbox']}")

    out_rgb = result["annotated"]
    cv2.imwrite(args.out, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
    print(f"Annotated image saved to: {args.out}")
