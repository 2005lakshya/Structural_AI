"""
physics_informed_cracknet.py
=============================
Physics-Informed Dual-Branch Crack Assessment Network (PI-CrackNet).

Motivation
----------
CrackNet and SwinCrackNet are plain image classifiers: crack vs. no-crack.
That alone is not a defensible patent claim — "a CNN/Transformer classifies
images" is standard prior art regardless of which backbone is chosen.

This module instead couples the *learned* visual branch to the project's
own novel physics engine (physics_fracture_engine.py) and its own
skeleton/width measurement method (crack_width_measure.py), so the network
is trained to produce outputs that are consistent with Linear Elastic
Fracture Mechanics — not just visually plausible.

Architecture
------------
                image ──► CNN encoder (CrackNet backbone) ──► visual embedding (1024)
                                                                        │
   binary mask ──► morphology descriptor (skeleton/width/branching) ──► MLP encoder (32)
                                                                        │
                                                    concat ──► shared trunk (256)
                                                                   │         │
                                                     classification head   physics head
                                                     (crack / no_crack)   (predicted fracture
                                                                           ratio K_I / K_IC)

The physics head is supervised with a *pseudo-label* computed by running
the crack mask through crack_width_measure.measure_crack_width() to get a
width estimate, feeding that into
physics_fracture_engine.evaluate_fracture_and_capacity() to get a
fracture_ratio, and using that as the regression target. This means the
network's structural-risk output at inference time is grounded in the same
fracture-mechanics equations the rest of the system already uses for
manual analysis — the vision model and the physics engine are trained as
one system instead of being two disconnected stages (detect, then
separately compute physics on the result).

This is a prototype/reference implementation: the physics pseudo-labels
depend on a pixel-to-mm calibration, which during training is a nominal
placeholder (see NOMINAL_PIXELS_PER_MM below). A production version should
generate pseudo-labels from images with a known scale (e.g. a reference
marker in frame), consistent with the calibration options already
implemented in crack_width_measure.py / backend/main.py's
/measure_crack_width_image endpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from scripts.cracknet import DepthwiseSepConv
from scripts.crack_width_measure import measure_crack_width
from scripts.physics_fracture_engine import evaluate_fracture_and_capacity


# Nominal scale used only to generate training-time physics pseudo-labels
# when no calibration reference is available (see module docstring).
NOMINAL_PIXELS_PER_MM = 5.0
NOMINAL_CRACK_DEPTH_MM = 15.0  # placeholder depth; refine with calculate_crack_depth logic


# ─────────────────────────────────────────────────────────────────────────────
#  1. Morphology descriptor — turns a binary crack mask into a fixed-length
#     geometric feature vector (reuses the project's own skeletonization).
# ─────────────────────────────────────────────────────────────────────────────

def extract_morphology_descriptor(mask: np.ndarray, num_orientation_bins: int = 8) -> np.ndarray:
    """
    Compute a fixed-length descriptor of crack shape from a binary mask
    (uint8, 255 = crack). Returns a float32 vector of length 6 + num_orientation_bins:

        [crack_pixel_ratio, skeleton_length_norm, mean_width_px,
         width_std_px, num_branch_points_norm, tortuosity,
         *orientation_histogram]

    This is the geometric information the physics engine actually reasons
    over (width, branching, orientation) — feeding it to the network
    directly, alongside raw pixels, is what lets the shared trunk learn a
    representation aligned with the physics head's target.
    """
    import cv2

    h, w = mask.shape[:2]
    binary = (mask > 127).astype(np.uint8)
    total_px = float(h * w)
    crack_pixel_ratio = float(np.sum(binary)) / total_px

    if np.sum(binary) < 20:
        return np.zeros(6 + num_orientation_bins, dtype=np.float32)

    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skeleton = _skeletonize(binary)
    skel_ys, skel_xs = np.where(skeleton > 0)

    if len(skel_ys) < 5:
        return np.zeros(6 + num_orientation_bins, dtype=np.float32)

    widths_px = dist_transform[skel_ys, skel_xs] * 2.0
    mean_width_px = float(np.mean(widths_px))
    width_std_px = float(np.std(widths_px))

    skeleton_length_px = len(skel_ys)
    diag = math.sqrt(h ** 2 + w ** 2)
    skeleton_length_norm = skeleton_length_px / diag

    # Branch points: skeleton pixels with 3+ neighbours in an 8-connected sense.
    branch_count = _count_branch_points(skeleton)
    num_branch_points_norm = branch_count / max(skeleton_length_px, 1) * 100.0

    # Tortuosity: skeleton path length vs. straight-line end-to-end distance.
    end_to_end = math.hypot(
        skel_xs.max() - skel_xs.min(), skel_ys.max() - skel_ys.min()
    )
    tortuosity = skeleton_length_px / max(end_to_end, 1.0)

    # Local orientation histogram via image gradient of the mask.
    gx = cv2.Sobel(binary.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(binary.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    angles = np.arctan2(gy[skel_ys, skel_xs], gx[skel_ys, skel_xs])
    hist, _ = np.histogram(angles, bins=num_orientation_bins, range=(-math.pi, math.pi))
    hist = hist.astype(np.float32)
    hist = hist / max(hist.sum(), 1.0)

    descriptor = np.concatenate([
        np.array([
            crack_pixel_ratio,
            skeleton_length_norm,
            mean_width_px,
            width_std_px,
            num_branch_points_norm,
            tortuosity,
        ], dtype=np.float32),
        hist,
    ])
    return descriptor


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Same iterative morphological thinning used in crack_width_measure.py."""
    import cv2
    skel = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(skel, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(skel, temp)
        skel = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel * 255


def _count_branch_points(skeleton: np.ndarray) -> int:
    """Count skeleton pixels with 3+ of 8 neighbours also on the skeleton."""
    import cv2
    binary = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    branch_mask = (binary == 1) & (neighbor_count >= 3)
    return int(np.sum(branch_mask))


# ─────────────────────────────────────────────────────────────────────────────
#  2. Physics pseudo-label — the supervisory signal for the physics head.
# ─────────────────────────────────────────────────────────────────────────────

def compute_physics_pseudo_target(
    mask: np.ndarray,
    original_bgr: Optional[np.ndarray] = None,
    pixels_per_mm: float = NOMINAL_PIXELS_PER_MM,
    crack_depth_mm: float = NOMINAL_CRACK_DEPTH_MM,
) -> float:
    """
    Run the mask through the project's own width-measurement and fracture
    engine to produce a scalar fracture_ratio (K_I / K_IC) target for the
    physics head. fracture_ratio >= 1.0 means unstable brittle propagation
    per physics_fracture_engine's classification.
    """
    if original_bgr is None:
        original_bgr = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)

    result = measure_crack_width(
        mask, original_bgr,
        known_length_px=pixels_per_mm, known_length_mm=1.0,  # forces given scale
    )
    width_mm = result.p95_width_mm if result.crack_pixels >= 20 else 0.0

    physics = evaluate_fracture_and_capacity(
        crack_depth_mm=crack_depth_mm if width_mm > 0 else 1.0,
        crack_width_mm=max(width_mm, 0.01),
    )
    return float(physics.fracture_ratio)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Network
# ─────────────────────────────────────────────────────────────────────────────

class MorphologyEncoder(nn.Module):
    """Small MLP embedding for the geometric descriptor."""

    def __init__(self, in_dim: int, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsInformedCrackNet(nn.Module):
    """
    Dual-branch crack assessment network: CNN visual branch + crack-morphology
    branch, fused into a shared trunk with two heads (classification +
    physics-consistency regression). See module docstring for rationale.
    """

    def __init__(
        self,
        morphology_dim: int,
        num_classes: int = 2,
        morphology_embed_dim: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__()

        # Visual branch — same depthwise-separable stack as CrackNet.
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            DepthwiseSepConv(32, 64, stride=1),
            DepthwiseSepConv(64, 128, stride=2),
            DepthwiseSepConv(128, 128, stride=1),
            DepthwiseSepConv(128, 256, stride=2),
            DepthwiseSepConv(256, 256, stride=1),
            DepthwiseSepConv(256, 512, stride=2),
            DepthwiseSepConv(512, 512, stride=1),
            DepthwiseSepConv(512, 512, stride=1),
            DepthwiseSepConv(512, 512, stride=1),
            DepthwiseSepConv(512, 512, stride=1),
            DepthwiseSepConv(512, 512, stride=1),
            DepthwiseSepConv(512, 1024, stride=2),
            DepthwiseSepConv(1024, 1024, stride=1),
        )
        self.visual_pool = nn.AdaptiveAvgPool2d(1)

        self.morphology_encoder = MorphologyEncoder(morphology_dim, morphology_embed_dim)

        trunk_in = 1024 + morphology_embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classification_head = nn.Linear(256, num_classes)
        self.physics_head = nn.Linear(256, 1)  # predicted fracture_ratio

    def forward(self, image: torch.Tensor, morphology: torch.Tensor):
        visual = self.visual_encoder(image)
        visual = self.visual_pool(visual).flatten(1)

        morph = self.morphology_encoder(morphology)

        fused = torch.cat([visual, morph], dim=1)
        trunk_out = self.trunk(fused)

        class_logits = self.classification_head(trunk_out)
        physics_pred = self.physics_head(trunk_out).squeeze(-1)
        return class_logits, physics_pred


# ─────────────────────────────────────────────────────────────────────────────
#  4. Combined loss
# ─────────────────────────────────────────────────────────────────────────────

def combined_loss(
    class_logits: torch.Tensor,
    class_target: torch.Tensor,
    physics_pred: torch.Tensor,
    physics_target: torch.Tensor,
    lambda_physics: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """
    Total loss = classification cross-entropy + lambda * physics-consistency MSE.

    Only cracked samples contribute to the physics term (fracture_ratio is
    undefined/meaningless for no-crack patches).
    """
    ce = nn.functional.cross_entropy(class_logits, class_target)

    crack_mask = class_target == 1
    if crack_mask.any():
        physics_mse = nn.functional.mse_loss(
            physics_pred[crack_mask], physics_target[crack_mask]
        )
    else:
        physics_mse = torch.tensor(0.0, device=class_logits.device)

    total = ce + lambda_physics * physics_mse
    return total, {"ce_loss": ce.item(), "physics_mse": physics_mse.item()}


if __name__ == "__main__":
    # Smoke test with random tensors, mirroring cracknet.py's convention.
    descriptor_dim = 6 + 8
    model = PhysicsInformedCrackNet(morphology_dim=descriptor_dim)

    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_morph = torch.randn(2, descriptor_dim)
    class_logits, physics_pred = model(dummy_images, dummy_morph)
    print(f"class_logits shape : {class_logits.shape}")   # (2, 2)
    print(f"physics_pred shape : {physics_pred.shape}")   # (2,)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")

    # Descriptor + pseudo-label extraction demo on a synthetic mask.
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[60:68, 10:118] = 255  # a fake horizontal crack
    desc = extract_morphology_descriptor(mask)
    print(f"Descriptor shape : {desc.shape}")
    pseudo_target = compute_physics_pseudo_target(mask)
    print(f"Physics pseudo-label (fracture_ratio) : {pseudo_target:.3f}")
