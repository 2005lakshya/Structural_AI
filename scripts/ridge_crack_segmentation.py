"""
ridge_crack_segmentation.py
===========================
Training-free crack segmentation for close-up photos of concrete.

Cracks are thin, dark, elongated ridges. A multi-scale Sato ridge filter
responds to exactly that and, unlike the learned U-Net, does not need a
dataset; it also does not fire on large dark regions (shadows, clothing,
aggregate faces) because those are blobs, not ridges.

A connected component is kept only if it is long and thin
(skeleton length / sqrt(area) large), which removes speckle, pen marks that
are blob-like and aggregate edges that form closed loops.

An optional region of interest (x, y, w, h) restricts the search to the
concrete surface, which matters for full-scene photos (hands, table, etc.).

Known limitation: hand-drawn marks that are not near-black, and aggregate
edges, can still be picked up; there is no learned model to tell them apart.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import sato
from skimage.morphology import skeletonize


def segment_cracks(
    image_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    sigmas=(1.0, 1.5, 2.0, 3.0),
    percentile: float = 96.5,
    min_skeleton_px: int = 25,
    min_elongation: float = 2.0,
    exclude_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    """Return uint8 mask (255 = crack), same size as the input image."""
    h, w = image_bgr.shape[:2]
    x, y, rw, rh = roi if roi else (0, 0, w, h)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)[y:y + rh, x:x + rw]

    # Flatten slow illumination so the ridge response reflects local contrast only.
    bg = cv2.GaussianBlur(gray, (0, 0), 15)
    flat = gray - bg

    ridge = sato(flat, sigmas=sigmas, black_ridges=True)
    thr = np.percentile(ridge, percentile)
    binary = (ridge > thr).astype(np.uint8)
    # Cracks must also be darker than their surroundings (not merely edges).
    binary &= (flat < -6.0).astype(np.uint8)

    # Hand-drawn marks (ink, tally lines, labels) look like cracks to any ridge filter.
    # A near-black intensity cut was tried and rejected: it removes the darkest REAL
    # cracks too. Callers should pass the marked areas as exclusion boxes instead.
    for bx, by, bw, bh in (exclude_boxes or []):
        binary[max(by - y, 0):max(by + bh - y, 0), max(bx - x, 0):max(bx + bw - x, 0)] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        skel_len = int(skeletonize(comp > 0).sum())
        area = int(stats[i, cv2.CC_STAT_AREA])
        if skel_len < min_skeleton_px or skel_len / max(np.sqrt(area), 1.0) < min_elongation:
            continue
        # A closed ring whose inside is markedly darker than the ring's outside is the
        # outline of a dark blob (shadow, aggregate face), not a crack.
        interior = binary_fill_holes(comp > 0) & (comp == 0)
        if interior.sum() > 1.5 * area:
            ring = cv2.dilate(comp, np.ones((9, 9), np.uint8)) > 0
            outside = ring & ~(comp > 0) & ~interior
            if outside.any() and flat[interior].mean() < flat[outside].mean() - 25.0:
                continue
        keep |= comp

    full = np.zeros((h, w), dtype=np.uint8)
    full[y:y + rh, x:x + rw] = keep * 255
    return full
