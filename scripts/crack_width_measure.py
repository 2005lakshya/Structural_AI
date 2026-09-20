"""
crack_width_measure.py
======================
Image-based crack width measurement using morphological skeletonization
and the distance transform — no structural drawings required.

Pipeline
--------
1. Binary mask (from U-Net) → clean with morphological ops
2. Skeletonize  → 1-pixel-wide centre-line of the crack
3. Distance transform on the mask → at every skeleton pixel the value
   equals the radius of the largest inscribed circle, i.e. half the
   local crack width in pixels
4. Width in pixels  →  width in mm via pixels-per-mm scale factor
5. Classify against IS 456:2000 Table 3 exposure limits
6. Draw measurement overlay on original image

Scale calibration (choose one)
--------------------------------
A. Camera distance (cm) + image height (px) + sensor height (mm)
      pixels_per_mm = (image_height_px * focal_length_mm)
                      / (distance_cm * 10 * sensor_height_mm)
   Defaults assume a typical phone camera at 50 cm distance.

B. Reference object  – caller passes known_length_px and known_length_mm,
      pixels_per_mm = known_length_px / known_length_mm
   E.g. a standard Indian rupee coin is 25 mm in diameter.

C. Manual DPI   – caller passes dpi value (scanner / known-res camera)
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  IS 456 : 2000 exposure limits (Table 3 / Clause 35.3.2)
# ─────────────────────────────────────────────────────────────────────────────

IS456_LIMITS = {
    "mild":       0.30,
    "moderate":   0.20,
    "severe":     0.10,
    "very_severe": 0.10,
    "extreme":    0.10,
}

IS456_LABELS = {
    "mild":        "Mild",
    "moderate":    "Moderate",
    "severe":      "Severe",
    "very_severe": "Very Severe",
    "extreme":     "Extreme",
}


def _classify_severity(wcr_mm: float) -> dict:
    """Return IS 456 severity label and description for a crack width in mm."""
    if wcr_mm <= 0.10:
        return {
            "level": "Mild",
            "color_bgr": (0, 200, 80),
            "description": "Hairline crack. Compliant with ALL IS 456 exposure classes.",
            "action": "Routine monitoring. No intervention required."
        }
    elif wcr_mm <= 0.20:
        return {
            "level": "Moderate",
            "color_bgr": (0, 200, 200),
            "description": "Compliant for Mild & Moderate exposures only.",
            "action": "Apply protective coating if in marine or coastal environment."
        }
    elif wcr_mm <= 0.30:
        return {
            "level": "Severe",
            "color_bgr": (0, 140, 255),
            "description": "Compliant ONLY for Mild (protected) exposure.",
            "action": "Seal crack. Increase steel area or reduce bar spacing in future."
        }
    elif wcr_mm <= 0.50:
        return {
            "level": "Very Severe",
            "color_bgr": (0, 80, 255),
            "description": "Exceeds ALL IS 456 exposure limits.",
            "action": "Epoxy / polyurethane pressure grouting required urgently."
        }
    else:
        return {
            "level": "Extreme",
            "color_bgr": (0, 0, 220),
            "description": "Critical structural hazard (> 0.5 mm).",
            "action": "Immediate structural audit, load restriction, and full remediation."
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrackWidthResult:
    # Pixel-space measurements
    mean_width_px:   float = 0.0
    max_width_px:    float = 0.0
    median_width_px: float = 0.0
    p95_width_px:    float = 0.0      # 95th-percentile — robust "worst case"

    # Real-world measurements (mm)
    mean_width_mm:   float = 0.0
    max_width_mm:    float = 0.0
    median_width_mm: float = 0.0
    p95_width_mm:    float = 0.0      # this is the design crack width for IS 456

    # Scale
    pixels_per_mm:   float = 1.0
    scale_method:    str   = "unknown"

    # IS 456 compliance
    exposure:        str   = "moderate"
    exposure_limit_mm: float = 0.20
    is_compliant:    bool  = True
    severity:        dict  = field(default_factory=dict)

    # Skeleton stats
    skeleton_length_px: int = 0
    crack_pixels:        int = 0

    # Visualisation (BGR numpy array)
    overlay_image: Optional[np.ndarray] = None

    # Width distribution histogram data (for paper figures)
    width_histogram_mm: List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Core measurement function
# ─────────────────────────────────────────────────────────────────────────────

def measure_crack_width(
    mask: np.ndarray,                  # binary uint8, 255 = crack
    original_bgr: np.ndarray,          # original image for overlay
    *,
    # Scale — provide ONE of the following three options:
    camera_distance_cm: float = 50.0,  # option A (default)
    sensor_height_mm:   float = 4.8,   # option A  (1/3" sensor ≈ 4.8 mm)
    focal_length_mm:    float = 4.2,   # option A  (typical phone focal length)
    known_length_px:    float = 0.0,   # option B
    known_length_mm:    float = 0.0,   # option B
    dpi:                float = 0.0,   # option C
    exposure:           str   = "moderate",
) -> CrackWidthResult:
    """
    Measure crack widths from a binary segmentation mask.

    Parameters
    ----------
    mask              : uint8 array (H×W), white = crack
    original_bgr      : BGR image for drawing the overlay
    camera_distance_cm: distance from camera to wall surface (cm)
    sensor_height_mm  : physical sensor height (mm) — phone default 4.8 mm
    focal_length_mm   : camera focal length (mm) — phone default 4.2 mm
    known_length_px   : pixel length of a known reference object (option B)
    known_length_mm   : real length of that reference object in mm (option B)
    dpi               : scanner resolution in dots-per-inch (option C)
    exposure          : IS 456 exposure class key

    Returns
    -------
    CrackWidthResult  dataclass with all measurements + overlay image
    """
    result = CrackWidthResult(exposure=exposure)
    result.exposure_limit_mm = IS456_LIMITS.get(exposure, 0.20)

    h_px, w_px = mask.shape[:2]

    # ── 1. Clean mask ──────────────────────────────────────────────────────
    binary = (mask > 127).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    result.crack_pixels = int(np.sum(binary))
    if result.crack_pixels < 20:
        # No meaningful crack found
        result.severity = _classify_severity(0.0)
        result.overlay_image = original_bgr.copy()
        return result

    # ── 2. Distance transform ──────────────────────────────────────────────
    # At each crack pixel, dist_transform gives the distance to the nearest
    # background pixel → equals the local half-width of the crack.
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # ── 3. Skeletonize (Zhang-Suen via iterative thinning) ────────────────
    skeleton = _skeletonize(binary)
    result.skeleton_length_px = int(np.sum(skeleton > 0))

    if result.skeleton_length_px < 5:
        result.severity = _classify_severity(0.0)
        result.overlay_image = original_bgr.copy()
        return result

    # ── 4. Sample width at every skeleton pixel ────────────────────────────
    # half_widths_px: radius at each point → full width = 2 × radius
    skel_ys, skel_xs = np.where(skeleton > 0)
    half_widths_px = dist_transform[skel_ys, skel_xs]
    widths_px = half_widths_px * 2.0          # full width in pixels

    result.mean_width_px   = float(np.mean(widths_px))
    result.max_width_px    = float(np.max(widths_px))
    result.median_width_px = float(np.median(widths_px))
    result.p95_width_px    = float(np.percentile(widths_px, 95))

    # ── 5. Pixels → mm  ───────────────────────────────────────────────────
    ppm, method = _compute_pixels_per_mm(
        h_px, w_px,
        camera_distance_cm=camera_distance_cm,
        sensor_height_mm=sensor_height_mm,
        focal_length_mm=focal_length_mm,
        known_length_px=known_length_px,
        known_length_mm=known_length_mm,
        dpi=dpi,
    )
    result.pixels_per_mm = ppm
    result.scale_method  = method

    result.mean_width_mm   = round(result.mean_width_px   / ppm, 4)
    result.max_width_mm    = round(result.max_width_px    / ppm, 4)
    result.median_width_mm = round(result.median_width_px / ppm, 4)
    result.p95_width_mm    = round(result.p95_width_px    / ppm, 4)

    # Histogram for paper figure (20 bins 0–2 mm)
    widths_mm_all = widths_px / ppm
    hist, edges = np.histogram(widths_mm_all, bins=20, range=(0, 2.0))
    result.width_histogram_mm = hist.tolist()

    # ── 6. IS 456 compliance ──────────────────────────────────────────────
    # Use 95th percentile as the design crack width (conservative but fair)
    design_wcr = result.p95_width_mm
    result.is_compliant = design_wcr <= result.exposure_limit_mm
    result.severity = _classify_severity(design_wcr)

    # ── 7. Overlay image ───────────────────────────────────────────────────
    result.overlay_image = _draw_overlay(
        original_bgr, skeleton, dist_transform, widths_px, ppm,
        result.p95_width_mm, result.severity, result.is_compliant,
        result.exposure_limit_mm,
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Scale computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pixels_per_mm(
    h_px: int, w_px: int,
    camera_distance_cm: float,
    sensor_height_mm: float,
    focal_length_mm: float,
    known_length_px: float,
    known_length_mm: float,
    dpi: float,
) -> Tuple[float, str]:
    """Return (pixels_per_mm, method_name)."""

    # Option B — reference object (most accurate)
    if known_length_px > 0 and known_length_mm > 0:
        return known_length_px / known_length_mm, "reference_object"

    # Option C — scanner DPI
    if dpi > 0:
        return dpi / 25.4, "dpi"

    # Option A — pinhole camera model (default)
    # pixels_per_mm_real = (h_px / sensor_h_mm) * (f_mm / dist_mm)
    dist_mm = camera_distance_cm * 10.0
    if dist_mm > 0 and sensor_height_mm > 0 and focal_length_mm > 0:
        # mm of scene per pixel height
        mm_per_px = (dist_mm * sensor_height_mm) / (focal_length_mm * h_px)
        return 1.0 / mm_per_px, "camera_model"

    # Fallback: assume 5 px per mm (very rough)
    return 5.0, "fallback_5px_per_mm"


# ─────────────────────────────────────────────────────────────────────────────
#  Skeletonization  (Zhang-Suen thinning)
# ─────────────────────────────────────────────────────────────────────────────

def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """
    Iterative morphological thinning (Zhang-Suen).
    Returns uint8 array with skeleton pixels = 255.
    """
    skel = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded   = cv2.erode(skel, kernel)
        temp     = cv2.dilate(eroded, kernel)
        temp     = cv2.subtract(skel, temp)
        skel     = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel * 255


# ─────────────────────────────────────────────────────────────────────────────
#  Overlay visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _draw_overlay(
    original_bgr: np.ndarray,
    skeleton: np.ndarray,
    dist_transform: np.ndarray,
    widths_px: np.ndarray,
    ppm: float,
    design_wcr_mm: float,
    severity: dict,
    is_compliant: bool,
    limit_mm: float,
) -> np.ndarray:
    """Draw coloured skeleton + IS 456 compliance banner on the image."""
    overlay = original_bgr.copy()

    # Colour-map the skeleton by local width (blue=thin → red=wide)
    skel_ys, skel_xs = np.where(skeleton > 0)
    if len(skel_ys) == 0:
        return overlay

    max_w = max(float(np.max(widths_px)), 1.0)
    for y, x, w in zip(skel_ys, skel_xs, widths_px):
        t = min(w / max_w, 1.0)
        # interpolate: green (thin) → yellow → red (wide)
        b = int(0)
        g = int(255 * (1.0 - t))
        r = int(255 * t)
        cv2.circle(overlay, (int(x), int(y)), 1, (b, g, r), -1)

    # Banner at top
    banner_h = 56
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], banner_h),
                  (30, 30, 30), -1)

    color = (0, 200, 80) if is_compliant else (0, 60, 220)
    status_text = "COMPLIANT" if is_compliant else "NON-COMPLIANT"
    cv2.putText(overlay,
                f"IS 456 {status_text}  |  W_cr (p95) = {design_wcr_mm:.3f} mm  |  Limit = {limit_mm:.2f} mm  |  {severity['level']}",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: run from a raw image (no pre-computed mask)
# ─────────────────────────────────────────────────────────────────────────────

def measure_from_image(
    image_rgb: np.ndarray,
    unet_mask_fn,             # callable: image_rgb → binary mask uint8
    camera_distance_cm: float = 50.0,
    known_length_px: float = 0.0,
    known_length_mm: float = 0.0,
    dpi: float = 0.0,
    exposure: str = "moderate",
) -> CrackWidthResult:
    """
    End-to-end: RGB image → CrackWidthResult.
    unet_mask_fn is the run_unet_mask() function from main.py.
    """
    mask = unet_mask_fn(image_rgb)
    bgr  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return measure_crack_width(
        mask, bgr,
        camera_distance_cm=camera_distance_cm,
        known_length_px=known_length_px,
        known_length_mm=known_length_mm,
        dpi=dpi,
        exposure=exposure,
    )
