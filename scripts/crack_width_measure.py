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

    # Measurement quality
    width_method:        str   = "none"   # "subpixel_profile" | "mask_distance_transform"
    valid_profile_count: int   = 0
    scale_reliable:      bool  = False    # True only for reference object / DPI scale
    resolution_ok:       bool  = False    # median width >= MIN_RELIABLE_WIDTH_PX
    min_resolvable_width_mm: float = 0.0
    warnings:            List[str] = field(default_factory=list)

    # Resolution-robust photometric width, 90 % interval and certified decision
    # (photometric_assessment; empty dict when not available)
    photometric:         dict  = field(default_factory=dict)
    # All five certified stages: measurability limit, photo / fusion / network estimators,
    # intervals, decisions (location and element maximum), capture advice, gauge anchoring
    # (scripts/certified_width.py)
    certified:           dict  = field(default_factory=dict)


# Below this many pixels the crack edge blur dominates and the width is only an
# upper bound, not a measurement.
MIN_RELIABLE_WIDTH_PX = 2.0

# The grayscale image is smoothed with a Gaussian of this sigma before profiling.
# Gaussian blurs add in quadrature to a Gaussian-like line, so the FWHM of the
# smoothing is removed from every profile width (validated in
# scripts/validate_width_synthetic.py). Camera blur is NOT known and is not removed.
PREBLUR_SIGMA_PX = 0.8
PREBLUR_FWHM_PX = 2.3548 * PREBLUR_SIGMA_PX


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-pixel width from the real image intensity profile
# ─────────────────────────────────────────────────────────────────────────────

def _centres_and_normals(skeleton: np.ndarray, max_points: int = 600):
    """Sampled centre-line points (y, x) and unit normals (dy, dx) of the crack skeleton."""
    ys, xs = np.where(skeleton > 0)
    n = len(ys)
    if n < 5:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    step = max(1, n // max_points)
    sel = np.arange(0, n, step)
    pts = np.stack([ys, xs], axis=1).astype(np.float32)
    centres = pts[sel]

    # Local tangent: principal axis of skeleton neighbours within 7 px
    normals = np.zeros_like(centres)
    for i, c in enumerate(centres):
        d2 = np.sum((pts - c) ** 2, axis=1)
        nb = pts[d2 <= 49.0]
        if len(nb) < 3:
            normals[i] = (np.nan, np.nan)
            continue
        nb = nb - nb.mean(axis=0)
        _, _, vt = np.linalg.svd(nb, full_matrices=False)
        ty, tx = vt[0]
        normals[i] = (tx, -ty)           # (dy, dx) of the normal

    ok = ~np.isnan(normals[:, 0])
    return centres[ok], normals[ok]


def _profile_widths(
    gray: np.ndarray,
    skeleton: np.ndarray,
    dist_transform: np.ndarray,
    max_points: int = 600,
) -> np.ndarray:
    """
    Measure crack width at full image resolution.

    The U-Net mask is only used to find the crack centre-line.  At sampled
    centre-line points we cut a profile perpendicular to the crack in the
    ORIGINAL grayscale image and take the full width at half maximum (FWHM)
    of the dark dip, with linear interpolation -> sub-pixel width.

    Returns an array of widths in pixels (possibly empty).
    """
    from scipy.ndimage import map_coordinates

    ys, xs = np.where(skeleton > 0)
    centres, normals = _centres_and_normals(skeleton, max_points)
    if len(centres) == 0:
        return np.empty(0)

    half_mask = float(np.percentile(dist_transform[ys, xs], 95))
    R = float(np.clip(1.5 * half_mask + 4.0, 6.0, 60.0))
    t = np.arange(-R, R + 1e-6, 0.5, dtype=np.float32)
    py = centres[:, 0:1] + normals[:, 0:1] * t[None, :]
    px = centres[:, 1:2] + normals[:, 1:2] * t[None, :]
    prof = map_coordinates(gray, [py, px], order=1, mode="nearest")

    widths = []
    for row in prof:
        w = fwhm_from_profile(row, t)
        if w is not None:
            widths.append(w)
    return np.asarray(widths, dtype=np.float32)


def fwhm_from_profile(row: np.ndarray, t: np.ndarray, min_depth: float = 10.0):
    """Full width at half maximum (same units as `t`) of the dark dip in a 1-D
    intensity profile, with linear interpolation of both half-height crossings.
    Returns None when there is no clear dip. Used by the image pipeline and by the
    real-data resolution experiment (scripts/evaluate_resolution_krkCMd.py)."""
    T = len(t)
    edge = max(2, int(0.2 * T))
    lo = int(0.2 * T)
    hi = int(0.8 * T)
    imin = lo + int(np.argmin(row[lo:hi]))
    vmin = row[imin]
    base = 0.5 * (np.median(row[:edge]) + np.median(row[-edge:]))
    depth = base - vmin
    if depth < min_depth:
        return None
    half = vmin + 0.5 * depth

    i = imin
    while i > 0 and row[i] < half:
        i -= 1
    if row[i] < half:
        return None
    tl = t[i] + (row[i] - half) / max(row[i] - row[i + 1], 1e-6) * (t[i + 1] - t[i])

    j = imin
    while j < T - 1 and row[j] < half:
        j += 1
    if row[j] < half:
        return None
    tr = t[j - 1] + (half - row[j - 1]) / max(row[j] - row[j - 1], 1e-6) * (t[j] - t[j - 1])

    w = tr - tl
    return float(w) if 0.3 <= w <= 2.0 * (t[-1]) * 0.8 else None


# ─────────────────────────────────────────────────────────────────────────────
#  Photometric width + conformal interval + three-way compliance decision
# ─────────────────────────────────────────────────────────────────────────────

def crack_profiles(gray_raw: np.ndarray, skeleton: np.ndarray, pixels_per_mm: float, max_points: int = 300):
    """Profiles across the crack in the unsmoothed grey image at 1-pixel steps over +-0.8 mm
    (at least 6 px), as in the krkCMd calibration. Returns (rows, k) or None."""
    from scipy.ndimage import map_coordinates

    centres, normals = _centres_and_normals(skeleton, max_points)
    if len(centres) < 5 or pixels_per_mm <= 0:
        return None
    R = int(np.ceil(max(800.0 * pixels_per_mm / 1000.0, 6.0)))
    k = np.arange(-R, R + 1, dtype=np.float32)
    py = centres[:, 0:1] + normals[:, 0:1] * k[None, :]
    px = centres[:, 1:2] + normals[:, 1:2] * k[None, :]
    return map_coordinates(gray_raw, [py, px], order=1, mode="nearest"), k


def certified_assessment(gray_raw: np.ndarray, skeleton: np.ndarray, pixels_per_mm: float, limit_mm: float,
                         location_id: str = "", gauge_mm: float = 0.0, max_points: int = 300, **camera) -> dict:
    """All five stages (scripts/certified_width.py) on the profiles of this crack."""
    from scripts.certified_width import assess

    p = crack_profiles(gray_raw, skeleton, pixels_per_mm, max_points)
    if p is None:
        return dict(available=False, reason="too few centre-line points")
    return assess(p[0], p[1], pixels_per_mm, limit_mm, location_id=location_id, gauge_mm=gauge_mm, **camera)


def photometric_assessment(gray_raw: np.ndarray, skeleton: np.ndarray, pixels_per_mm: float,
                           limit_mm: float, max_points: int = 300) -> dict:
    """Resolution-robust width with a 90 % distribution-free interval, from the deficit area of
    profiles across the crack (scripts/photometric_width.py), and the decision
    conforms / does_not_conform / inconclusive against `limit_mm`.

    Calibration and conformal offsets come from docs/results/limits/photometric_calibration.json
    (krkCMd, leave-one-series-out validated), interpolated in log(px/mm). The interval is for the
    width at one location; it is applied at the 95th-percentile location along the crack."""
    from scipy.ndimage import map_coordinates
    from scripts.conformal_width import EPS_UM
    from scripts.photometric_width import deficit_features

    out = dict(available=False)
    cal = _load_photometric_table()
    if cal is None or pixels_per_mm <= 0:
        out["reason"] = "calibration table not found"
        return out
    grid = np.array([r["px_per_mm"] for r in cal["table"]])
    if pixels_per_mm < grid.min() * 0.95:
        out["reason"] = f"resolution {pixels_per_mm:.2f} px/mm is below the validated range ({grid.min():g} px/mm)"
        return out
    centres, normals = _centres_and_normals(skeleton, max_points)
    if len(centres) < 5:
        out["reason"] = "too few centre-line points"
        return out
    pitch_um = 1000.0 / pixels_per_mm
    R = int(np.ceil(max(800.0 / pitch_um, 6.0)))
    k = np.arange(-R, R + 1, dtype=np.float32)
    py = centres[:, 0:1] + normals[:, 0:1] * k[None, :]
    px = centres[:, 1:2] + normals[:, 1:2] * k[None, :]
    prof = map_coordinates(gray_raw, [py, px], order=1, mode="nearest")
    P = np.array([f["P"] for f in (deficit_features(r, k, pitch_um) for r in prof) if f is not None])
    if len(P) < 5:
        out["reason"] = "no usable background around the crack"
        return out

    # interpolate coefficients and conformal offsets in log(px/mm), clamped to the table
    lg = np.log(np.clip(pixels_per_mm, grid.min(), grid.max()))
    order = np.argsort(grid)

    def _interp(get):
        return float(np.interp(lg, np.log(grid[order]), np.array([get(cal["table"][i]) for i in order])))

    a = _interp(lambda r: r["photo"]["coef"][0])
    b = _interp(lambda r: r["photo"]["coef"][1])
    z_lo = _interp(lambda r: r["photo"]["z_lo"])
    z_hi = _interp(lambda r: r["photo"]["z_hi"])

    w_um = np.clip(a * P + b, 0.0, None)
    w95 = float(np.percentile(w_um, 95))
    zc = np.log(w95 + EPS_UM)
    lo, hi = max(float(np.exp(zc + z_lo)) - EPS_UM, 0.0), float(np.exp(zc + z_hi)) - EPS_UM
    L = limit_mm * 1000.0
    decision = "conforms" if hi <= L else ("does_not_conform" if lo > L else "inconclusive")
    out.update(available=True, median_mm=round(float(np.median(w_um)) / 1000, 4), p95_mm=round(w95 / 1000, 4),
               interval90_mm=[round(lo / 1000, 4), round(hi / 1000, 4)], limit_mm=limit_mm, decision=decision,
               n_profiles=int(len(P)),
               basis=("photometric width (deficit area / background), calibrated on krkCMd; 90 % conformal "
                      "interval at the 95th-percentile location; each one-sided error <= 5 % if this surface "
                      "resembles the calibration concrete"))
    return out


_PHOTOMETRIC_TABLE = None


def _load_photometric_table():
    global _PHOTOMETRIC_TABLE
    if _PHOTOMETRIC_TABLE is None:
        import json
        import os
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "docs", "results", "limits", "photometric_calibration.json")
        if os.path.exists(p):
            _PHOTOMETRIC_TABLE = json.load(open(p))
    return _PHOTOMETRIC_TABLE


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
    location_id:        str   = "",    # gauge anchoring: name of the monitored location
    gauge_reading_mm:   float = 0.0,   # gauge anchoring: reading taken at installation
    lens_focal_mm:      float = 0.0,   # re-shoot distance: lens focal length of the camera used
    pixel_size_um:      float = 0.0,   # re-shoot distance: sensor pixel size (uncropped, full resolution)
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
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
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

    # ── 4. Width at every sampled centre-line point ────────────────────────
    # Preferred: sub-pixel FWHM of the real intensity profile (full resolution).
    # Fallback : 2 x distance-transform radius of the (coarse) mask.
    skel_ys, skel_xs = np.where(skeleton > 0)
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), PREBLUR_SIGMA_PX)
    prof_widths = _profile_widths(gray, skeleton, dist_transform)
    n_sampled = min(len(skel_ys), 600)
    result.valid_profile_count = int(len(prof_widths))

    if len(prof_widths) >= 5 and len(prof_widths) >= 0.3 * n_sampled:
        widths_px = np.sqrt(np.clip(prof_widths ** 2 - PREBLUR_FWHM_PX ** 2, 0.0, None))
        result.width_method = "subpixel_profile"
    else:
        widths_px = dist_transform[skel_ys, skel_xs] * 2.0
        result.width_method = "mask_distance_transform"
        result.warnings.append(
            "Could not find a clear dark crack profile in the photo; width fell back to the "
            "coarse segmentation mask, which over-estimates hairline cracks."
        )

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
    result.scale_reliable = method in ("reference_object", "dpi")
    result.min_resolvable_width_mm = round(MIN_RELIABLE_WIDTH_PX / ppm, 4)
    if not result.scale_reliable:
        result.warnings.append(
            "Scale is only an estimate (no reference object). Absolute mm values can be off by "
            "several times. Draw a reference line over an object of known length for a real measurement."
        )
    result.resolution_ok = result.median_width_px >= MIN_RELIABLE_WIDTH_PX
    if not result.resolution_ok:
        result.warnings.append(
            f"Crack is only {result.median_width_px:.1f} px wide in this photo (needs >= "
            f"{MIN_RELIABLE_WIDTH_PX:.0f} px). Anything below ~{result.min_resolvable_width_mm:.2f} mm cannot be "
            "resolved at this scale; treat the value as an upper bound and re-shoot closer / with a macro lens."
        )

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

    # Photometric width: does not collapse when the crack is narrower than the blur
    # (docs/RESULTS_limits.md). Uses the unsmoothed grey image.
    gray_raw = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result.photometric = photometric_assessment(gray_raw, skeleton, ppm, result.exposure_limit_mm)
    result.certified = certified_assessment(gray_raw, skeleton, ppm, result.exposure_limit_mm,
                                            location_id=location_id, gauge_mm=gauge_reading_mm,
                                            focal_mm=lens_focal_mm, pixel_um=pixel_size_um)
    if result.certified.get("available"):
        # the physical limit of this photo replaces the fixed 2-pixel rule of thumb
        m = result.certified["measurability"]
        result.min_resolvable_width_mm = m["wc_mm"]
        result.warnings = [w for w in result.warnings if not w.startswith("Crack is only")]
        if not m["measurable"]:
            result.warnings.append(
                f"Crack is below the measurability limit of this photo (w_c = {m['wc_mm']} mm from its "
                f"resolution, blur and background noise {m['noise_grey']} grey levels): the width cannot be "
                f"determined, only bounded.")
    if result.photometric.get("available") and result.photometric["decision"] == "inconclusive":
        advice = "Re-shoot closer or use a crack gauge."
        try:
            from scripts.capture_planner import plan
            p = plan(result.photometric["p95_mm"], result.exposure_limit_mm, ppm, method="photo")
            result.photometric["capture_advice"] = p
            advice = p["advice"]
        except (FileNotFoundError, KeyError, ValueError):
            pass
        result.warnings.append(
            f"At this resolution the 90 % interval {result.photometric['interval90_mm']} mm straddles the "
            f"{result.exposure_limit_mm} mm limit: compliance cannot be decided from this photo. {advice}")

    # ── 7. Overlay image ───────────────────────────────────────────────────
    result.overlay_image = _draw_overlay(
        original_bgr, skeleton, dist_transform, dist_transform[skel_ys, skel_xs] * 2.0, ppm,
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
#  Skeletonization
# ─────────────────────────────────────────────────────────────────────────────

def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """
    1-pixel-wide centre-line of a binary mask (skimage Zhang-Suen style thinning).
    Returns uint8 array with skeleton pixels = 255.

    (The previous erode/dilate loop returned an empty skeleton for thin cracks.)
    """
    from skimage.morphology import skeletonize
    return skeletonize(binary > 0).astype(np.uint8) * 255


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
    location_id: str = "",
    gauge_reading_mm: float = 0.0,
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
        location_id=location_id,
        gauge_reading_mm=gauge_reading_mm,
    )
