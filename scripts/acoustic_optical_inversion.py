"""
acoustic_optical_inversion.py
=============================
Combines two MEASURED things about a crack:

  * from the photo : the surface trace of the crack and its local width
  * from UPV       : one crack depth, via IS 13311 (Part 1) / BS 1881-203

        d = x0 * sqrt((t2 / t1)^2 - 1)

Nothing is extrapolated. In particular this module does NOT estimate a depth
profile along the crack or an internal void volume: that would need a model
that has not been validated against cores or CT. If you know the void volume
(e.g. from measured injection take-up), pass it to the grouting stage yourself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

MAX_POINTS_RETURNED = 200


@dataclass
class CrackPoint:
    x_mm: float
    y_mm: float
    width_mm: float


@dataclass
class InversionResult:
    max_depth_mm: float                   # the UPV depth (measured at the transducer)
    penetration_pct: float
    surface_tortuosity: float             # skeleton length / extent diagonal (from the photo)
    acoustic_diffraction_velocity_km_s: float
    rebar_breached: bool
    neutral_axis_breached: bool
    surface_length_mm: float = 0.0
    width_profile: List[float] = field(default_factory=list)   # measured surface widths
    summary: str = ""


def extract_crack_morphology(
    skeleton: np.ndarray,
    dist_transform: np.ndarray,
    pixels_per_mm: float = 2.0,
) -> List[CrackPoint]:
    """Skeleton pixels -> list of (x_mm, y_mm, local width mm), ordered along the main axis.

    Local width = 2 x distance-transform value (inscribed-circle diameter) of the mask.
    """
    ys, xs = np.where(skeleton > 0)
    if len(ys) == 0:
        return []
    ppm = max(float(pixels_per_mm), 1e-6)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centred = pts - pts.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0] if len(pts) > 2 else np.array([1.0, 0.0])
    order = np.argsort(centred @ axis)
    widths = 2.0 * dist_transform[ys, xs] / ppm
    return [CrackPoint(float(pts[i, 0] / ppm), float(pts[i, 1] / ppm), float(widths[i])) for i in order]


def solve_optically_guided_acoustic_inversion(
    skeleton_points: List[CrackPoint],
    x0_mm: float,
    t1_us: float,
    t2_us: float,
    h_beam_mm: float,
    c_cover_mm: float,
    neutral_axis_x_mm: float = 150.0,
) -> InversionResult:
    x0 = max(x0_mm, 10.0)
    t1 = max(t1_us, 1.0)
    t2 = max(t2_us, t1)
    h = max(h_beam_mm, 50.0)

    velocity_km_s = (2.0 * x0 * 1000.0 / t1) / 1000.0
    depth = min(x0 * math.sqrt(max((t2 / t1) ** 2 - 1.0, 0.0)), h)

    tortuosity, length_mm, widths = 1.0, 0.0, []
    if skeleton_points:
        xs = np.array([p.x_mm for p in skeleton_points])
        ys = np.array([p.y_mm for p in skeleton_points])
        w = np.array([p.width_mm for p in skeleton_points])
        seg = np.hypot(np.diff(xs), np.diff(ys))
        ds = float(np.median(seg[seg > 0])) if np.any(seg > 0) else 1.0
        length_mm = len(xs) * ds
        extent = float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        tortuosity = max(length_mm / extent, 1.0) if extent > 0 else 1.0
        idx = np.linspace(0, len(w) - 1, min(len(w), MAX_POINTS_RETURNED)).astype(int)
        widths = [round(float(w[i]), 4) for i in idx]

    return InversionResult(
        max_depth_mm=round(depth, 2),
        penetration_pct=round(100.0 * depth / h, 1),
        surface_tortuosity=round(tortuosity, 3),
        acoustic_diffraction_velocity_km_s=round(velocity_km_s, 2),
        rebar_breached=depth >= c_cover_mm,
        neutral_axis_breached=depth >= (h - neutral_axis_x_mm),
        surface_length_mm=round(length_mm, 1),
        width_profile=widths,
        summary=(f"UPV crack depth {depth:.1f} mm (IS 13311). Surface trace and widths are measured "
                 "from the photo. No depth profile or void volume is estimated."),
    )
