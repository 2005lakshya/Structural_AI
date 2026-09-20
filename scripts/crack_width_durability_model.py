"""
crack_width_durability_model.py
================================
Novel method: Photo-Derived Crack Width -> Chloride-Ingress Durability Model
-> Remaining-Service-Life Impact.

Why this is the width-measurement novelty, not just "measure pixels"
---------------------------------------------------------------------
Skeleton + distance-transform crack width measurement (crack_width_measure.py)
is standard image processing — width-in-mm from a photo is not, by itself,
patentable subject matter; equivalent techniques are published prior art.

What nothing else in this codebase currently does is connect that measured
width to a *consequence*: the RUL (remaining useful life) prediction in
this app is produced by an XGBoost model trained on 9 generic tabular
inputs (train_ml.py) — it never sees the actual photographed crack. This
module closes that loop with a physically-grounded (not black-box)
chloride-diffusion durability model, so a crack-width measurement from a
photo is converted directly into a "years of service life lost" figure
for that specific structure.

Method
------
1. Chloride ingress into concrete follows Fick's second law of diffusion.
   Crank's error-function solution gives chloride concentration at depth x
   (the rebar cover) after time t:

        C(x, t) / C_s = 1 - erf( x / (2 * sqrt(D_eff * t)) )

   where C_s is the surface chloride concentration and D_eff is the
   effective chloride diffusion coefficient of the concrete.

2. A crack is a diffusion short-circuit: chlorides reach the rebar faster
   through a crack than through sound concrete. Durability literature
   consistently reports (a) a *threshold* crack width below which the
   effect is negligible, and (b) an effective-diffusion-coefficient
   increase that grows with crack width up to a saturation width, beyond
   which wider cracks don't meaningfully speed ingress further (the crack
   already acts as a direct, unobstructed path).
   `crack_width_diffusion_enhancement()` encodes that qualitative shape as
   a calibratable piecewise-linear multiplier — the specific threshold/
   saturation/max-multiplier constants below are engineering defaults, NOT
   a precise fit to any single published dataset; a production/patent-
   grade deployment should calibrate them against project-specific
   chloride profiling or accelerated ingress test data.

3. Corrosion initiates once chloride concentration at the rebar reaches a
   critical threshold (commonly cited around 0.35-0.4% by mass of cement,
   IS 456 / durability literature). Inverting Crank's solution for t gives
   the corrosion-initiation time for a given crack width, cover depth, and
   exposure severity.

4. Comparing the cracked-section initiation time against (a) the same
   section with no crack and (b) the IS 456 assumed design life (typically
   50 years) yields the two output metrics that matter for an inspection
   report: years of service life lost to this specific crack, and the
   fraction of design life already consumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from scipy.special import erfinv

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0

# Exposure-class surface chloride concentration, % by mass of cement
# (IS 456:2000 Table 3 exposure severity -> typical marine/de-icing-salt
# surface chloride loading used in durability design practice).
EXPOSURE_SURFACE_CHLORIDE_PCT = {
    "mild": 0.50,
    "moderate": 0.80,
    "severe": 1.40,
    "very_severe": 2.00,
    "extreme": 3.00,
}

CRITICAL_CHLORIDE_PCT = 0.40  # threshold chloride content that initiates rebar corrosion
BASE_DIFFUSION_M2_S = 2.5e-12  # typical D0 for good-quality OPC concrete, sound (uncracked) section


def crack_width_diffusion_enhancement(
    width_mm: float,
    threshold_mm: float = 0.05,
    saturation_mm: float = 0.40,
    max_multiplier: float = 6.0,
) -> float:
    """
    Multiplier on the baseline diffusion coefficient caused by a crack of
    this width. Below `threshold_mm` the crack is treated as not
    significantly increasing ingress rate (self-healing / capillary
    blocking dominates). Between threshold and `saturation_mm` the
    multiplier increases linearly. Above saturation the crack behaves as
    an open path and the multiplier plateaus at `max_multiplier`.
    """
    if width_mm <= threshold_mm:
        return 1.0
    if width_mm >= saturation_mm:
        return max_multiplier
    frac = (width_mm - threshold_mm) / (saturation_mm - threshold_mm)
    return 1.0 + frac * (max_multiplier - 1.0)


def _initiation_time_years(
    cover_mm: float,
    diffusion_m2_s: float,
    surface_chloride_pct: float,
    critical_chloride_pct: float = CRITICAL_CHLORIDE_PCT,
) -> float:
    """Invert Crank's error-function solution for corrosion-initiation time."""
    ratio = 1.0 - (critical_chloride_pct / max(surface_chloride_pct, 1e-6))
    ratio = min(max(ratio, -0.999), 0.999)  # keep erfinv() well-defined
    erf_arg = erfinv(ratio)
    if erf_arg <= 0:
        # Surface loading already at/below the critical threshold — no
        # ingress-driven initiation under this exposure at all.
        return float("inf")

    x_m = cover_mm / 1000.0
    t_seconds = (x_m / (2.0 * erf_arg)) ** 2 / diffusion_m2_s
    return t_seconds / SECONDS_PER_YEAR


@dataclass
class DurabilityImpactResult:
    width_mm: float
    cover_mm: float
    exposure: str
    diffusion_enhancement: float
    initiation_years_cracked: float
    initiation_years_uncracked: float
    years_lost_to_crack: float
    design_life_years: float
    life_fraction_consumed_pct: float
    remaining_life_years: float
    narrative: str = ""
    sensitivity_curve_mm: List[float] = field(default_factory=list)
    sensitivity_curve_years: List[float] = field(default_factory=list)


def estimate_service_life_impact(
    width_mm: float,
    cover_mm: float,
    exposure: str = "moderate",
    design_life_years: float = 50.0,
) -> DurabilityImpactResult:
    """
    Convert a photo-derived crack width into a durability/service-life
    impact assessment. This is the function that should be called with
    the p95 width from crack_width_measure.measure_crack_width().
    """
    surface_chloride = EXPOSURE_SURFACE_CHLORIDE_PCT.get(exposure, EXPOSURE_SURFACE_CHLORIDE_PCT["moderate"])

    enhancement = crack_width_diffusion_enhancement(width_mm)
    d_cracked = BASE_DIFFUSION_M2_S * enhancement
    d_uncracked = BASE_DIFFUSION_M2_S

    years_cracked = _initiation_time_years(cover_mm, d_cracked, surface_chloride)
    years_uncracked = _initiation_time_years(cover_mm, d_uncracked, surface_chloride)

    years_lost = 0.0 if math.isinf(years_uncracked) else max(years_uncracked - years_cracked, 0.0)

    remaining_life = design_life_years if math.isinf(years_cracked) else max(design_life_years - years_cracked, 0.0)
    if math.isinf(years_cracked):
        life_fraction_pct = 0.0
    else:
        # % of design life already "spent" by the time corrosion initiates
        life_fraction_pct = round(min(years_cracked / design_life_years, 1.0) * 100.0, 1)

    if math.isinf(years_cracked):
        narrative = (
            f"Crack width {width_mm:.3f} mm at {cover_mm:.0f} mm cover: surface chloride loading for "
            f"'{exposure}' exposure never reaches the corrosion-initiation threshold under this model — "
            f"no diffusion-driven service-life impact predicted."
        )
    else:
        narrative = (
            f"Crack width {width_mm:.3f} mm at {cover_mm:.0f} mm cover ('{exposure}' exposure): predicted "
            f"corrosion initiation in {years_cracked:.1f} yr (vs. {years_uncracked:.1f} yr uncracked) — "
            f"{years_lost:.1f} yr of service life lost to this crack, "
            f"{life_fraction_pct:.0f}% of the {design_life_years:.0f}-yr design life consumed at initiation."
        )

    # Sensitivity curve: how initiation time would change across a range of
    # widths, holding cover/exposure fixed — useful for the inspection report
    # ("here's what happens if this crack keeps widening").
    sensitivity_widths = [round(0.05 * i, 3) for i in range(1, 13)]  # 0.05mm .. 0.6mm
    sensitivity_years = []
    for w in sensitivity_widths:
        d_w = BASE_DIFFUSION_M2_S * crack_width_diffusion_enhancement(w)
        y = _initiation_time_years(cover_mm, d_w, surface_chloride)
        sensitivity_years.append(round(y, 1) if not math.isinf(y) else design_life_years * 2)

    return DurabilityImpactResult(
        width_mm=width_mm,
        cover_mm=cover_mm,
        exposure=exposure,
        diffusion_enhancement=round(enhancement, 3),
        initiation_years_cracked=round(years_cracked, 1) if not math.isinf(years_cracked) else -1.0,
        initiation_years_uncracked=round(years_uncracked, 1) if not math.isinf(years_uncracked) else -1.0,
        years_lost_to_crack=round(years_lost, 1),
        design_life_years=design_life_years,
        life_fraction_consumed_pct=life_fraction_pct,
        remaining_life_years=round(remaining_life, 1),
        narrative=narrative,
        sensitivity_curve_mm=sensitivity_widths,
        sensitivity_curve_years=sensitivity_years,
    )


if __name__ == "__main__":
    print("width_mm | initiation_yr | uncracked_yr | years_lost | %life_consumed")
    for w in (0.05, 0.10, 0.20, 0.30, 0.50, 0.80):
        r = estimate_service_life_impact(w, cover_mm=40.0, exposure="moderate")
        print(f"{w:8.2f} | {r.initiation_years_cracked:13.1f} | {r.initiation_years_uncracked:12.1f} | "
              f"{r.years_lost_to_crack:10.1f} | {r.life_fraction_consumed_pct:14.1f}")
