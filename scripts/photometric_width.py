"""
photometric_width.py
====================
Crack-width estimators that work from the brightness deficit of a profile rather than from
the position of its edges.

Why: blur and pixel integration move intensity around but conserve it, so the deficit area
A = int (B - I(x)) dx is unchanged by them (scripts/width_information_limit.py: below the
resolution limit A is the ONLY quantity about the crack that a profile still determines).
Width = A / contrast, so the estimator needs a contrast. Cracks in concrete photographed in
reflected light are close to black inside (krkCMd: median interior grey 0 at 252 px/mm), so
the contrast is taken as the local background B and the normalised deficit

    P = int max(B - I(x), 0) dx / B          [um]

is a resolution-independent width proxy. It is mapped to the reference width definition by a
linear calibration fitted on reference data (per resolution, with conformal offsets, in
docs/results/limits/photometric_calibration.json; see scripts/conformal_width.py export).

Also here, for comparison, is the closest published estimator: the rectangle transform of
Benz & Rodehorst (MVA 2021), which turns the profile into an equal-area rectangle with an
assumed bottom grey level of at most 20.
"""
from __future__ import annotations

import numpy as np


def window_half_um(pitch_um: float, psf_frac: float = 0.7, min_half_um: float = 500.0) -> float:
    """Integration half-window: at least 0.5 mm, and at least 3 effective blur radii so that
    the blurred dip of an unresolved crack is not truncated. Chosen a priori (not tuned on the
    evaluation data)."""
    s_eff = pitch_um * np.hypot(psf_frac, 1.0 / np.sqrt(12.0))
    return float(max(min_half_um, 3.0 * s_eff, 1.5 * pitch_um))


def deficit_features(row: np.ndarray, k: np.ndarray, pitch_um: float, half_um: float | None = None):
    """Background B (median of pixels outside the window), clipped deficit area A (um*grey),
    normalised deficit P = A/B (um) and depth (B - min) of one pixel profile centred on the crack
    (k = pixel index, 0 at the crack centre). Returns None when there is no usable background."""
    row = np.asarray(row, dtype=np.float64)
    half = window_half_um(pitch_um) if half_um is None else half_um
    x = np.asarray(k, dtype=np.float64) * pitch_um
    inner = np.abs(x) <= half
    outer = ~inner
    if outer.sum() < 2:                                   # window covers everything: use the end pixels
        outer = np.zeros_like(inner)
        outer[:1] = outer[-1:] = True
        inner = ~outer
    B = float(np.median(row[outer]))
    if B <= 5.0:
        return None
    d = np.clip(B - row[inner], 0.0, None)
    A = float(d.sum() * pitch_um)
    return dict(B=B, A=A, P=A / B, depth=float(B - row[inner].min()))


def photometric_width_raw(row, k, pitch_um, half_um=None):
    """Uncalibrated photometric width P = A/B in um (None if no background)."""
    f = deficit_features(row, k, pitch_um, half_um)
    return None if f is None else f["P"]


def rectangle_transform(row: np.ndarray, pitch_um: float, bottom_cap: float = 20.0):
    """Benz & Rodehorst, 'Model-based crack width estimation using rectangle transform', MVA 2021,
    eqs. (5)-(13): a_hat = 0.9*(median - min) + min, b_hat = min(min, 20), A_hat = int (a_hat -
    min(g, a_hat)) dx, w_hat = A_hat / (a_hat - b_hat). Returns um."""
    g = np.asarray(row, dtype=np.float64)
    gmin, gmed = float(g.min()), float(np.median(g))
    a = 0.9 * (gmed - gmin) + gmin
    b = min(gmin, bottom_cap)
    if a - b <= 1e-6:
        return None
    A = float(np.sum(a - np.minimum(g, a)) * pitch_um)
    return A / (a - b)


