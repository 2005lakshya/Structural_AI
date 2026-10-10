"""
validate_width_synthetic.py
===========================
Controlled validation of the crack-width measurement: render cracks of KNOWN
width on a noisy concrete-like texture, blur like a camera, run the ridge
segmentation + sub-pixel profile measurement, and tabulate the error.

Usage:  python scripts/validate_width_synthetic.py [--out outputs/width_validation.csv]

Purpose: show where the method works and where it stops (width < ~2 px).
This is a synthetic benchmark, NOT a substitute for gauge-referenced real photos.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.crack_width_measure import measure_crack_width  # noqa: E402
from scripts.ridge_crack_segmentation import segment_cracks  # noqa: E402


def render(width_px: float, seed: int, size=(400, 600)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = size
    base = rng.normal(165, 14, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    ys = np.arange(w)
    cy = 200 + 40 * np.sin(ys / 90.0)
    yy = np.arange(h)[:, None]
    dist = np.abs(yy - cy[None, :])
    # Gaussian-profile dark line whose FWHM equals width_px
    sigma = max(width_px, 0.05) / 2.355
    base -= 85.0 * np.exp(-0.5 * (dist / sigma) ** 2)
    base = cv2.GaussianBlur(base, (0, 0), 0.7)            # camera PSF
    base += rng.normal(0, 3.0, base.shape)
    img = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/width_validation.csv")
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for true_w in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        est, found = [], 0
        for seed in range(args.trials):
            img = render(true_w, seed)
            mask = segment_cracks(img)
            if mask.sum() == 0:
                continue
            r = measure_crack_width(mask, img, known_length_px=1, known_length_mm=1)  # 1 px = 1 mm -> mm == px
            if r.valid_profile_count > 0 or r.median_width_px > 0:
                est.append(r.median_width_px)
                found += 1
        if est:
            m = float(np.mean(est)); sd = float(np.std(est))
            rows.append([true_w, found, round(m, 3), round(sd, 3), round(m - true_w, 3), round(100 * (m - true_w) / true_w, 1)])
        else:
            rows.append([true_w, 0, "", "", "", "not detected"])
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["true_width_px", "detected_trials", "mean_est_px", "std_px", "bias_px", "bias_pct"])
        wr.writerows(rows)
    print(f"{'true px':>8} {'found':>6} {'mean est':>9} {'std':>7} {'bias px':>8} {'bias %':>8}")
    for r in rows:
        print(f"{r[0]:>8} {r[1]:>6} {str(r[2]):>9} {str(r[3]):>7} {str(r[4]):>8} {str(r[5]):>8}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
