"""
compare_estimators_krkCMd.py
============================
Head-to-head comparison of crack-width estimators on the krkCMd re-imaging study, including
the learned estimator: MAE per resolution and paired differences with 95 % cluster-bootstrap
confidence intervals (resampling the 36 scanned images, same resample for both methods).

    python scripts/compare_estimators_krkCMd.py --res outputs/limits
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

METHODS = ["fwhm_cal", "rect_cal", "photo_cal", "fusion_cal", "cnn"]
PAIRS = [("fusion_cal", "fwhm_cal"), ("photo_cal", "rect_cal"), ("fusion_cal", "rect_cal"),
         ("cnn", "fusion_cal"), ("cnn", "rect_cal"), ("cnn", "fwhm_cal")]


def load(res_dir, scenario):
    cols = ["scenario", "px_per_mm", "i", "series", "cluster", "stage", "ref_um"] + [m for m in METHODS if m != "cnn"]
    rec = pd.read_csv(os.path.join(res_dir, "measurements.csv.gz"), usecols=cols)
    rec = rec[(rec.scenario == scenario) & (rec.ref_um > 0)]
    f = os.path.join(res_dir, "cnn_predictions.csv.gz")
    if os.path.exists(f):
        c = pd.read_csv(f)
        c = c[(c.scenario == scenario) & (c.role == "test")][["px_per_mm", "i", "cnn_q50"]]
        rec = rec.merge(c.rename(columns={"cnn_q50": "cnn"}), on=["px_per_mm", "i"], how="left")
    return rec


def paired_ci(e1, e2, clusters, n_boot=2000, seed=0):
    """Difference of MAE (method 1 - method 2) with a cluster bootstrap over images."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(clusters, return_inverse=True)
    d = np.abs(e1) - np.abs(e2)
    s = np.bincount(inv, weights=d, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="outputs/limits")
    ap.add_argument("--scenario", default="phone_like")
    ap.add_argument("--publish", default="docs/results/limits")
    a = ap.parse_args()
    rec = load(a.res, a.scenario)
    methods = [m for m in METHODS if m in rec.columns]
    mae_rows, pair_rows = [], []
    for res, g in rec.groupby("px_per_mm"):
        ok = g[methods].notna().all(axis=1)          # same rows for every method
        h = g[ok]
        for m in methods:
            e = (h[m] - h.ref_um).to_numpy()
            mae_rows.append(dict(px_per_mm=res, method=m, n=len(h), mae_um=float(np.abs(e).mean()),
                                 median_rel_pct=float(np.median(np.abs(e) / h.ref_um) * 100),
                                 within_25pct=float(np.mean(np.abs(e) / h.ref_um <= 0.25))))
        for m1, m2 in PAIRS:
            if m1 not in methods or m2 not in methods:
                continue
            d, lo, hi = paired_ci((h[m1] - h.ref_um).to_numpy(), (h[m2] - h.ref_um).to_numpy(), h.cluster.to_numpy())
            pair_rows.append(dict(px_per_mm=res, method=m1, versus=m2, mae_diff_um=d, ci_lo=lo, ci_hi=hi,
                                  significant=bool(hi < 0 or lo > 0)))
    mae = pd.DataFrame(mae_rows).round(2)
    pairs = pd.DataFrame(pair_rows).round(2)
    mae.to_csv(os.path.join(a.res, f"compare_mae_{a.scenario}.csv"), index=False)
    pairs.to_csv(os.path.join(a.res, f"compare_pairs_{a.scenario}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(mae.pivot(index="px_per_mm", columns="method", values="mae_um").sort_index(ascending=False).to_string())
    print(pairs.sort_values(["method", "versus", "px_per_mm"], ascending=[True, True, False]).to_string(index=False))
    if a.publish:
        for f in (f"compare_mae_{a.scenario}.csv", f"compare_pairs_{a.scenario}.csv"):
            shutil.copy(os.path.join(a.res, f), os.path.join(a.publish, f))


if __name__ == "__main__":
    main()
