"""
differential_width_krkCMd.py
============================
Can a coarse photo track how a crack CHANGES (self-healing, opening) even where it cannot
measure the absolute width?

For every krkCMd gridline that is open at stage 1 (t = 0 d) we compare stage s (2, 4, 7, 14,
28 d) with stage 1. Each stage is imaged independently (own sub-pixel phase and noise, as two
photos taken weeks apart would be), measured, and calibrated leave-one-series-out
(scripts/evaluate_limits_krkCMd.py). Fully healed later stages (reference width 0) are kept:
they are part of the healing signal.

Reported per resolution and estimator:
  * MAE of the change dw = w_s - w_1 vs MAE of the absolute width (does the bias cancel?)
  * sign agreement of the change when |dw_ref| >= 50 um
  * mean healing ratio per scanned image and stage, HR = 1 - mean(w_s)/mean(w_1), the quantity
    self-healing studies report, and its error against the reference.

Usage:
    python scripts/differential_width_krkCMd.py --meas outputs/limits/measurements.csv.gz \
        --cnn outputs/limits/cnn_predictions.csv.gz --out outputs/limits
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

METHODS = {"fwhm_cal": "sub-pixel FWHM, cal.", "rect_cal": "rectangle transform, cal.",
           "photo_cal": "photometric (ours)", "fusion_cal": "FWHM+photometric fusion (ours)",
           "cnn": "learned (ours)"}
STAGE_DAYS = {1: 0, 2: 2, 3: 4, 4: 7, 5: 14, 6: 28}


def load(meas, cnn, scenario):
    rec = pd.read_csv(meas)
    rec = rec[rec.scenario == scenario]
    if cnn and os.path.exists(cnn):
        c = pd.read_csv(cnn)
        c = c[(c.scenario == scenario) & (c.role == "test")][["px_per_mm", "i", "cnn_q50"]]
        rec = rec.merge(c.rename(columns={"cnn_q50": "cnn"}), on=["px_per_mm", "i"], how="left")
    return rec


def pairs(rec):
    key = ["px_per_mm", "series", "image", "gridline"]
    base = rec[rec.stage == 1].set_index(key)
    later = rec[rec.stage > 1].set_index(key)
    j = later.join(base, rsuffix="_s1", how="inner")
    j = j[j.ref_um_s1 > 0].reset_index()
    return j


def run(meas, cnn, out, scenario="phone_like"):
    rec = load(meas, cnn, scenario)
    methods = [m for m in METHODS if m in rec.columns]
    P = pairs(rec)
    rows, hr_rows = [], []
    for res, g in P.groupby("px_per_mm"):
        dref = (g.ref_um - g.ref_um_s1).to_numpy()
        big = np.abs(dref) >= 50
        for m in methods:
            est, est1 = g[m].to_numpy(), g[m + "_s1"].to_numpy()
            ok = np.isfinite(est) & np.isfinite(est1)
            d = est - est1
            a = rec[(rec.px_per_mm == res) & (rec.ref_um > 0) & rec[m].notna()]
            rows.append(dict(px_per_mm=res, method=m, label=METHODS[m], n_pairs=int(ok.sum()),
                             mae_abs_um=float((a[m] - a.ref_um).abs().mean()),
                             mae_change_um=float(np.mean(np.abs(d[ok] - dref[ok]))),
                             bias_change_um=float(np.mean(d[ok] - dref[ok])),
                             sd_change_err_um=float(np.std(d[ok] - dref[ok])),
                             sign_agree_big=float(np.mean(np.sign(d[ok & big]) == np.sign(dref[ok & big]))),
                             corr_change=float(np.corrcoef(d[ok], dref[ok])[0, 1])))
            # healing ratio per image and stage
            gg = g[ok].copy()
            gg["est"], gg["est1"] = est[ok], est1[ok]
            agg = gg.groupby(["series", "image", "stage"]).agg(ref=("ref_um", "mean"), ref1=("ref_um_s1", "mean"),
                                                              e=("est", "mean"), e1=("est1", "mean"), n=("est", "size"))
            agg = agg[agg.n >= 10]
            hr_ref = 1 - agg.ref / agg.ref1
            hr_est = 1 - agg.e / agg.e1
            hr_rows.append(dict(px_per_mm=res, method=m, n_image_stages=int(len(agg)),
                                hr_mae_pct=float(np.mean(np.abs(hr_est - hr_ref)) * 100),
                                hr_bias_pct=float(np.mean(hr_est - hr_ref) * 100),
                                hr_corr=float(np.corrcoef(hr_est, hr_ref)[0, 1]) if len(agg) > 2 else np.nan))
        print(f"differential @ {res:g} done", flush=True)
    t = pd.DataFrame(rows).round(3)
    h = pd.DataFrame(hr_rows).round(3)
    t.to_csv(os.path.join(out, "differential_change.csv"), index=False)
    h.to_csv(os.path.join(out, "healing_ratio.csv"), index=False)
    # reference healing curve (mean width ratio vs day) for context
    ref = P[P.px_per_mm == P.px_per_mm.max()].groupby("stage").apply(
        lambda d: pd.Series(dict(day=STAGE_DAYS[d.name], mean_ratio=float(d.ref_um.mean() / d.ref_um_s1.mean()),
                                 frac_fully_healed=float((d.ref_um == 0).mean()), n=len(d))))
    ref.to_csv(os.path.join(out, "healing_reference_curve.csv"))
    return t, h, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", default="outputs/limits/measurements.csv.gz")
    ap.add_argument("--cnn", default="outputs/limits/cnn_predictions.csv.gz")
    ap.add_argument("--out", default="outputs/limits")
    ap.add_argument("--publish", default="docs/results/limits")
    a = ap.parse_args()
    t, h, ref = run(a.meas, a.cnn, a.out)
    pd.set_option("display.width", 200)
    print(ref.to_string())
    print(t[["px_per_mm", "method", "mae_abs_um", "mae_change_um", "bias_change_um", "sign_agree_big", "corr_change"]]
          .sort_values(["method", "px_per_mm"], ascending=[True, False]).to_string(index=False))
    print(h.sort_values(["method", "px_per_mm"], ascending=[True, False]).to_string(index=False))
    if a.publish:
        for f in ("differential_change.csv", "healing_ratio.csv", "healing_reference_curve.csv"):
            shutil.copy(os.path.join(a.out, f), os.path.join(a.publish, f))


if __name__ == "__main__":
    main()
