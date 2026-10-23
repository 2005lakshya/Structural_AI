"""
export_app_calibration.py
=========================
Deployment tables for the image pipeline (scripts/crack_width_measure.py), beyond the
photometric/fusion table written by conformal_width.py --export:

  measurability : look-up table of the measurability limit w_c (smallest width whose
                  Cramer-Rao relative sd is <= 50 %, contrast prior 30 %, blur 0.7 px) over pixel
                  pitch 3-450 um (18 values) x SNR h/sigma 2-100 (12 values)
  cnn           : CQR offsets for the resolution-conditioned network (scores from models that
                  never saw the scored series: test predictions of the leave-one-series-out folds)
  anchoring     : beta and conformal offsets of the gauge-anchored estimate w* = w - beta (w1 - g1)
                  for photo / fusion / cnn (comparator-card gauge, 0.05 mm steps)
  element       : max-statistic conformal offsets for the element maximum, and the number of
                  locations per element they were calibrated with

Everything is pooled over the four krkCMd series (phone-like imaging); the leave-one-series-out
validation of each step is in docs/RESULTS_limits.md.

Usage:
    python scripts/export_app_calibration.py --res outputs/limits --out docs/results/limits/app_calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.conformal_width import conformal_quantile, z  # noqa: E402
from scripts.element_decisions_krkCMd import element_table, load_predictions, nested_predictions  # noqa: E402
from scripts.gauge_anchoring_krkCMd import GAUGES, gauge, pairs  # noqa: E402
from scripts.width_information_limit import min_resolvable_width  # noqa: E402

PSF_FRAC, REL_TOL, H_PRIOR_CV = 0.7, 0.5, 0.3
PITCH_GRID = np.geomspace(3.0, 450.0, 18)
SNR_GRID = np.geomspace(2.0, 100.0, 12)


def measurability_lut():
    wc = np.full((len(PITCH_GRID), len(SNR_GRID)), np.nan)
    for i, p in enumerate(PITCH_GRID):
        for j, snr in enumerate(SNR_GRID):
            wc[i, j] = min_resolvable_width(p, PSF_FRAC * p, 1.0, snr, rel_tol=REL_TOL, h_prior_cv=H_PRIOR_CV,
                                            w_grid=np.geomspace(0.5, 20.0 * p + 3000.0, 220))
        print(f"  w_c @ pitch {p:.1f} um: {np.round(wc[i], 1).tolist()}", flush=True)
    return dict(pitch_um=PITCH_GRID.round(4).tolist(), snr=SNR_GRID.round(4).tolist(), wc_um=wc.round(3).tolist(),
                psf_frac=PSF_FRAC, rel_tol=REL_TOL, h_prior_cv=H_PRIOR_CV)


def cnn_offsets(res_dir, alpha, scenario):
    c = pd.read_csv(os.path.join(res_dir, "cnn_predictions.csv.gz"))
    c = c[(c.scenario == scenario) & (c.role == "test")]
    m = pd.read_csv(os.path.join(res_dir, "measurements.csv.gz"), usecols=["scenario", "px_per_mm", "i", "ref_um"])
    c = c.merge(m[m.scenario == scenario].drop(columns="scenario"), on=["px_per_mm", "i"])
    c = c[c.ref_um > 0]
    out = {}
    for res, g in c.groupby("px_per_mm"):
        s_lo = z(g.ref_um.to_numpy()) - z(g.cnn_q05.to_numpy())
        s_hi = z(g.ref_um.to_numpy()) - z(g.cnn_q95.to_numpy())
        out[float(res)] = dict(z_lo=conformal_quantile(s_lo, alpha / 2), z_hi=conformal_quantile(s_hi, 1 - alpha / 2))
    return out


def anchoring_and_element(res_dir, alpha, scenario, seed=0):
    rec, cnn = load_predictions(res_dir, scenario)
    anch, elem = {}, {}
    for res, g in rec.groupby("px_per_mm"):
        g = g.reset_index(drop=True)
        anch[float(res)], elem[float(res)] = {}, {}
        for name in ("photo", "fusion", "cnn"):
            te = pd.concat([nested_predictions(g, name, s, cnn)[0] for s in sorted(g.series.unique())])
            p = pairs(te)
            p["g1"] = gauge(p.ref1.to_numpy(), GAUGES["card_50um"], np.random.default_rng(seed))
            x, y = (p.est1 - p.g1).to_numpy(), (p.est - p.ref_um).to_numpy()
            beta = float(np.dot(x, y) / np.dot(x, x))
            a = np.clip(p.est - beta * (p.est1 - p.g1), 0, None)
            ok = (p.ref_um > 0).to_numpy()
            sc = z(p.ref_um.to_numpy()[ok]) - z(a.to_numpy()[ok])
            anch[float(res)][name] = dict(beta=beta, z_lo=conformal_quantile(sc, alpha / 2),
                                          z_hi=conformal_quantile(sc, 1 - alpha / 2), gauge="card_50um")
            e = element_table(te, "est")
            se = z(e.ref_max.to_numpy()) - z(e.est_max.to_numpy())
            elem[float(res)][name] = dict(z_lo=conformal_quantile(se, alpha / 2), z_hi=conformal_quantile(se, 1 - alpha / 2),
                                          n_locations=int(e.n.median()))
        print(f"  anchoring / element offsets @ {res:g} px/mm done", flush=True)
    return anch, elem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="outputs/limits")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--scenario", default="phone_like")
    ap.add_argument("--out", default="docs/results/limits/app_calibration.json")
    args = ap.parse_args()
    print("measurability look-up table ...", flush=True)
    lut = measurability_lut()
    print("CNN conformal offsets ...", flush=True)
    cnn = cnn_offsets(args.res, args.alpha, args.scenario)
    print("gauge anchoring and element-maximum offsets ...", flush=True)
    anch, elem = anchoring_and_element(args.res, args.alpha, args.scenario)
    table = [dict(px_per_mm=r, cnn=cnn.get(r), anchoring=anch[r], element=elem[r]) for r in sorted(anch)]
    meta = dict(source="krkCMd (Zenodo 10.5281/zenodo.14568863), phone-like imaging (PSF 0.7 px, noise 4 grey)",
                alpha=args.alpha, eps_um=20.0, measurability=lut, table=table,
                note="pooled over the four krkCMd series; leave-one-series-out validation in docs/RESULTS_limits.md")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(meta, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
