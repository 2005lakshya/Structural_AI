"""
element_decisions_krkCMd.py
===========================
Certified compliance of a whole inspected element, not of one spot.

Codes limit the WIDEST crack on a member. A photo gives widths at many locations, so a
per-location interval applied at the worst-looking location is not a valid element-level test,
and the maximum of noisy estimates is biased upwards. Here the inspected element is one krkCMd
scanned image at one stage (216 elements, ~88 gridlines each, max reference width 0-814 um).

Decision rules compared at limits 0.3 / 0.4 / 0.5 mm, leave-one-series-out:
  naive        : max over locations of the point estimate <= L
  worst_spot   : 90 % per-location conformal interval at the location with the largest estimate
                 (what an app does when it reports one interval per photo) -- not valid for the max
  bonferroni   : every location's interval at level alpha/m; conforms if all upper bounds <= L,
                 does not conform if some lower bound > L (valid, conservative)
  max_conformal: split-conformal on the element maximum itself, score z(max ref) - z(max est),
                 calibrated on elements of other series (valid under exchangeability of elements)
Calibration scores never include the held-out series (nested LOSO for the linear estimators; the
nested models of learned_width_estimator.py --nested for the learned one).

Usage:
    python scripts/element_decisions_krkCMd.py --res outputs/limits
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.conformal_width import _apply_line, _fit_line, conformal_quantile, z, zinv  # noqa: E402

LIMITS_UM = (300.0, 400.0, 500.0)
ELEMENT = ["series", "image", "stage"]


def linear_predictions(g: pd.DataFrame, feats, fit_series, apply_mask):
    """Fit ref ~ feats on rows of `fit_series` with ref > 0; predict rows in apply_mask (NaN if a
    feature is missing). Fusion falls back to photometric where FWHM is missing."""
    F = g[feats].to_numpy(dtype=np.float64)
    y = g.ref_um.to_numpy()
    ok = np.isfinite(F).all(axis=1)
    fit = g.series.isin(fit_series).to_numpy() & ok & (y > 0)
    pred = np.full(len(g), np.nan)
    m = apply_mask & ok
    pred[m] = _apply_line(_fit_line(F[fit], y[fit]), F[m])
    if feats == ["fwhm", "photo"]:
        miss = apply_mask & ~ok
        P = g[["photo"]].to_numpy(dtype=np.float64)
        okp = np.isfinite(P[:, 0])
        fitp = g.series.isin(fit_series).to_numpy() & okp & (y > 0)
        mm = miss & okp
        pred[mm] = _apply_line(_fit_line(P[fitp], y[fitp]), P[mm])
    return pred


ESTIMATORS = {"photo": ["photo"], "fusion": ["fwhm", "photo"], "fwhm": ["fwhm"]}


def load_predictions(res_dir, scenario="phone_like"):
    """Measurements (raw features) and, if present, the learned estimator's test and nested predictions."""
    rec = pd.read_csv(os.path.join(res_dir, "measurements.csv.gz"),
                      usecols=["scenario", "px_per_mm", "i", "series", "image", "gridline", "stage", "ref_um",
                               "fwhm", "photo"])
    rec = rec[rec.scenario == scenario]
    cnn = None
    if os.path.exists(os.path.join(res_dir, "cnn_nested_cal.csv.gz")):
        c1 = pd.read_csv(os.path.join(res_dir, "cnn_predictions.csv.gz"))
        c2 = pd.read_csv(os.path.join(res_dir, "cnn_nested_cal.csv.gz"))
        cnn = pd.concat([c1[c1.role == "test"], c2])
        cnn = cnn[cnn.scenario == scenario][["px_per_mm", "i", "fold", "role", "cnn_q50"]]
    return rec, cnn


def nested_predictions(g: pd.DataFrame, name: str, s: str, cnn: pd.DataFrame | None):
    """Point estimates (column 'est', um) for held-out series s, and for every other series t from a
    model that saw neither s nor t (calibration rows). g = one resolution of the measurement table."""
    series = sorted(g.series.unique())
    if name == "cnn":
        cs = cnn[cnn.px_per_mm == g.px_per_mm.iloc[0]]
        te = g[g.series == s].merge(cs[(cs.fold == s) & (cs.role == "test")][["i", "cnn_q50"]], on="i")
        cal = g[g.series != s].merge(cs[(cs.fold == s) & (cs.role == "nested_cal")][["i", "cnn_q50"]], on="i")
        te, cal = te.rename(columns={"cnn_q50": "est"}), cal.rename(columns={"cnn_q50": "est"})
    else:
        feats = ESTIMATORS[name]
        gg = g.copy()
        gg["est"] = linear_predictions(gg, feats, [x for x in series if x != s], (gg.series == s).to_numpy())
        te = gg[gg.series == s].copy()
        parts = []
        for t in series:
            if t == s:
                continue
            h = g.copy()
            h["est"] = linear_predictions(h, feats, [x for x in series if x not in (s, t)], (h.series == t).to_numpy())
            parts.append(h[h.series == t])
        cal = pd.concat(parts)
    te["est"] = te.est.clip(lower=0)
    cal["est"] = cal.est.clip(lower=0)
    return te, cal


def element_table(d: pd.DataFrame, col: str):
    """Per element: reference max, estimated max, and the location index of the estimated max."""
    d = d[d[col].notna()]
    idx = d.groupby(ELEMENT)[col].idxmax()
    e = d.groupby(ELEMENT).agg(ref_max=("ref_um", "max"), est_max=(col, "max"), n=(col, "size"))
    e["worst_row"] = idx
    return e


def run(res_dir, alpha=0.10, scenario="phone_like"):
    rec, cnn = load_predictions(res_dir, scenario)
    rows, intervals = [], []
    for res, g in rec.groupby("px_per_mm"):
        g = g.reset_index(drop=True)
        series = sorted(g.series.unique())
        names = list(ESTIMATORS) + (["cnn"] if cnn is not None else [])
        for name in names:
            test_parts = []
            for s in series:
                te, cal = nested_predictions(g, name, s, cnn)
                et, ec = element_table(te, "est"), element_table(cal, "est")
                # max-statistic conformal
                sc = z(ec.ref_max.to_numpy()) - z(ec.est_max.to_numpy())
                et["mc_lo"] = np.clip(zinv(z(et.est_max) + conformal_quantile(sc, alpha / 2)), 0, None)
                et["mc_hi"] = zinv(z(et.est_max) + conformal_quantile(sc, 1 - alpha / 2))
                # per-location scores (for worst-spot and Bonferroni rules)
                cl = cal[cal.ref_um > 0]
                ls = z(cl.ref_um.to_numpy()) - z(cl.est.to_numpy())
                m = float(et.n.median())
                q_hi, q_lo = conformal_quantile(ls, 1 - alpha / 2), conformal_quantile(ls, alpha / 2)
                qb_hi, qb_lo = conformal_quantile(ls, 1 - alpha / (2 * m)), conformal_quantile(ls, alpha / (2 * m))
                et["ws_lo"] = np.clip(zinv(z(et.est_max) + q_lo), 0, None)
                et["ws_hi"] = zinv(z(et.est_max) + q_hi)
                # Bonferroni: max over locations of the upper bound = bound at the max estimate (monotone);
                # lower bound of the max = largest per-location lower bound = also at the max estimate.
                et["bf_lo"] = np.clip(zinv(z(et.est_max) + qb_lo), 0, None)
                et["bf_hi"] = zinv(z(et.est_max) + qb_hi)
                et["held_out"] = s
                test_parts.append(et)
            E = pd.concat(test_parts)
            E["px_per_mm"], E["method"] = res, name
            intervals.append(E.reset_index())
            for L in LIMITS_UM:
                exceed = E.ref_max > L
                for rule, lo, hi in (("naive", E.est_max, E.est_max), ("worst_spot", E.ws_lo, E.ws_hi),
                                     ("bonferroni", E.bf_lo, E.bf_hi), ("max_conformal", E.mc_lo, E.mc_hi)):
                    if rule == "naive":
                        conf, nonc = E.est_max <= L, E.est_max > L
                    else:
                        conf, nonc = hi <= L, lo > L
                    rows.append(dict(px_per_mm=res, method=name, rule=rule, limit_um=L, n_elements=len(E),
                                     n_exceeding=int(exceed.sum()),
                                     frac_conclusive=float(np.mean(conf | nonc)),
                                     false_conform_given_exceed=float(np.mean(conf[exceed])) if exceed.any() else np.nan,
                                     false_nonconform_given_ok=float(np.mean(nonc[~exceed])) if (~exceed).any() else np.nan,
                                     coverage=float(np.mean((E.ref_max >= lo) & (E.ref_max <= hi))) if rule != "naive" else np.nan))
            print(f"element decisions @ {res:g} px/mm, {name}: max MAE {np.mean(np.abs(E.est_max - E.ref_max)):.0f} um, "
                  f"bias {np.mean(E.est_max - E.ref_max):+.0f} um", flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(os.path.join(res_dir, "element_decisions.csv"), index=False)
    pd.concat(intervals).to_csv(os.path.join(res_dir, "element_intervals.csv.gz"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="outputs/limits")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--publish", default="docs/results/limits")
    a = ap.parse_args()
    out = run(a.res, a.alpha)
    pd.set_option("display.width", 220)
    t = out[(out.limit_um == 300) & out.px_per_mm.isin([252, 25, 8, 3.6])]
    print(t[["px_per_mm", "method", "rule", "frac_conclusive", "false_conform_given_exceed",
             "false_nonconform_given_ok", "coverage"]].to_string(index=False))
    if a.publish:
        shutil.copy(os.path.join(a.res, "element_decisions.csv"), os.path.join(a.publish, "element_decisions.csv"))


if __name__ == "__main__":
    main()
