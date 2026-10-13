"""
conformal_width.py
==================
Distribution-free prediction intervals for image-based crack width, and code-compliance
decisions that carry an error-rate guarantee.

Method (split conformal, one score per side, Mondrian by image resolution)
--------------------------------------------------------------------------
Work in z = log(w + 20 um) so that interval width scales with the crack.
For a point estimator with prediction zhat, the calibration scores are the signed residuals
e = z - zhat; the upper bound is zhat + Q_{1-a/2}(e) and the lower bound zhat + Q_{a/2}(e)
(finite-sample quantile levels). For the learned estimator's quantile outputs (CQR,
Romano et al. 2019) the scores are z - q95 and z - q05.
Each one-sided bound then holds with probability >= 1 - a/2 when calibration and test cracks
are exchangeable. Resolution is known at inspection time (from the scale), so scores are
calibrated separately per resolution.

Exchangeability between specimen series is NOT assumed: everything is evaluated
leave-one-series-out, and the calibration cracks never include the held-out series:
  * linear-calibrated estimators: nested LOSO. For held-out series s, every training series t
    is predicted by a calibration fitted on the other training series; those out-of-fold
    residuals are the conformal scores; the final calibration is fitted on all training series.
  * learned estimator: scores from the 10 % of training images held out from its training.

Decision rule against a crack-width limit L (e.g. IS 456 / EN 1992 0.1, 0.2, 0.3 mm):
    upper bound <= L  -> CONFORMS          (false-accept risk <= a/2)
    lower bound >  L  -> DOES NOT CONFORM  (false-reject risk <= a/2)
    otherwise         -> INCONCLUSIVE (re-photograph closer, or measure with a gauge)
compared with the naive rule "point estimate <= L".

Usage:
    python scripts/conformal_width.py --meas outputs/limits/measurements.csv.gz \
        --cnn outputs/limits/cnn_predictions.csv.gz --out outputs/limits
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

EPS_UM = 20.0
LIMITS_UM = (100.0, 200.0, 300.0)
POINT_METHODS = {"photo": "photometric (ours)", "fwhm": "sub-pixel FWHM", "fusion": "FWHM+photometric fusion (ours)",
                 "rect": "rectangle transform [Benz 2021]"}


def z(w):
    return np.log(np.clip(w, 0, None) + EPS_UM)


def zinv(v):
    return np.exp(v) - EPS_UM


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile at `level` (upper side); for a lower-side quantile pass
    level < 0.5 and it returns the matching lower order statistic."""
    s = np.sort(scores[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return np.nan
    if level >= 0.5:
        r = int(np.ceil((n + 1) * level))
        return float(s[min(r, n) - 1]) if r <= n else float(np.inf)
    r = int(np.floor((n + 1) * level))
    return float(s[r - 1]) if r >= 1 else float(-np.inf)


def _fit_line(F, y):
    Z = np.c_[F, np.ones(len(F))]
    return np.linalg.lstsq(Z, y, rcond=None)[0]


def _apply_line(c, F):
    return np.clip(np.c_[F, np.ones(len(F))] @ c, 0, None)


def nested_loso_intervals(g: pd.DataFrame, feats, alpha: float):
    """Intervals for every row of one (scenario, resolution) group. Returns lo, hi, point (um)."""
    lo = np.full(len(g), np.nan)
    hi, pt = lo.copy(), lo.copy()
    F = g[feats].to_numpy(dtype=np.float64)
    ok = np.isfinite(F).all(axis=1)
    y = g.ref_um.to_numpy(dtype=np.float64)
    pos = y > 0
    ser = g.series.to_numpy()
    for s in np.unique(ser):
        train = (ser != s) & ok & pos
        scores = []
        for t in np.unique(ser[train]):
            fit = train & (ser != t)
            cal = train & (ser == t)
            c = _fit_line(F[fit], y[fit])
            scores.append(z(y[cal]) - z(_apply_line(c, F[cal])))
        scores = np.concatenate(scores)
        c = _fit_line(F[train], y[train])
        te = (ser == s) & ok
        zh = z(_apply_line(c, F[te]))
        pt[te] = zinv(zh)
        lo[te] = zinv(zh + conformal_quantile(scores, alpha / 2))
        hi[te] = zinv(zh + conformal_quantile(scores, 1 - alpha / 2))
    return np.clip(lo, 0, None), hi, pt


def cqr_intervals(c: pd.DataFrame, alpha: float, cal_role: str = "cal"):
    """CQR on the learned estimator's 5/95 % quantiles for one (scenario, resolution) group of the
    CNN prediction table. Per fold (held-out series): test = role 'test' rows; calibration =
      'cal'        : training images withheld from that fold's training (same series as training:
                     does NOT represent the shift to a new specimen series), or
      'nested_cal' : predictions of models trained without the held-out series AND without the
                     calibration series, on the calibration series (shift-aware, like nested LOSO).
    Returns the test rows with lo/hi/pt columns."""
    out = []
    for fold, gg in c.groupby("fold"):
        cal = gg[(gg.role == cal_role) & (gg.ref_um > 0)]
        te = gg[gg.role == "test"].copy()
        s_hi = z(cal.ref_um.to_numpy()) - z(cal.cnn_q95.to_numpy())
        s_lo = z(cal.ref_um.to_numpy()) - z(cal.cnn_q05.to_numpy())
        te["lo"] = np.clip(zinv(z(te.cnn_q05.to_numpy()) + conformal_quantile(s_lo, alpha / 2)), 0, None)
        te["hi"] = zinv(z(te.cnn_q95.to_numpy()) + conformal_quantile(s_hi, 1 - alpha / 2))
        te["pt"] = np.clip(te.cnn_q50.to_numpy(), 0, None)
        out.append(te)
    return pd.concat(out)


def decisions(lo, hi, point, ref, L):
    conf = hi <= L
    nonc = lo > L
    inc = ~(conf | nonc)
    exceed = ref > L
    out = dict(limit_um=L, frac_conclusive=float(np.mean(~inc)),
               false_conform_given_exceed=float(np.mean(conf[exceed])) if exceed.any() else np.nan,
               false_nonconform_given_ok=float(np.mean(nonc[~exceed])) if (~exceed).any() else np.nan,
               error_among_conclusive=float(np.mean((conf & exceed) | (nonc & ~exceed)) / max(np.mean(~inc), 1e-9)))
    naive_conf = point <= L
    out.update(naive_false_conform_given_exceed=float(np.mean(naive_conf[exceed])) if exceed.any() else np.nan,
               naive_false_nonconform_given_ok=float(np.mean(~naive_conf[~exceed])) if (~exceed).any() else np.nan,
               naive_error=float(np.mean(naive_conf != ~exceed)))
    return out


def run(meas_csv, cnn_csv, out, alpha, scenarios=("phone_like", "ideal_scan")):
    rec = pd.read_csv(meas_csv)
    rec = rec[rec.scenario.isin(scenarios)].reset_index(drop=True)
    have_cnn = bool(cnn_csv) and os.path.exists(cnn_csv)
    if have_cnn:
        cnn = pd.read_csv(cnn_csv)
        nested_csv = os.path.join(os.path.dirname(cnn_csv), "cnn_nested_cal.csv.gz")
        if os.path.exists(nested_csv):
            cnn = pd.concat([cnn, pd.read_csv(nested_csv)], ignore_index=True)
        cnn = cnn[cnn.scenario.isin(scenarios)]
        base = rec.drop_duplicates("i")[["i", "series", "ref_um"]]
        cnn = cnn.merge(base, on="i", how="left")
    cov_rows, dec_rows, interval_tabs = [], [], []
    for (scen, res), g in rec.groupby(["scenario", "px_per_mm"]):
        g = g.reset_index(drop=True)
        cands = {"photo": ["photo"], "fwhm": ["fwhm"], "fusion": ["fwhm", "photo"], "rect": ["rect"]}
        ints = {}
        for name, feats in cands.items():
            lo, hi, pt = nested_loso_intervals(g, feats, alpha)
            if name == "fusion":   # where FWHM is missing fall back to photometric
                lo2, hi2, pt2 = ints["photo"]
                m = ~np.isfinite(pt)
                lo[m], hi[m], pt[m] = lo2[m], hi2[m], pt2[m]
            ints[name] = (lo, hi, pt)
        y = g.ref_um.to_numpy()
        if have_cnn:
            cg = cnn[(cnn.scenario == scen) & (cnn.px_per_mm == res)]
            for name, role in (("cnn_samecal", "cal"), ("cnn", "nested_cal")):
                if (cg.role == role).any():
                    t = cqr_intervals(cg, alpha, role).set_index("i").reindex(g.i.to_numpy())
                    ints[name] = (t.lo.to_numpy(), t.hi.to_numpy(), t.pt.to_numpy())
        for name, (lo, hi, pt) in ints.items():
            m = (y > 0) & np.isfinite(lo) & np.isfinite(hi)
            if not m.any():
                continue
            cover = (y[m] >= lo[m]) & (y[m] <= hi[m])
            row = dict(scenario=scen, px_per_mm=res, method=name, n=int(m.sum()), target_coverage=1 - alpha,
                       coverage=float(cover.mean()), below_lower=float(np.mean(y[m] < lo[m])),
                       above_upper=float(np.mean(y[m] > hi[m])),
                       median_interval_um=float(np.median(hi[m] - lo[m])),
                       median_rel_halfwidth=float(np.median((hi[m] - lo[m]) / 2 / np.maximum(y[m], 1))))
            for s in np.unique(g.series[m]):
                ms = g.series.to_numpy()[m] == s
                row[f"coverage_{s}"] = float(cover[ms].mean())
            cov_rows.append(row)
            for L in LIMITS_UM:
                dec_rows.append(dict(scenario=scen, px_per_mm=res, method=name, **decisions(lo[m], hi[m], pt[m], y[m], L)))
            interval_tabs.append(pd.DataFrame(dict(scenario=scen, px_per_mm=res, method=name, i=g.i.to_numpy()[m],
                                                   ref_um=y[m], lo_um=lo[m], hi_um=hi[m], point_um=pt[m])))
        print(f"conformal {scen} @ {res:g} done", flush=True)
    cov = pd.DataFrame(cov_rows).round(4)
    dec = pd.DataFrame(dec_rows).round(4)
    cov.to_csv(os.path.join(out, "conformal_coverage.csv"), index=False)
    dec.to_csv(os.path.join(out, "compliance_decisions.csv"), index=False)
    pd.concat(interval_tabs).to_csv(os.path.join(out, "conformal_intervals.csv.gz"), index=False)
    return cov, dec


def export_calibration(meas_csv: str, out_json: str, alpha: float, scenario: str = "phone_like"):
    """Deployment table for the image pipeline: per resolution, the linear calibration of the
    photometric width (and of the FWHM+photometric fusion) fitted on ALL krkCMd series, plus
    conformal offsets in z = log(w + 20) from leave-one-series-out residuals."""
    rec = pd.read_csv(meas_csv, usecols=["scenario", "px_per_mm", "series", "ref_um", "fwhm", "photo"])
    rec = rec[(rec.scenario == scenario) & (rec.ref_um > 0)]
    table = []
    for res, g in rec.groupby("px_per_mm"):
        row = dict(px_per_mm=float(res))
        for name, feats in (("photo", ["photo"]), ("fusion", ["fwhm", "photo"])):
            t = g.dropna(subset=feats)
            F, y, ser = t[feats].to_numpy(), t.ref_um.to_numpy(), t.series.to_numpy()
            scores = []
            for s in np.unique(ser):
                c = _fit_line(F[ser != s], y[ser != s])
                scores.append(z(y[ser == s]) - z(_apply_line(c, F[ser == s])))
            scores = np.concatenate(scores)
            c = _fit_line(F, y)
            row[name] = dict(coef=[float(v) for v in c], z_lo=conformal_quantile(scores, alpha / 2),
                             z_hi=conformal_quantile(scores, 1 - alpha / 2))
        table.append(row)
    meta = dict(source="krkCMd (Zenodo 10.5281/zenodo.14568863), phone-like imaging (PSF 0.7 px, noise 4 grey)",
                alpha=alpha, eps_um=EPS_UM, note="interval = zinv(z(point) + z_lo/z_hi); valid for dark-interior "
                "cracks on concrete; coverage was checked leave-one-series-out, not on field photographs",
                table=table)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(meta, open(out_json, "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", default="outputs/limits/measurements.csv.gz")
    ap.add_argument("--cnn", default="outputs/limits/cnn_predictions.csv.gz")
    ap.add_argument("--out", default="outputs/limits")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--publish", default="docs/results/limits")
    ap.add_argument("--export", default="docs/results/limits/photometric_calibration.json",
                    help="write the deployment calibration table here ('' to skip)")
    a = ap.parse_args()
    if a.export:
        export_calibration(a.meas, a.export, a.alpha)
    cov, dec = run(a.meas, a.cnn, a.out, a.alpha)
    pd.set_option("display.width", 200)
    print(cov[cov.scenario == "phone_like"][["px_per_mm", "method", "coverage", "below_lower", "above_upper",
                                              "median_interval_um"]].to_string(index=False))
    d = dec[(dec.scenario == "phone_like") & (dec.limit_um == 300)]
    print(d[["px_per_mm", "method", "frac_conclusive", "false_conform_given_exceed", "naive_false_conform_given_exceed",
             "error_among_conclusive", "naive_error"]].to_string(index=False))
    if a.publish:
        os.makedirs(a.publish, exist_ok=True)
        for f in ("conformal_coverage.csv", "compliance_decisions.csv"):
            shutil.copy(os.path.join(a.out, f), os.path.join(a.publish, f))


if __name__ == "__main__":
    main()
