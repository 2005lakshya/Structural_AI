"""
evaluate_limits_krkCMd.py
=========================
Real-data test of the information-limit analysis (scripts/width_information_limit.py) on the
krkCMd profiles (Zenodo 10.5281/zenodo.14568863, CC BY 4.0): every profile is re-imaged at
2.5-252 px/mm and measured by edge-based estimators (threshold count, gradient edges,
sub-pixel FWHM) and deficit-based estimators (rectangle transform of Benz & Rodehorst 2021,
photometric width of this project), each raw and after a leave-one-series-out (LOSO) linear
calibration. A contrast oracle (true per-crack contrast from the native scan) isolates how much
of the remaining error is contrast variability.

Statistics: confidence intervals are cluster bootstraps over the 36 scanned images (profiles
of the same image share the specimen, and the 6 stages of a gridline are the same crack).

Outputs (default outputs/limits, small tables copied to docs/results/limits by --publish):
    measurements.csv.gz       one row per (scenario, resolution, profile)
    summary.csv               MAE / median / bias / within-25 % per method, with 95 % CIs
    contrast_by_bin.csv       effective contrast c = P_native / w_ref by width bin, stage, series
    sim/*.npz                 the simulated pixel rows (re-used by the learned estimator)

Usage:
    python scripts/evaluate_limits_krkCMd.py --csv data/krkCMd_raw/krkCMd_table.csv --out outputs/limits
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate_resolution_krkCMd import (NATIVE_PITCH_UM, WINDOW_HALF_UM,  # noqa: E402
                                                measure_gradient_edges, measure_ours, measure_threshold)
from scripts.photometric_width import window_half_um  # noqa: E402

RESOLUTIONS = [252.0, 100.0, 50.0, 25.0, 12.5, 8.0, 5.0, 3.6, 2.5]
SCENARIOS = {
    "ideal_scan": dict(psf_frac=0.0, noise=2.0, ramp=0.0),
    "phone_like": dict(psf_frac=0.7, noise=4.0, ramp=0.0),
    # sensitivity scenarios (run on SENS_RES only)
    "blurry": dict(psf_frac=1.5, noise=4.0, ramp=0.0),
    "noisy": dict(psf_frac=0.7, noise=10.0, ramp=0.0),
    "shading": dict(psf_frac=0.7, noise=4.0, ramp=25.0),    # +-25 grey linear ramp over +-0.8 mm
}
SENS_RES = [25.0, 8.0, 3.6]
WIDTH_BINS_UM = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 900)]
RAW_METHODS = ["thr", "grad", "fwhm", "rect", "photo"]


# ─────────────────────────────────────────────────────────────────────────────
#  Imaging simulation (vectorised version of evaluate_resolution_krkCMd.coarse_row)
# ─────────────────────────────────────────────────────────────────────────────

def coarse_rows(X: np.ndarray, pitch_um: float, psf_frac: float, noise: float, rng: np.random.Generator,
                ramp: float = 0.0):
    """Image every native profile (rows of X, pitch 3.97 um) at pixel pitch `pitch_um`:
    Gaussian PSF (sigma = psf_frac * pitch) -> area integration over pixels with a random sub-pixel
    phase per profile -> Gaussian noise -> 8-bit. Pixels outside the scan take the end value.
    Returns (rows uint8 [N, 2R+1], k)."""
    N, n = X.shape
    d = NATIVE_PITCH_UM
    prof = X.astype(np.float64)
    if ramp > 0:
        x_mm = (np.arange(n) - (n - 1) / 2) * d / 1000.0
        slope = rng.uniform(-1, 1, size=(N, 1)) * ramp / 0.8
        prof = prof + slope * x_mm[None, :]
    if psf_frac > 0:
        prof = gaussian_filter1d(prof, psf_frac * pitch_um / d, axis=1, mode="nearest")
    cum = np.concatenate([np.zeros((N, 1)), np.cumsum(prof, axis=1) * d], axis=1)   # value at edge j*d
    L = n * d
    R = int(np.ceil(max(WINDOW_HALF_UM / pitch_um, 6.0)))
    k = np.arange(-R, R + 1)
    jitter = rng.uniform(-0.5, 0.5, size=(N, 1))
    b0 = L / 2 + (k[None, :] - 0.5 + jitter) * pitch_um
    b1 = b0 + pitch_um

    def cum_at(b):
        pos = np.clip(b, 0.0, L) / d
        i = np.clip(np.floor(pos).astype(int), 0, n - 1)
        f = pos - i
        rows = np.arange(N)[:, None]
        return cum[rows, i] + f * (cum[rows, i + 1] - cum[rows, i])

    c0, c1 = cum_at(b0), cum_at(b1)
    inside = np.clip(b1, 0, L) - np.clip(b0, 0, L)
    edge_val = np.where(b0 < 0, prof[:, :1], prof[:, -1:])
    mean = np.where(inside > 0, (c1 - c0) / np.maximum(inside, 1e-9), edge_val)
    vals = mean + rng.normal(0.0, noise, size=mean.shape)
    return np.clip(np.round(vals), 0, 255).astype(np.uint8), k


def deficit_batch(rows: np.ndarray, k: np.ndarray, pitch_um: float):
    """Vectorised photometric features (same definition as photometric_width.deficit_features)."""
    g = rows.astype(np.float64)
    half = window_half_um(pitch_um)
    x = k * pitch_um
    inner = np.abs(x) <= half
    outer = ~inner
    if outer.sum() < 2:
        outer = np.zeros_like(inner)
        outer[[0, -1]] = True
        inner = ~outer
    B = np.median(g[:, outer], axis=1)
    A = np.clip(B[:, None] - g[:, inner], 0, None).sum(axis=1) * pitch_um
    depth = B - g[:, inner].min(axis=1)
    P = np.where(B > 5, A / np.maximum(B, 1e-9), np.nan)
    return P, B, depth


def rect_batch(rows: np.ndarray, pitch_um: float, cap: float = 20.0):
    g = rows.astype(np.float64)
    gmin, gmed = g.min(axis=1), np.median(g, axis=1)
    a = 0.9 * (gmed - gmin) + gmin
    b = np.minimum(gmin, cap)
    A = np.sum(a[:, None] - np.minimum(g, a[:, None]), axis=1) * pitch_um
    return np.where(a - b > 1e-6, A / np.maximum(a - b, 1e-9), np.nan)


def measure_all(rows: np.ndarray, k: np.ndarray, pitch_um: float):
    P, B, depth = deficit_batch(rows, k, pitch_um)
    out = dict(photo=P, bg=B, depth=depth, rect=rect_batch(rows, pitch_um))
    g = rows.astype(np.float64)
    thr, grad, fwhm = (np.full(len(rows), np.nan) for _ in range(3))
    for i in range(len(rows)):
        v = measure_threshold(g[i], pitch_um)
        thr[i] = np.nan if v is None else v
        v = measure_gradient_edges(g[i], pitch_um)
        grad[i] = np.nan if v is None else v
        v = measure_ours(g[i], k.astype(np.float64), pitch_um)
        fwhm[i] = np.nan if v is None else v
    out.update(thr=thr, grad=grad, fwhm=fwhm)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration, metrics, bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def loso_linear(df: pd.DataFrame, feat_cols, target="ref_um", fit_mask=None):
    """Leave-one-series-out least squares target ~ feats + 1. Rows with a missing feature get NaN.
    Fitted only on rows with fit_mask (default: ref > 0). Returns predictions for all rows."""
    pred = np.full(len(df), np.nan)
    fit_mask = (df[target] > 0).to_numpy() if fit_mask is None else fit_mask
    F = df[feat_cols].to_numpy(dtype=np.float64)
    ok = np.isfinite(F).all(axis=1)
    y = df[target].to_numpy(dtype=np.float64)
    ser = df["series"].to_numpy()
    for s in np.unique(ser):
        tr = (ser != s) & ok & fit_mask
        te = (ser == s) & ok
        Z = np.c_[F[tr], np.ones(tr.sum())]
        c = np.linalg.lstsq(Z, y[tr], rcond=None)[0]
        pred[te] = np.c_[F[te], np.ones(te.sum())] @ c
    return np.clip(pred, 0, None)


def metrics(err: np.ndarray, ref: np.ndarray):
    rel = np.abs(err) / ref
    return dict(mae_um=float(np.mean(np.abs(err))), median_abs_um=float(np.median(np.abs(err))),
                bias_um=float(np.mean(err)), median_rel_pct=float(np.median(rel) * 100),
                within_25pct=float(np.mean(rel <= 0.25)), within_50pct=float(np.mean(rel <= 0.5)))


def cluster_bootstrap_mae(err: np.ndarray, clusters: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """95 % CI of the MAE resampling whole clusters (scanned images)."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(clusters, return_inverse=True)
    s = np.bincount(inv, weights=np.abs(err), minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(os.path.join(args.out, "sim"), exist_ok=True)
    tab = pd.read_csv(args.csv)
    xcols = [c for c in tab.columns if c.startswith("x") and c[1:].isdigit()]
    X = tab[xcols].to_numpy(dtype=np.float64)
    meta = pd.DataFrame(dict(i=np.arange(len(tab)), series=tab.Series, image=tab.Image, gridline=tab.Gridline,
                             stage=tab.Stage, ref_um=tab.MANwidth, dlm_um=tab.DLMwidth, aed_um=tab.AEDwidth))
    meta["cluster"] = meta.series + "/" + meta.image.astype(str)
    scen_names = [s for s in SCENARIOS if (args.scenarios is None or s in args.scenarios)]

    # Native-resolution contrast (needed by the contrast oracle): c = P_native / ref
    rows_nat, k_nat = coarse_rows(X, NATIVE_PITCH_UM, 0.0, 0.0, np.random.default_rng(0))
    P_nat, B_nat, _ = deficit_batch(rows_nat, k_nat, NATIVE_PITCH_UM)
    meta["P_native_um"] = P_nat
    meta["c_true"] = np.where(meta.ref_um > 0, P_nat / np.maximum(meta.ref_um, 1e-9), np.nan)

    parts = []
    for si, scen in enumerate(scen_names):
        cfg = SCENARIOS[scen]
        res_list = RESOLUTIONS if scen in ("ideal_scan", "phone_like") else SENS_RES
        for res in res_list:
            pitch = 1000.0 / res
            rng = np.random.default_rng(args.seed + int(res * 10) + 1000 * si)
            rows, k = coarse_rows(X, pitch, cfg["psf_frac"], cfg["noise"], rng, cfg["ramp"])
            np.savez_compressed(os.path.join(args.out, "sim", f"{scen}_{res:g}.npz"), rows=rows, k=k,
                                pitch_um=pitch)
            m = measure_all(rows, k, pitch)
            d = meta.copy()
            d["scenario"], d["px_per_mm"], d["pitch_um"] = scen, res, pitch
            for key, v in m.items():
                d[key] = v
            # LOSO-calibrated versions of every raw estimator + the two-feature fusion
            for meth in RAW_METHODS:
                d[meth + "_cal"] = loso_linear(d, [meth])
            fus = loso_linear(d, ["fwhm", "photo"])
            d["fusion_cal"] = np.where(np.isfinite(fus), fus, d["photo_cal"])
            # contrast oracle: true per-crack contrast from the native scan (not available in practice)
            d["photo_oracle"] = d["photo"] / d["c_true"]
            parts.append(d)
            print(f"{scen} @ {res:g} px/mm done", flush=True)
    rec = pd.concat(parts, ignore_index=True)
    rec.to_csv(os.path.join(args.out, "measurements.csv.gz"), index=False)
    return rec, meta


METHOD_LABELS = {
    "thr": "threshold count", "grad": "gradient edges", "fwhm": "sub-pixel FWHM",
    "rect": "rectangle transform [Benz 2021]", "photo": "photometric width (ours)",
    "thr_cal": "threshold count, calibrated", "grad_cal": "gradient edges, calibrated",
    "fwhm_cal": "sub-pixel FWHM, calibrated", "rect_cal": "rectangle transform, calibrated",
    "photo_cal": "photometric width, calibrated (ours)", "fusion_cal": "FWHM+photometric fusion (ours)",
    "photo_oracle": "photometric, contrast oracle",
}


def summarise(rec: pd.DataFrame, out: str, n_boot: int):
    rows = []
    pos = rec[rec.ref_um > 0]
    for (scen, res), g in pos.groupby(["scenario", "px_per_mm"]):
        for meth in METHOD_LABELS:
            ok = g[meth].notna().to_numpy()
            if not ok.any():
                continue
            err = (g[meth] - g.ref_um).to_numpy()[ok]
            s = metrics(err, g.ref_um.to_numpy()[ok])
            lo, hi = cluster_bootstrap_mae(err, g.cluster.to_numpy()[ok], n_boot)
            s.update(scenario=scen, px_per_mm=res, method=meth, label=METHOD_LABELS[meth], n=int(len(g)),
                     measured_rate=float(ok.mean()), mae_ci_lo=lo, mae_ci_hi=hi)
            rows.append(s)
    summ = pd.DataFrame(rows)
    cols = ["scenario", "px_per_mm", "method", "label", "n", "measured_rate", "mae_um", "mae_ci_lo", "mae_ci_hi",
            "median_abs_um", "bias_um", "median_rel_pct", "within_25pct", "within_50pct"]
    summ = summ[cols].sort_values(["scenario", "method", "px_per_mm"], ascending=[True, True, False])
    summ.round(3).to_csv(os.path.join(out, "summary.csv"), index=False)

    # dataset's own benchmarks at native resolution (DLM was trained on this dataset)
    nat = pos.drop_duplicates("i")
    bench = {}
    for col in ("dlm_um", "aed_um"):
        e = (nat[col] - nat.ref_um).to_numpy()
        bench[col] = dict(metrics(e, nat.ref_um.to_numpy()),
                          mae_ci=cluster_bootstrap_mae(e, nat.cluster.to_numpy(), n_boot))
    w = nat.ref_um.to_numpy()
    ser = nat.series.to_numpy()
    ns = np.concatenate([np.abs(w[ser == s] - w[ser != s].mean()) for s in np.unique(ser)])
    bench["no_skill_loso_mean"] = dict(mae_um=float(ns.mean()))
    json.dump(bench, open(os.path.join(out, "benchmarks_native.json"), "w"), indent=1)

    # error by true-width bin (phone-like)
    brow = []
    ph = pos[pos.scenario == "phone_like"]
    for (res), g in ph.groupby("px_per_mm"):
        for lo, hi in WIDTH_BINS_UM:
            gb = g[(g.ref_um > lo) & (g.ref_um <= hi)]
            for meth in ("fwhm_cal", "photo_cal", "fusion_cal", "rect_cal", "photo_oracle"):
                ok = gb[meth].notna()
                rel = ((gb.loc[ok, meth] - gb.loc[ok, "ref_um"]) / gb.loc[ok, "ref_um"]).abs()
                brow.append(dict(px_per_mm=res, width_bin_um=f"{lo}-{hi}", method=meth, n=len(gb),
                                 mae_um=float((gb.loc[ok, meth] - gb.loc[ok, "ref_um"]).abs().mean()),
                                 median_rel_pct=float(rel.median() * 100)))
    pd.DataFrame(brow).round(2).to_csv(os.path.join(out, "error_by_width_bin.csv"), index=False)
    return summ, bench


def contrast_analysis(meta: pd.DataFrame, out: str):
    m = meta[(meta.ref_um > 0) & meta.c_true.notna()].copy()

    def robust(v):
        v = np.asarray(v)
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        return dict(n=int(len(v)), median=med, robust_cv=iqr / 1.349 / med, mean=float(v.mean()),
                    cv=float(v.std() / v.mean()))

    rows = [dict(group="all", key="all", **robust(m.c_true))]
    for lo, hi in WIDTH_BINS_UM:
        rows.append(dict(group="width_bin_um", key=f"{lo}-{hi}", **robust(m.c_true[(m.ref_um > lo) & (m.ref_um <= hi)])))
    for s, g in m.groupby("stage"):
        rows.append(dict(group="stage", key=str(s), **robust(g.c_true)))
    for s, g in m.groupby("series"):
        rows.append(dict(group="series", key=s, **robust(g.c_true)))
    # contrast of the same crack across stages: within-gridline spread vs between-gridline spread
    g = m.groupby(["series", "image", "gridline"]).c_true
    within = float(np.sqrt((g.var(ddof=1)).mean()) / m.c_true.mean())
    between = float(g.mean().std() / m.c_true.mean())
    rows.append(dict(group="decomposition", key="within_gridline_cv", n=int(g.ngroups), median=np.nan,
                     robust_cv=np.nan, mean=np.nan, cv=within))
    rows.append(dict(group="decomposition", key="between_gridline_cv", n=int(g.ngroups), median=np.nan,
                     robust_cv=np.nan, mean=np.nan, cv=between))
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(os.path.join(out, "contrast_by_group.csv"), index=False)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/krkCMd_raw/krkCMd_table.csv")
    ap.add_argument("--out", default="outputs/limits")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--reuse", action="store_true", help="re-summarise an existing measurements.csv.gz")
    ap.add_argument("--publish", default="docs/results/limits", help="copy small tables here ('' to skip)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.reuse:
        rec = pd.read_csv(os.path.join(args.out, "measurements.csv.gz"))
        meta = rec.drop_duplicates("i")
    else:
        rec, meta = run(args)
    summ, bench = summarise(rec, args.out, args.n_boot)
    con = contrast_analysis(meta, args.out)
    print(con.to_string(index=False))
    print(json.dumps(bench, indent=1))
    show = summ[summ.scenario == "phone_like"].pivot(index="px_per_mm", columns="method", values="mae_um")
    print(show.sort_index(ascending=False).round(1).to_string())
    if args.publish:
        os.makedirs(args.publish, exist_ok=True)
        for f in ("summary.csv", "benchmarks_native.json", "error_by_width_bin.csv", "contrast_by_group.csv"):
            shutil.copy(os.path.join(args.out, f), os.path.join(args.publish, f))


if __name__ == "__main__":
    main()
