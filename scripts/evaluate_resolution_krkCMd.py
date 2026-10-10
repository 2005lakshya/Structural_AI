"""
evaluate_resolution_krkCMd.py
=============================
Real-data resolution study for photo-based crack-width measurement.

Data: krkCMd (Zenodo 10.5281/zenodo.14568863, CC BY 4.0; Sci. Data 2025,
10.1038/s41597-025-04485-z). 19,098 brightness profiles taken across real
concrete cracks in scanner images (pixel pitch 3.96875 um, ~252 px/mm), each with
a manual reference width (MANwidth, um) from one trained operator.

Question: how does the error of the sub-pixel FWHM width measurement used in this
project grow as the photo gets coarser, and at what resolution can a given crack
width be measured?

The measurement window is a fixed +-0.8 mm (at least +-6 px), chosen without knowing
the true width, so the reference cannot leak into the measurement.

Method: for each real profile and each target resolution r (px/mm) we simulate
the camera: optional Gaussian optical PSF -> integration over square pixels of pitch
1000/r um with a random sub-pixel phase -> Gaussian sensor noise -> 8-bit
quantisation. Then the SAME width code as the image pipeline
(scripts.crack_width_measure.fwhm_from_profile, with the same 0.8 px pre-blur and
its quadrature correction) measures the width. A naive threshold-and-count
baseline (no sub-pixel interpolation) is run on the same simulated pixels.

What this does and does not show
--------------------------------
* It isolates the profile-measurement step (the resolution-limiting step). It does
  not test segmentation, rough surfaces, lighting, or perspective.
* The reference is a single operator's manual measurement, not a physical gauge.
* Coarse images are simulated from a scan; they are not real phone photographs.

Usage:
    python scripts/evaluate_resolution_krkCMd.py \
        --csv data/krkCMd_raw/krkCMd_table.csv --out outputs/krkCMd
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.crack_width_measure import PREBLUR_FWHM_PX, PREBLUR_SIGMA_PX, fwhm_from_profile  # noqa: E402

NATIVE_PITCH_UM = 3.96875
RESOLUTIONS = [252.0, 100.0, 50.0, 25.0, 12.5, 8.0, 5.0, 3.6, 2.5]   # px/mm
SCENARIOS = {
    "ideal_scan": dict(psf_frac=0.0, noise=2.0),    # no optical blur, mild noise
    "phone_like": dict(psf_frac=0.7, noise=4.0),    # PSF sigma = 0.7 px, more noise
}
WIDTH_BINS_UM = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 900)]


WINDOW_HALF_UM = 800.0   # fixed physical half-window; independent of the (unknown) true width


def coarse_row(profile: np.ndarray, pitch_um: float, psf_frac: float, noise: float,
               rng: np.random.Generator):
    """Simulate imaging one real profile at pixel pitch `pitch_um`.
    Returns (pixel_values, pixel_index_axis) centred on the crack, window +-R px."""
    n = len(profile)
    d = NATIVE_PITCH_UM
    prof = profile.astype(np.float64)
    if psf_frac > 0:
        prof = gaussian_filter1d(prof, psf_frac * pitch_um / d, mode="nearest")
    edges = np.arange(n + 1) * d
    cum = np.concatenate([[0.0], np.cumsum(prof) * d])
    L = n * d
    R = int(np.ceil(max(WINDOW_HALF_UM / pitch_um, 6.0)))
    jitter = rng.uniform(-0.5, 0.5)
    k = np.arange(-R, R + 1)
    b0 = L / 2 + (k - 0.5 + jitter) * pitch_um
    b1 = b0 + pitch_um
    c0 = np.interp(b0, edges, cum)     # np.interp clamps -> edge replication outside the scan
    c1 = np.interp(b1, edges, cum)
    # Pixels wholly outside the scan take the nearest edge value (uniform surrounding concrete).
    inside = (np.clip(b1, 0, L) - np.clip(b0, 0, L))
    mean = np.where(inside > 0, (c1 - c0) / np.maximum(inside, 1e-9), np.where(b0 < 0, prof[0], prof[-1]))
    vals = mean + rng.normal(0.0, noise, size=mean.shape)
    return np.clip(np.round(vals), 0, 255), k.astype(np.float64)


def measure_ours(row: np.ndarray, k: np.ndarray, pitch_um: float):
    """Pipeline measurement: Gaussian pre-blur, 0.5 px resampling, FWHM, quadrature correction."""
    sm = gaussian_filter1d(row, PREBLUR_SIGMA_PX, mode="nearest")
    t = np.arange(k[0], k[-1] + 1e-6, 0.5)
    prof = np.interp(t, k, sm)
    w_px = fwhm_from_profile(prof, t)   # t is symmetric about the crack centre, as in the pipeline
    if w_px is None:
        return None
    w_px = float(np.sqrt(max(w_px ** 2 - PREBLUR_FWHM_PX ** 2, 0.0)))
    return w_px * pitch_um


def measure_threshold(row: np.ndarray, pitch_um: float):
    """Baseline: count pixels darker than the half-depth level; no sub-pixel interpolation."""
    T = len(row)
    edge = max(2, int(0.2 * T))
    base = 0.5 * (np.median(row[:edge]) + np.median(row[-edge:]))
    lo, hi = int(0.2 * T), int(0.8 * T)
    vmin = row[lo:hi].min()
    depth = base - vmin
    if depth < 10.0:
        return None
    half = vmin + 0.5 * depth
    imin = lo + int(np.argmin(row[lo:hi]))
    i = imin
    while i > 0 and row[i - 1] < half:
        i -= 1
    j = imin
    while j < T - 1 and row[j + 1] < half:
        j += 1
    return (j - i + 1) * pitch_um


def measure_gradient_edges(row: np.ndarray, pitch_um: float):
    """Baseline: distance between the steepest descending and ascending edge around the
    darkest point (parabolic sub-pixel refinement). This mirrors how the reference
    widths were marked (strongest brightness gradient on each side of the crack)."""
    sm = gaussian_filter1d(row, PREBLUR_SIGMA_PX, mode="nearest")
    T = len(sm)
    lo, hi = int(0.2 * T), int(0.8 * T)
    imin = lo + int(np.argmin(sm[lo:hi]))
    edge = max(2, int(0.2 * T))
    if 0.5 * (np.median(sm[:edge]) + np.median(sm[-edge:])) - sm[imin] < 10.0:
        return None
    g = np.gradient(sm)
    if imin < 2 or imin > T - 3:
        return None
    left = int(np.argmin(g[:imin + 1]))      # steepest drop (left flank)
    right = imin + int(np.argmax(g[imin:]))   # steepest rise (right flank)

    def refine(idx, arr, sign):
        if 0 < idx < len(arr) - 1:
            a, b, c = arr[idx - 1], arr[idx], arr[idx + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-9:
                return idx + 0.5 * (a - c) / den
        return float(idx)
    xl, xr = refine(left, g, -1), refine(right, g, 1)
    w = xr - xl
    return w * pitch_um if w > 0 else None


def summarize(err, ref, found, n_total):
    err = np.asarray(err)
    ref = np.asarray(ref)
    out = {"n": int(n_total), "detect_rate": round(len(err) / max(n_total, 1), 4)}
    if len(err) == 0:
        return out
    rel = err / ref
    out.update(
        mae_um=round(float(np.mean(np.abs(err))), 2),
        median_abs_err_um=round(float(np.median(np.abs(err))), 2),
        bias_um=round(float(np.mean(err)), 2),
        median_rel_err_pct=round(float(np.median(np.abs(rel)) * 100), 1),
        within_25pct=round(float(np.mean(np.abs(rel) <= 0.25)), 3),
        within_50pct=round(float(np.mean(np.abs(rel) <= 0.50)), 3),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/krkCMd_raw/krkCMd_table.csv")
    ap.add_argument("--out", default="outputs/krkCMd")
    ap.add_argument("--max-profiles", type=int, default=0, help="0 = all (speed-up for quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    xcols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
    df = df[df["MANwidth"] > 0].reset_index(drop=True)      # w = 0 -> fully healed, no crack
    if args.max_profiles:
        df = df.sample(args.max_profiles, random_state=args.seed).reset_index(drop=True)
    X = df[xcols].to_numpy(dtype=np.float64)
    W = df["MANwidth"].to_numpy(dtype=np.float64)
    print(f"{len(df)} profiles with a measurable crack (MANwidth > 0)")

    # Published benchmarks from the dataset itself (at native resolution).
    bench = {}
    for name in ("DLMwidth", "AEDwidth"):
        v = df[name].to_numpy(dtype=np.float64)
        ok = np.isfinite(v) & (v > 0)
        bench[name] = summarize(v[ok] - W[ok], W[ok], ok, len(df))
    print("dataset benchmarks @native:", json.dumps(bench))

    records, rows = [], []
    for scen, cfg in SCENARIOS.items():
        for res in RESOLUTIONS:
            pitch = 1000.0 / res
            rng = np.random.default_rng(args.seed + int(res * 10))
            for idx in range(len(df)):
                row, k = coarse_row(X[idx], pitch, cfg["psf_frac"], cfg["noise"], rng)
                ours = measure_ours(row, k, pitch)
                thr = measure_threshold(row, pitch)
                grd = measure_gradient_edges(row, pitch)
                records.append((scen, res, idx, df.at[idx, "Series"], W[idx], ours, thr, grd))
            print(f"done {scen} @ {res} px/mm", flush=True)

    rec = pd.DataFrame(records, columns=["scenario", "px_per_mm", "i", "series", "ref_um", "ours_um", "thr_um", "grad_um"])
    rec.to_csv(os.path.join(args.out, "all_measurements.csv.gz"), index=False)

    summary = []
    for (scen, res), g in rec.groupby(["scenario", "px_per_mm"]):
        for method, col in (("subpixel_fwhm (ours)", "ours_um"), ("threshold_count (baseline)", "thr_um"),
                            ("gradient_edges (baseline)", "grad_um")):
            ok = g[col].notna()
            s = summarize((g.loc[ok, col] - g.loc[ok, "ref_um"]).to_numpy(), g.loc[ok, "ref_um"].to_numpy(), ok, len(g))
            s.update(scenario=scen, px_per_mm=res, method=method, pitch_um=round(1000 / res, 1))
            summary.append(s)
    summ = pd.DataFrame(summary)[["scenario", "method", "px_per_mm", "pitch_um", "n", "detect_rate", "mae_um",
                                  "median_abs_err_um", "bias_um", "median_rel_err_pct", "within_25pct", "within_50pct"]]
    summ = summ.sort_values(["scenario", "method", "px_per_mm"], ascending=[True, True, False])
    summ.to_csv(os.path.join(args.out, "summary_by_resolution.csv"), index=False)

    # Error by true-width bin (ours): what width is resolvable at which resolution?
    rows = []
    for (scen, res), g in rec.groupby(["scenario", "px_per_mm"]):
        for lo, hi in WIDTH_BINS_UM:
            gb = g[(g.ref_um > lo) & (g.ref_um <= hi)]
            ok = gb.ours_um.notna()
            rel = ((gb.loc[ok, "ours_um"] - gb.loc[ok, "ref_um"]) / gb.loc[ok, "ref_um"]).abs()
            rows.append(dict(scenario=scen, px_per_mm=res, width_bin_um=f"{lo}-{hi}", n=len(gb),
                             detect_rate=round(ok.mean(), 3) if len(gb) else None,
                             median_rel_err_pct=round(float(rel.median() * 100), 1) if ok.any() else None,
                             within_50pct=round(float((rel <= 0.5).mean()), 3) if ok.any() else None))
    bins = pd.DataFrame(rows)
    bins.to_csv(os.path.join(args.out, "error_by_width_bin.csv"), index=False)

    # Leave-one-series-out calibration: fit ref ~ a*measured + b on 3 series, test on the 4th.
    cal = []
    for (scen, res), g in rec.groupby(["scenario", "px_per_mm"]):
        g = g[g.ours_um.notna()]
        errs_raw, errs_cal = [], []
        for held in g.series.unique():
            tr, te = g[g.series != held], g[g.series == held]
            if len(tr) < 50 or len(te) == 0:
                continue
            a, b = np.polyfit(tr.ours_um, tr.ref_um, 1)
            errs_raw.append(np.abs(te.ours_um - te.ref_um).to_numpy())
            errs_cal.append(np.abs(a * te.ours_um + b - te.ref_um).to_numpy())
        if errs_raw:
            cal.append(dict(scenario=scen, px_per_mm=res,
                            mae_raw_um=round(float(np.concatenate(errs_raw).mean()), 2),
                            mae_calibrated_loso_um=round(float(np.concatenate(errs_cal).mean()), 2)))
    pd.DataFrame(cal).to_csv(os.path.join(args.out, "calibration_leave_one_series_out.csv"), index=False)

    # Per-series robustness at a phone-like resolution.
    ser = []
    for (scen, res, series), g in rec[(rec.px_per_mm == 8.0) & (rec.scenario == "phone_like")].groupby(["scenario", "px_per_mm", "series"]):
        ok = g.ours_um.notna()
        s = summarize((g.loc[ok, "ours_um"] - g.loc[ok, "ref_um"]).to_numpy(), g.loc[ok, "ref_um"].to_numpy(), ok, len(g))
        s["series"] = series
        ser.append(s)
    pd.DataFrame(ser).to_csv(os.path.join(args.out, "per_series_8pxmm_phone_like.csv"), index=False)

    # Bootstrap CI on MAE (resampling profiles) for the headline rows.
    rng = np.random.default_rng(123)
    ci_rows = []
    for (scen, res), g in rec.groupby(["scenario", "px_per_mm"]):
        g = g[g.ours_um.notna()]
        e = (g.ours_um - g.ref_um).abs().to_numpy()
        if len(e) == 0:
            continue
        boots = [rng.choice(e, len(e)).mean() for _ in range(300)]
        ci_rows.append(dict(scenario=scen, px_per_mm=res, mae_um=round(e.mean(), 2),
                            ci95_lo=round(float(np.percentile(boots, 2.5)), 2),
                            ci95_hi=round(float(np.percentile(boots, 97.5)), 2)))
    pd.DataFrame(ci_rows).to_csv(os.path.join(args.out, "mae_bootstrap_ci.csv"), index=False)
    json.dump(bench, open(os.path.join(args.out, "dataset_benchmarks_native.json"), "w"), indent=1)

    # Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, scen in zip(axes, SCENARIOS):
        for method, style in (("subpixel_fwhm (ours)", "o-"), ("threshold_count (baseline)", "s--"),
                              ("gradient_edges (baseline)", "^:")):
            d = summ[(summ.scenario == scen) & (summ.method == method)].sort_values("px_per_mm")
            ax.plot(d.px_per_mm, d.mae_um, style, label=method)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("image resolution (px/mm)"); ax.set_title(scen.replace("_", " "))
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("mean absolute error (um)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_mae_vs_resolution.png"), dpi=200); plt.close(fig)

    piv = bins[bins.scenario == "phone_like"].pivot(index="width_bin_um", columns="px_per_mm", values="median_rel_err_pct")
    piv = piv.reindex([f"{lo}-{hi}" for lo, hi in WIDTH_BINS_UM])[sorted(piv.columns, reverse=True)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis_r", vmin=0, vmax=200)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for (i, j), v in np.ndenumerate(piv.to_numpy(dtype=float)):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xlabel("image resolution (px/mm)"); ax.set_ylabel("true width bin (um)")
    ax.set_title("Median relative error (%), phone-like imaging")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_error_heatmap.png"), dpi=200); plt.close(fig)

    print(summ.to_string(index=False))
    print("\nsaved to", args.out)


if __name__ == "__main__":
    main()
