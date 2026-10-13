"""
make_limits_figures.py
======================
Paper figures for the information-limit study, from the tables written by
evaluate_limits_krkCMd.py, conformal_width.py, differential_width_krkCMd.py,
learned_width_estimator.py and width_information_limit.py.

    python scripts/make_limits_figures.py --res outputs/limits --out paper/figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.width_information_limit import crlb  # noqa: E402

STYLE = {  # method -> (label, colour, marker, linestyle)
    "thr_cal": ("threshold count", "0.6", "v", ":"),
    "fwhm_cal": ("sub-pixel FWHM", "C3", "o", "-"),
    "rect_cal": ("rectangle transform [Benz 2021]", "C4", "D", "--"),
    "photo_cal": ("photometric width (ours)", "C0", "s", "-"),
    "fusion_cal": ("FWHM + photometric (ours)", "C2", "^", "-"),
    "cnn": ("learned, resolution-conditioned (ours)", "k", "*", "-"),
    "cnn_samecal": ("learned, calibrated on same series (not shift-aware)", "0.5", "*", ":"),
    "photo_oracle": ("photometric, contrast oracle", "C0", "", ":"),
}


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    plt.close(fig)


def cnn_summary(res_dir):
    """MAE per resolution of the learned estimator on the held-out series (role 'test')."""
    f = os.path.join(res_dir, "cnn_predictions.csv.gz")
    if not os.path.exists(f):
        return None
    c = pd.read_csv(f)
    c = c[c.role == "test"]
    base = pd.read_csv(os.path.join(res_dir, "measurements.csv.gz"), usecols=["i", "ref_um", "cluster"]).drop_duplicates("i")
    c = c.merge(base, on="i")
    c = c[c.ref_um > 0]
    rows = []
    for (scen, res), g in c.groupby(["scenario", "px_per_mm"]):
        e = (g.cnn_q50 - g.ref_um).to_numpy()
        rows.append(dict(scenario=scen, px_per_mm=res, method="cnn", mae_um=float(np.abs(e).mean()),
                         median_rel_pct=float(np.median(np.abs(e) / g.ref_um) * 100), bias_um=float(e.mean())))
    return pd.DataFrame(rows)


def fig_mae(res_dir, out, scenario="phone_like"):
    s = pd.read_csv(os.path.join(res_dir, "summary.csv"))
    s = s[s.scenario == scenario]
    bench = json.load(open(os.path.join(res_dir, "benchmarks_native.json")))
    cs = cnn_summary(res_dir)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m, (lab, col, mk, ls) in STYLE.items():
        if m == "cnn_samecal":          # an interval-calibration variant, same point estimate as "cnn"
            continue
        if m == "cnn":
            if cs is None:
                continue
            d = cs[cs.scenario == scenario].sort_values("px_per_mm")
            ax.plot(d.px_per_mm, d.mae_um, color=col, marker=mk, ls=ls, label=lab, ms=7)
            continue
        d = s[s.method == m].sort_values("px_per_mm")
        ax.plot(d.px_per_mm, d.mae_um, color=col, marker=mk, ls=ls, label=lab, ms=4)
        if m in ("fwhm_cal", "photo_cal", "fusion_cal"):
            ax.fill_between(d.px_per_mm, d.mae_ci_lo, d.mae_ci_hi, color=col, alpha=0.12, lw=0)
    ax.axhline(bench["no_skill_loso_mean"]["mae_um"], color="0.3", lw=0.8, ls="-.", label="no skill (predict mean width)")
    ax.plot([252], [bench["dlm_um"]["mae_um"]], "x", color="C1", ms=8, label="dataset DLM (native, trained on this data)")
    ax.axhline(13.3, color="C1", lw=0.6, ls=":", label="reference noise (5 operators, RMS 16.6 um)")
    ax.set_xscale("log")
    ax.set_xlabel("image resolution (px/mm)")
    ax.set_ylabel("mean absolute error (um)")
    ax.set_ylim(0, 130)
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("krkCMd, 14,198 real profiles, leave-one-series-out, phone-like imaging", fontsize=8)
    save(fig, out, "fig_limits_mae_vs_resolution")


def fig_collapse(res_dir, out, scenario="phone_like"):
    """Relative error vs resolvability w/s_eff for edge- and deficit-based estimators, with the
    CRLB curves (contrast unknown / contrast prior) for the same imaging model."""
    r = pd.read_csv(os.path.join(res_dir, "measurements.csv.gz"),
                    usecols=["scenario", "px_per_mm", "pitch_um", "ref_um", "fwhm_cal", "photo_cal", "rect_cal"])
    r = r[(r.scenario == scenario) & (r.ref_um > 0)]
    s_eff = r.pitch_um * np.hypot(0.7, 1 / np.sqrt(12))
    r = r.assign(x=r.ref_um / s_eff)
    bins = np.geomspace(0.05, 300, 30)
    r["bin"] = pd.cut(r.x, bins)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for m in ("fwhm_cal", "rect_cal", "photo_cal"):
        lab, col, mk, ls = STYLE[m]
        g = r.dropna(subset=[m]).groupby("bin", observed=True).apply(
            lambda d: pd.Series(dict(x=d.x.median(), e=np.median(np.abs(d[m] - d.ref_um) / d.ref_um), n=len(d))),
            include_groups=False)
        g = g[g.n >= 40]
        ax.plot(g.x, g.e, color=col, marker=mk, ls=ls, ms=4, label=lab)
    # CRLB for a slot of width w at a phone-like pitch (relative sd), as a function of w/s_eff
    pitch = 125.0
    se = pitch * np.hypot(0.7, 1 / np.sqrt(12))
    ws = np.geomspace(0.05, 300, 50) * se
    cu = [np.sqrt(crlb(w, 160, pitch, 0.7 * pitch, 4.0, n_phase=8)) / w for w in ws]
    cp = [np.sqrt(crlb(w, 160, pitch, 0.7 * pitch, 4.0, h_prior_cv=0.25, n_phase=8)) / w for w in ws]
    ax.plot(ws / se, cu, "k-", lw=1, label="CRLB, contrast unknown")
    ax.plot(ws / se, cp, "k--", lw=1, label="CRLB, contrast prior (CV 25 %)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 30)
    ax.set_xlabel(r"resolvability  $w_{ref}/s_{eff}$")
    ax.set_ylabel("median relative error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5)
    save(fig, out, "fig_limits_collapse")


def fig_conformal(res_dir, out):
    cov = pd.read_csv(os.path.join(res_dir, "conformal_coverage.csv"))
    dec = pd.read_csv(os.path.join(res_dir, "compliance_decisions.csv"))
    cov, dec = cov[cov.scenario == "phone_like"], dec[dec.scenario == "phone_like"]
    meths = [m for m in ("fwhm", "rect", "photo", "fusion", "cnn", "cnn_samecal") if m in cov.method.unique()]
    key = {"fwhm": "fwhm_cal", "rect": "rect_cal", "photo": "photo_cal", "fusion": "fusion_cal", "cnn": "cnn",
           "cnn_samecal": "cnn_samecal"}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for m in meths:
        lab, col, mk, ls = STYLE[key[m]]
        d = cov[cov.method == m].sort_values("px_per_mm")
        axs[0].plot(d.px_per_mm, d.coverage, color=col, marker=mk, ls=ls, ms=4, label=lab)
        axs[1].plot(d.px_per_mm, d.median_interval_um, color=col, marker=mk, ls=ls, ms=4)
        e = dec[(dec.method == m) & (dec.limit_um == 300)].sort_values("px_per_mm")
        axs[2].plot(e.px_per_mm, e.frac_conclusive, color=col, marker=mk, ls=ls, ms=4)
    axs[0].axhline(0.9, color="0.4", lw=0.8, ls="--")
    axs[0].set_ylim(0.75, 1.0)
    axs[0].set_ylabel("coverage of 90 % interval\n(held-out series)")
    axs[1].set_ylabel("median interval width (um)")
    axs[2].set_ylabel("fraction decided at 0.3 mm limit")
    for a in axs:
        a.set_xscale("log")
        a.invert_xaxis()
        a.set_xlabel("image resolution (px/mm)")
        a.grid(True, which="both", alpha=0.25)
    axs[0].legend(fontsize=6.5)
    save(fig, out, "fig_limits_conformal")

    # false acceptance: naive point rule vs certified rule, 0.3 mm
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for m in meths:
        lab, col, mk, ls = STYLE[key[m]]
        e = dec[(dec.method == m) & (dec.limit_um == 300)].sort_values("px_per_mm")
        ax.plot(e.px_per_mm, e.naive_false_conform_given_exceed, color=col, marker=mk, ls=":", ms=4)
        ax.plot(e.px_per_mm, e.false_conform_given_exceed, color=col, marker=mk, ls="-", ms=4, label=lab)
    ax.axhline(0.05, color="0.4", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("image resolution (px/mm)")
    ax.set_ylabel("cracks > 0.3 mm declared compliant")
    ax.set_title("dotted: point estimate <= limit;  solid: conformal upper bound <= limit", fontsize=7)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5)
    save(fig, out, "fig_limits_false_accept")


def fig_healing(res_dir, out):
    h = pd.read_csv(os.path.join(res_dir, "healing_ratio.csv"))
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for m in ("fwhm_cal", "rect_cal", "photo_cal", "fusion_cal", "cnn"):
        if m not in h.method.unique():
            continue
        lab, col, mk, ls = STYLE[m]
        d = h[h.method == m].sort_values("px_per_mm")
        ax.plot(d.px_per_mm, d.hr_corr, color=col, marker=mk, ls=ls, ms=4, label=lab)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(-0.4, 1.0)
    ax.set_xlabel("image resolution (px/mm)")
    ax.set_ylabel("correlation with reference\nhealing ratio (per image & stage)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5)
    save(fig, out, "fig_limits_healing")


def fig_contrast(res_dir, out):
    r = pd.read_csv(os.path.join(res_dir, "measurements.csv.gz"),
                    usecols=["scenario", "px_per_mm", "ref_um", "P_native_um", "stage"])
    r = r[(r.scenario == "phone_like") & (r.px_per_mm == 252) & (r.ref_um > 0)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    axs[0].hexbin(r.ref_um, r.P_native_um, gridsize=60, bins="log", cmap="Greys", extent=(0, 800, 0, 1000))
    a, b = np.polyfit(r.ref_um, r.P_native_um, 1)
    xx = np.array([0, 800])
    axs[0].plot(xx, a * xx + b, "C0-", label=f"P = {a:.2f} w + {b:.0f} um")
    axs[0].plot(xx, xx, "k:", lw=0.8, label="P = w")
    axs[0].set_xlabel("reference width w (um)")
    axs[0].set_ylabel("photometric width P at 252 px/mm (um)")
    axs[0].legend(fontsize=7)
    rows = []
    for s, g in r.groupby("stage"):
        a_s, b_s = np.polyfit(g.ref_um, g.P_native_um, 1)
        rows.append((s, a_s, b_s))
    rows = np.array(rows)
    axs[1].plot([0, 2, 4, 7, 14, 28], rows[:, 1], "C0o-", label="slope a (contrast per um)")
    ax2 = axs[1].twinx()
    ax2.plot([0, 2, 4, 7, 14, 28], rows[:, 2], "C1s--", label="offset b (um)")
    axs[1].set_xlabel("self-healing time (days)")
    axs[1].set_ylabel("slope a", color="C0")
    ax2.set_ylabel("offset b (um)", color="C1")
    axs[1].set_title("P = a w + b fitted per stage", fontsize=8)
    save(fig, out, "fig_limits_contrast")
    return dict(slope=float(a), offset_um=float(b), per_stage=rows.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="outputs/limits")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_mae(a.res, a.out)
    fig_collapse(a.res, a.out)
    if os.path.exists(os.path.join(a.res, "conformal_coverage.csv")):
        fig_conformal(a.res, a.out)
    if os.path.exists(os.path.join(a.res, "healing_ratio.csv")):
        fig_healing(a.res, a.out)
    info = fig_contrast(a.res, a.out)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
