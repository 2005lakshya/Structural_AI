"""
capture_planner.py
==================
Decision-driven image acquisition: how close must a photo be taken so that the certified
compliance check (scripts/conformal_width.py) is likely to be CONCLUSIVE?

From the leave-one-series-out conformal intervals on krkCMd (outputs/limits/conformal_intervals.csv.gz)
we estimate the decidability map

    D(r, w) = P(decision is conclusive | resolution r, true width w)          (limit L fixed)

Near the limit every check is inconclusive; the "dead band" of widths around L in which
D < p shrinks as resolution increases. The planner inverts this: given an inconclusive estimate
w_hat at resolution r0, it returns the smallest resolution r with D(r, w_hat) >= p, i.e. how many
times closer to re-shoot (resolution is inversely proportional to camera distance), and with the
camera's focal length and pixel pitch, the distance itself.

CLI:
    python scripts/capture_planner.py map  --intervals outputs/limits/conformal_intervals.csv.gz
    python scripts/capture_planner.py plan --estimate-mm 0.26 --limit-mm 0.3 --px-per-mm 6 \
        [--focal-mm 6 --pixel-um 1.4]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "docs", "results", "limits", "decidability.json")
WIDTH_EDGES_UM = np.arange(0, 825, 25.0)


def decidability_map(intervals_csv: str, limits_um=(100.0, 200.0, 300.0), methods=("photo", "fusion", "cnn"),
                     scenario="phone_like"):
    d = pd.read_csv(intervals_csv)
    d = d[(d.scenario == scenario) & d.method.isin(methods)]
    centres = 0.5 * (WIDTH_EDGES_UM[1:] + WIDTH_EDGES_UM[:-1])
    out = dict(scenario=scenario, width_centres_um=centres.tolist(), maps={})
    for (m, res), g in d.groupby(["method", "px_per_mm"]):
        b = np.digitize(g.ref_um.to_numpy(), WIDTH_EDGES_UM) - 1
        for L in limits_um:
            conclusive = ((g.hi_um <= L) | (g.lo_um > L)).to_numpy()
            wrong = (((g.hi_um <= L) & (g.ref_um > L)) | ((g.lo_um > L) & (g.ref_um <= L))).to_numpy()
            p = [float(conclusive[b == k].mean()) if (b == k).sum() >= 20 else None for k in range(len(centres))]
            e = [float(wrong[b == k].mean()) if (b == k).sum() >= 20 else None for k in range(len(centres))]
            out["maps"].setdefault(m, {}).setdefault(f"{L:g}", {})[f"{res:g}"] = dict(p_conclusive=p, p_wrong=e)
    return out


def p_conclusive(table, method, limit_um, px_per_mm, width_um):
    """Interpolated D(r, w): linear in w between bin centres, linear in log r between resolutions."""
    maps = table["maps"][method][f"{limit_um:g}"]
    res = np.array(sorted(float(k) for k in maps))
    c = np.array(table["width_centres_um"])

    def at(r):
        p = np.array([np.nan if v is None else v for v in maps[f"{r:g}"]["p_conclusive"]])
        ok = np.isfinite(p)
        return float(np.interp(width_um, c[ok], p[ok]))

    r = float(np.clip(px_per_mm, res.min(), res.max()))
    j = np.searchsorted(res, r)
    if j == 0 or res[min(j, len(res) - 1)] == r:
        return at(res[min(j, len(res) - 1)])
    r0, r1 = res[j - 1], res[j]
    f = (np.log(r) - np.log(r0)) / (np.log(r1) - np.log(r0))
    return (1 - f) * at(r0) + f * at(r1)


def required_px_per_mm(table, method, limit_um, width_um, prob=0.8):
    """Smallest resolution in the validated range with D(r, w) >= prob, or None."""
    res = sorted(float(k) for k in table["maps"][method][f"{limit_um:g}"])
    grid = np.geomspace(min(res), max(res), 80)
    for r in grid:
        if p_conclusive(table, method, limit_um, r, width_um) >= prob:
            return float(r)
    return None


def plan(estimate_mm, limit_mm, px_per_mm, method="photo", prob=0.8, focal_mm=None, pixel_um=None, table=None):
    table = table or json.load(open(DEFAULT_TABLE))
    need = required_px_per_mm(table, method, limit_mm * 1000, estimate_mm * 1000, prob)
    out = dict(estimate_mm=estimate_mm, limit_mm=limit_mm, current_px_per_mm=px_per_mm, target_probability=prob,
               p_conclusive_now=round(p_conclusive(table, method, limit_mm * 1000, px_per_mm, estimate_mm * 1000), 3))
    if need is None:
        out.update(achievable=False, advice=("Too close to the limit to be decided by a photo even at 252 px/mm: "
                                             "measure this crack with a gauge or microscope."))
        return out
    factor = need / px_per_mm
    out.update(achievable=True, required_px_per_mm=round(need, 2), closer_factor=round(max(factor, 1.0), 2))
    if factor <= 1.0:
        out["advice"] = "Current resolution already decides such a crack with the target probability."
    else:
        out["advice"] = f"Re-shoot about {factor:.1f}x closer (>= {need:.1f} px/mm)."
        if focal_mm and pixel_um:
            dist_mm = focal_mm / (need * pixel_um / 1000.0)
            out["max_camera_distance_cm"] = round(dist_mm / 10.0, 1)
            out["advice"] += f" With a {focal_mm} mm lens and {pixel_um} um pixels: camera <= {dist_mm/10:.1f} cm."
    return out


def figure(table, out_dir, method="cnn", limit_um=300.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    maps = table["maps"][method][f"{limit_um:g}"]
    res = sorted((float(k) for k in maps), reverse=True)
    M = np.array([[np.nan if v is None else v for v in maps[f"{r:g}"]["p_conclusive"]] for r in res])
    c = np.array(table["width_centres_um"])
    keep = np.isfinite(M).any(axis=0)                      # drop width bins without data
    M, c = M[:, keep], c[keep]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper",
                   extent=(c[0] - 12.5, c[-1] + 12.5, len(res) - 0.5, -0.5))
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels([f"{r:g}" for r in res])
    ax.axvline(limit_um, color="w", lw=1, ls="--")
    ax.set_xlabel("true crack width (um)")
    ax.set_ylabel("image resolution (px/mm)")
    ax.set_title(f"P(conclusive certified decision), limit {limit_um/1000:g} mm, {method}", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_limits_decidability_{method}.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map")
    m.add_argument("--intervals", default="outputs/limits/conformal_intervals.csv.gz")
    m.add_argument("--out", default=DEFAULT_TABLE)
    m.add_argument("--figures", default="paper/figures")
    p = sub.add_parser("plan")
    p.add_argument("--estimate-mm", type=float, required=True)
    p.add_argument("--limit-mm", type=float, default=0.3)
    p.add_argument("--px-per-mm", type=float, required=True)
    p.add_argument("--method", default="photo")
    p.add_argument("--prob", type=float, default=0.8)
    p.add_argument("--focal-mm", type=float)
    p.add_argument("--pixel-um", type=float)
    a = ap.parse_args()
    if a.cmd == "map":
        t = decidability_map(a.intervals)
        json.dump(t, open(a.out, "w"))
        for meth in ("photo", "cnn"):
            if meth in t["maps"]:
                figure(t, a.figures, meth)
        # dead band summary: widths around 0.3 mm with P(conclusive) < 0.8
        for meth in t["maps"]:
            for r in sorted((float(k) for k in t["maps"][meth]["300"]), reverse=True):
                c = np.array(t["width_centres_um"])
                pc = np.array([np.nan if v is None else v for v in t["maps"][meth]["300"][f"{r:g}"]["p_conclusive"]])
                band = c[np.isfinite(pc) & (pc < 0.8)]
                print(f"{meth:6} {r:6g} px/mm: P<0.8 for widths {band.min():.0f}-{band.max():.0f} um" if len(band)
                      else f"{meth:6} {r:6g} px/mm: decided everywhere")
    else:
        print(json.dumps(plan(a.estimate_mm, a.limit_mm, a.px_per_mm, a.method, a.prob, a.focal_mm, a.pixel_um), indent=1))


if __name__ == "__main__":
    main()
