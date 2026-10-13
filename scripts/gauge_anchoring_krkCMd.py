"""
gauge_anchoring_krkCMd.py
=========================
Photo monitoring anchored by ONE gauge reading per location at installation.

Finding that motivates it: the error of a photo-based width at a given crack location is
persistent over time (krkCMd: correlation ~0.75 between a stage's error and the same gridline's
error at the other stages, at every resolution). Part of it is the crack's own rim/edge geometry
(docs/RESULTS_limits.md section 3), which no image model can know.

Scenario: at installation (stage 1, day 0) an inspector reads each monitored location with a
crack comparator card; later (days 2-28) only photos are taken. The anchored estimate is

    w_s* = w_s - beta * (w_1 - g_1)

with g_1 the gauge reading, w_1 / w_s the photo estimates and beta the least-squares
regression of the later error on the installation error, fitted on OTHER specimen series.
Gauge models: exact (the operator's reference), comparator card (0.05 mm steps), card + 15 um
noise. Conformal 90 % intervals and 0.3 mm decisions are recomputed for the anchored estimate
(scores from other series, as everywhere else).

Usage:
    python scripts/gauge_anchoring_krkCMd.py --res outputs/limits
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.conformal_width import conformal_quantile, z, zinv  # noqa: E402
from scripts.element_decisions_krkCMd import ESTIMATORS, load_predictions, nested_predictions  # noqa: E402

GAUGES = {"exact": dict(step=0.0, noise=0.0), "card_50um": dict(step=50.0, noise=0.0),
          "card_50um_noise15": dict(step=50.0, noise=15.0)}
LIMIT_UM = 300.0


def gauge(ref, cfg, rng):
    g = ref + (rng.normal(0, cfg["noise"], len(ref)) if cfg["noise"] > 0 else 0.0)
    if cfg["step"] > 0:
        g = np.round(g / cfg["step"]) * cfg["step"]
    return np.clip(g, 0, None)


def pairs(d: pd.DataFrame):
    """Stage s (2-6) joined with stage 1 of the same gridline; stage-1 crack must be open and both
    photos measured (rows without an estimate are dropped for photo-only and anchored alike)."""
    key = ["series", "image", "gridline"]
    d = d[d.est.notna()]
    b = d[d.stage == 1].set_index(key)[["ref_um", "est"]].rename(columns={"ref_um": "ref1", "est": "est1"})
    j = d[d.stage > 1].set_index(key).join(b, how="inner").reset_index()
    return j[j.ref1 > 0].reset_index(drop=True)


def error_persistence(te: pd.DataFrame) -> float:
    """Mean over stages of corr(error at stage s, mean error of the same gridline at the other stages)."""
    d = te[(te.ref_um > 0) & te.est.notna()].assign(e=lambda x: x.est - x.ref_um)
    piv = d.pivot_table(index=["series", "image", "gridline"], columns="stage", values="e")
    cs = []
    for s in piv.columns:
        others = piv.drop(columns=s).mean(axis=1)
        ok = piv[s].notna() & others.notna()
        cs.append(np.corrcoef(piv[s][ok], others[ok])[0, 1])
    return float(np.nanmean(cs))


def run(res_dir, alpha=0.10, seed=0):
    rec, cnn = load_predictions(res_dir)
    rows = []
    for res, g in rec.groupby("px_per_mm"):
        g = g.reset_index(drop=True)
        series = sorted(g.series.unique())
        for name in list(ESTIMATORS)[:2] + (["cnn"] if cnn is not None else []):     # photo, fusion, cnn
            preds = {s: nested_predictions(g, name, s, cnn) for s in series}
            persistence = error_persistence(pd.concat([preds[s][0] for s in series]))
            for gname, gcfg in GAUGES.items():
                out = []
                for s in series:
                    te, cal = preds[s]
                    rng = np.random.default_rng(seed + hash((s, gname)) % 1000)
                    pt, pc = pairs(te), pairs(cal)
                    pt["g1"], pc["g1"] = gauge(pt.ref1.to_numpy(), gcfg, rng), gauge(pc.ref1.to_numpy(), gcfg, rng)
                    # beta from the calibration (other) series only
                    x, y = (pc.est1 - pc.g1).to_numpy(), (pc.est - pc.ref_um).to_numpy()
                    beta = float(np.dot(x, y) / np.dot(x, x))
                    pt["anch"] = np.clip(pt.est - beta * (pt.est1 - pt.g1), 0, None)
                    pc["anch"] = np.clip(pc.est - beta * (pc.est1 - pc.g1), 0, None)
                    for col in ("est", "anch"):
                        ok = pc.ref_um > 0
                        sc = z(pc.ref_um[ok].to_numpy()) - z(pc[col][ok].to_numpy())
                        pt[col + "_lo"] = np.clip(zinv(z(pt[col]) + conformal_quantile(sc, alpha / 2)), 0, None)
                        pt[col + "_hi"] = zinv(z(pt[col]) + conformal_quantile(sc, 1 - alpha / 2))
                    pt["beta"] = beta
                    out.append(pt)
                P = pd.concat(out)
                pos = P[P.ref_um > 0]
                exceed = pos.ref_um > LIMIT_UM
                r = dict(px_per_mm=res, method=name, gauge=gname, n_pairs=len(P), error_persistence_r=persistence,
                         beta_mean=float(P.beta.mean()),
                         mae_photo_only=float((pos.est - pos.ref_um).abs().mean()),
                         mae_anchored=float((pos.anch - pos.ref_um).abs().mean()))
                for col in ("est", "anch"):
                    lo, hi = pos[col + "_lo"], pos[col + "_hi"]
                    conf, nonc = hi <= LIMIT_UM, lo > LIMIT_UM
                    tag = "photo_only" if col == "est" else "anchored"
                    r[f"coverage_{tag}"] = float(((pos.ref_um >= lo) & (pos.ref_um <= hi)).mean())
                    r[f"median_interval_um_{tag}"] = float((hi - lo).median())
                    r[f"decided_0.3mm_{tag}"] = float((conf | nonc).mean())
                    r[f"false_conform_0.3mm_{tag}"] = float(conf[exceed].mean())
                rows.append(r)
            print(f"anchoring @ {res:g} px/mm, {name} done", flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(os.path.join(res_dir, "gauge_anchoring.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="outputs/limits")
    ap.add_argument("--publish", default="docs/results/limits")
    a = ap.parse_args()
    out = run(a.res)
    pd.set_option("display.width", 250)
    t = out[out.px_per_mm.isin([252, 25, 8, 3.6])]
    print(t[["px_per_mm", "method", "gauge", "beta_mean", "mae_photo_only", "mae_anchored", "coverage_photo_only",
             "coverage_anchored", "median_interval_um_photo_only", "median_interval_um_anchored",
             "decided_0.3mm_photo_only", "decided_0.3mm_anchored", "false_conform_0.3mm_anchored"]].to_string(index=False))
    if a.publish:
        shutil.copy(os.path.join(a.res, "gauge_anchoring.csv"), os.path.join(a.publish, "gauge_anchoring.csv"))


if __name__ == "__main__":
    main()
