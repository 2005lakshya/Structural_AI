"""
evaluate_end_to_end_krkCMd.py
=============================
End-to-end (2-D) version of the krkCMd resolution study: real scanner image ->
simulated lower resolution -> crack segmentation -> skeleton -> sub-pixel width at
every annotated gridline -> compare to the manual reference width.

Unlike evaluate_resolution_krkCMd.py (1-D profiles, perfect crack localisation),
this includes segmentation and localisation errors: a gridline is a "miss" when no
skeleton point lies within 0.3 mm of the annotated crack centre.

Segmentation hyper-parameters are tuned on a DEVELOPMENT band only and then frozen
for the held-out band (a different specimen series).

Input bands come from the range-request fetcher (see docs/RESULTS_krkCMd_2D.md).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.crack_width_measure import PREBLUR_FWHM_PX, PREBLUR_SIGMA_PX, fwhm_from_profile  # noqa: E402
from scripts.ridge_crack_segmentation import segment_cracks  # noqa: E402

NATIVE_PX_PER_MM = 252.0
SCENARIOS = {"ideal_scan": dict(psf=0.0, noise=2.0), "phone_like": dict(psf=0.7, noise=4.0)}
RESOLUTIONS = [50.0, 25.0, 12.5, 8.0, 5.0, 3.6]
WINDOW_HALF_MM = 0.8


def degrade(gray, res, psf, noise, rng):
    f = NATIVE_PX_PER_MM / res
    g = gray.astype(np.float32)
    if psf > 0:
        g = cv2.GaussianBlur(g, (0, 0), psf * f)
    h, w = int(round(g.shape[0] / f)), int(round(g.shape[1] / f))
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    g = g + rng.normal(0, noise, g.shape).astype(np.float32)
    return np.clip(np.round(g), 0, 255).astype(np.uint8)


def segment_ridge(img, res, cfg):
    sig = tuple(max(0.7, s * res) for s in (0.02, 0.05, 0.10, 0.20))
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return segment_cracks(bgr, sigmas=sig, percentile=cfg["percentile"],
                          min_skeleton_px=max(6, int(0.4 * res)), min_elongation=cfg["elong"],
                          bg_sigma_px=max(5.0, 1.5 * res), min_darkness=cfg["dark"]) > 0


def segment_otsu(img, res, cfg):
    bg = cv2.GaussianBlur(img.astype(np.float32), (0, 0), max(5.0, 1.5 * res))
    sm = cv2.GaussianBlur(img.astype(np.float32) - bg, (0, 0), 0.8)
    mask = (sm < threshold_otsu(sm)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    min_area = max(10, int(0.4 * res) ** 2 // 4)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep > 0


def width_at(blurred, skel_pts, normals, centre, res):
    """Median sub-pixel FWHM width (um) at skeleton points near `centre` (x, y).
    Returns (width_um or None, hit)."""
    pitch_um = 1000.0 / res
    rad = max(0.3 * res, 3.0)
    d = np.hypot(skel_pts[:, 0] - centre[0], skel_pts[:, 1] - centre[1])
    sel = np.where(d <= rad)[0]
    if len(sel) == 0:
        return None, False
    R = max(WINDOW_HALF_MM * res, 6.0)
    t = np.arange(-R, R + 1e-6, 0.5, dtype=np.float32)
    out = []
    for i in sel[np.argsort(d[sel])][:12]:
        x, y = skel_pts[i]
        nx, ny = normals[i]
        if not np.isfinite(nx):
            continue
        row = map_coordinates(blurred, [y + ny * t, x + nx * t], order=1, mode="nearest")
        w = fwhm_from_profile(row, t)
        if w is not None:
            out.append(np.sqrt(max(w ** 2 - PREBLUR_FWHM_PX ** 2, 0.0)))
    if not out:
        return None, True
    return float(np.median(out)) * pitch_um, True


def local_normals(pts, radius):
    tree = cKDTree(pts)
    normals = np.full((len(pts), 2), np.nan)
    for i, nb in enumerate(tree.query_ball_point(pts, radius)):
        if len(nb) < 4:
            continue
        q = pts[nb] - pts[nb].mean(axis=0)
        v = np.linalg.svd(q, full_matrices=False)[2][0]
        normals[i] = (-v[1], v[0])
    return normals


def evaluate_band(npy, meta, ref, cfg, methods, resolutions, scenarios=None, seed=0):
    gray = np.load(npy)
    rows = []
    for scen, sc in SCENARIOS.items():
        if scenarios and scen not in scenarios:
            continue
        for res in resolutions:
            rng = np.random.default_rng(seed + int(res * 10))
            img = degrade(gray, res, sc["psf"], sc["noise"], rng)
            blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), PREBLUR_SIGMA_PX)
            f = NATIVE_PX_PER_MM / res
            for mname in methods:
                mask = segment_ridge(img, res, cfg) if mname == "ridge" else segment_otsu(img, res, cfg)
                ys, xs = np.nonzero(skeletonize(mask))
                pts = np.stack([xs, ys], 1).astype(np.float64)
                normals = local_normals(pts, max(3.5, 0.5 * res)) if len(pts) else pts
                for name, x1, y1, x2, y2 in meta["lines"]:
                    key = (meta["series"], meta["image"], meta["stage"], name)
                    if key not in ref:
                        continue
                    cx, cy = ((x1 + x2) / 2) / f, ((y1 + y2) / 2 - meta["row0"]) / f
                    w_um, hit = width_at(blurred, pts, normals, (cx, cy), res) if len(pts) else (None, False)
                    rows.append(dict(scenario=scen, px_per_mm=res, method=mname, series=meta["series"],
                                     stage=meta["stage"], gridline=name, ref_um=ref[key], hit=hit,
                                     meas_um=w_um))
            print(f"  {os.path.basename(npy)} {scen} {res} px/mm done", flush=True)
    return rows


def load_ref(csv):
    t = pd.read_csv(csv, usecols=["Series", "Image", "Gridline", "Stage", "MANwidth"])
    t = t[t.MANwidth > 0]
    return {(r.Series, int(r.Image), int(r.Stage), str(r.Gridline)): float(r.MANwidth) for r in t.itertuples()}


def summarize(df):
    out = []
    for (scen, res, m), g in df.groupby(["scenario", "px_per_mm", "method"]):
        ok = g.meas_um.notna()
        e = g.loc[ok, "meas_um"] - g.loc[ok, "ref_um"]
        rel = e / g.loc[ok, "ref_um"]
        out.append(dict(scenario=scen, px_per_mm=res, method=m, n=len(g), hit_rate=round(float(g.hit.mean()), 3),
                        measured_rate=round(float(ok.mean()), 3),
                        mae_um=round(float(e.abs().mean()), 1) if ok.any() else None,
                        bias_um=round(float(e.mean()), 1) if ok.any() else None,
                        median_rel_err_pct=round(float(rel.abs().median() * 100), 1) if ok.any() else None))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="krk_bands")
    ap.add_argument("--table", default="data/krkCMd_raw/krkCMd_table.csv")
    ap.add_argument("--dev-series", default="CMd_0.23_2mths")
    ap.add_argument("--out", default="outputs/krkCMd_2d")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref = load_ref(args.table)
    bands = sorted(glob.glob(os.path.join(args.bands, "*.npy")))
    metas = {b: json.load(open(b[:-4] + ".json")) for b in bands}
    dev = [b for b in bands if metas[b]["series"] == args.dev_series]
    print("bands:", [os.path.basename(b) for b in bands], "| dev:", [os.path.basename(b) for b in dev])

    # Tune ridge hyper-parameters on the development band ONLY.
    grid = [dict(percentile=p, elong=e, dark=d) for p in (94.0, 96.0, 97.5) for e in (1.5, 2.5) for d in (4.0, 8.0)]
    best, best_score, log = None, 1e18, []
    for cfg in grid:
        rows = evaluate_band(dev[0], metas[dev[0]], ref, cfg, ["ridge"], [25.0, 8.0], scenarios=["phone_like"])
        df = pd.DataFrame(rows)
        ok = df.meas_um.notna()
        mae = float((df.loc[ok, "meas_um"] - df.loc[ok, "ref_um"]).abs().mean()) if ok.any() else 1e9
        score = mae / max(float(df.hit.mean()), 0.05)
        log.append(dict(**cfg, mae_um=round(mae, 1), hit=round(float(df.hit.mean()), 3), score=round(score, 1)))
        if score < best_score:
            best, best_score = cfg, score
    json.dump(dict(best=best, log=log), open(os.path.join(args.out, "tuning.json"), "w"), indent=1)
    print("best config (dev only):", best, flush=True)

    rows = []
    for b in bands:
        rows += evaluate_band(b, metas[b], ref, best, ["ridge", "otsu"], RESOLUTIONS)
    df = pd.DataFrame(rows)
    df["split"] = np.where(df.series == args.dev_series, "dev(tuned)", "heldout")
    df.to_csv(os.path.join(args.out, "gridline_measurements.csv"), index=False)
    parts = []
    for split, g in df.groupby("split"):
        sm = summarize(g)
        sm["split"] = split
        parts.append(sm)
    summ = pd.concat(parts)
    summ.to_csv(os.path.join(args.out, "summary_2d.csv"), index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
