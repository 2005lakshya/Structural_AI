"""
evaluate_limits_2d_krkCMd.py
============================
End-to-end (2-D) check of the deficit-based and learned estimators on real scanner images of
held-out specimens: scan (252 px/mm) -> simulated phone-like image at 3.6-25 px/mm -> ridge
segmentation -> skeleton -> profile along the local normal at every annotated gridline ->
width by FWHM, rectangle transform, photometric width, fusion and the learned estimator.

Nothing is tuned on these images:
  * ridge-filter settings are the frozen development-split values from the earlier 2-D study
    (docs/results/2d/tuning.json, tuned on series CMd_0.23_2mths, which is not used here);
  * linear calibrations come from the 1-D study (outputs/limits/measurements.csv.gz), fitted on
    the three series OTHER than the image's own series, at the same resolution;
  * the learned estimator is the fold model trained without the image's series.

Bands are fetched with scripts/fetch_krkCMd_band.py (HTTP range requests into the Zenodo zip).

Usage:
    python scripts/evaluate_limits_2d_krkCMd.py --bands krk_bands --out outputs/limits_2d
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate_end_to_end_krkCMd import (NATIVE_PX_PER_MM, degrade, load_ref,  # noqa: E402
                                                local_normals, segment_ridge)
from scripts.evaluate_resolution_krkCMd import WINDOW_HALF_UM, measure_ours  # noqa: E402
from scripts.photometric_width import deficit_features, rectangle_transform  # noqa: E402

RESOLUTIONS = [25.0, 12.5, 8.0, 5.0, 3.6]
PSF, NOISE = 0.7, 4.0   # phone-like scenario, as in the 1-D study


def calibrations(meas_csv: str, scenario: str = "phone_like"):
    """{(series_held_out, px_per_mm): {method: coef}} from the 1-D measurements."""
    rec = pd.read_csv(meas_csv, usecols=["scenario", "px_per_mm", "series", "ref_um", "fwhm", "rect", "photo"])
    rec = rec[(rec.scenario == scenario) & (rec.ref_um > 0)]
    out = {}
    for res, g in rec.groupby("px_per_mm"):
        for s in g.series.unique():
            tr = g[g.series != s]
            c = {}
            for name, feats in (("fwhm", ["fwhm"]), ("rect", ["rect"]), ("photo", ["photo"]), ("fusion", ["fwhm", "photo"])):
                t = tr.dropna(subset=feats)
                Z = np.c_[t[feats].to_numpy(), np.ones(len(t))]
                c[name] = np.linalg.lstsq(Z, t.ref_um.to_numpy(), rcond=None)[0]
            out[(s, res)] = c
    return out


def apply_cal(c, *vals):
    if any(v is None or not np.isfinite(v) for v in vals):
        return np.nan
    return max(float(np.dot(c[:-1], vals) + c[-1]), 0.0)


def load_nets(models_dir: str):
    import torch
    from scripts.learned_width_estimator import WidthNet
    nets = {}
    for f in glob.glob(os.path.join(models_dir, "width_net_heldout_*.pth")):
        s = os.path.basename(f)[len("width_net_heldout_"):-4]
        net = WidthNet()
        net.load_state_dict(torch.load(f, map_location="cpu"))
        net.eval()
        nets[s] = net
    return nets


def cnn_width(net, row, k, pitch):
    import torch
    from scripts.learned_width_estimator import Z_OFFSET, make_inputs
    prof, sc = make_inputs(np.clip(np.round(row), 0, 255)[None, :].astype(np.uint8), k, pitch)
    with torch.no_grad():
        q = net(torch.from_numpy(prof), torch.from_numpy(sc)).numpy()[0]
    return float(np.exp(q[2]) - Z_OFFSET)


def gridline_widths(img, skel_pts, normals, centre, res, net):
    """Median width estimates (um, raw) at skeleton points within 0.3 mm of `centre`."""
    pitch = 1000.0 / res
    rad = max(0.3 * res, 3.0)
    d = np.hypot(skel_pts[:, 0] - centre[0], skel_pts[:, 1] - centre[1])
    sel = np.where(d <= rad)[0]
    if len(sel) == 0:
        return None
    R = int(np.ceil(max(WINDOW_HALF_UM / pitch, 6.0)))
    k = np.arange(-R, R + 1).astype(np.float64)
    vals = {m: [] for m in ("fwhm", "rect", "photo", "cnn")}
    g = img.astype(np.float32)
    for i in sel[np.argsort(d[sel])][:12]:
        x, y = skel_pts[i]
        nx, ny = normals[i]
        if not np.isfinite(nx):
            continue
        row = map_coordinates(g, [y + ny * k, x + nx * k], order=1, mode="nearest").astype(np.float64)
        f = measure_ours(row, k, pitch)
        if f is not None:
            vals["fwhm"].append(f)
        r = rectangle_transform(row, pitch)
        if r is not None:
            vals["rect"].append(r)
        p = deficit_features(row, k, pitch)
        if p is not None:
            vals["photo"].append(p["P"])
        if net is not None:
            vals["cnn"].append(cnn_width(net, row, k.astype(int), pitch))
    return {m: (float(np.median(v)) if v else np.nan) for m, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="krk_bands")
    ap.add_argument("--table", default="data/krkCMd_raw/krkCMd_table.csv")
    ap.add_argument("--meas", default="outputs/limits/measurements.csv.gz")
    ap.add_argument("--tuning", default="docs/results/2d/tuning.json")
    ap.add_argument("--models", default="models/width_net")
    ap.add_argument("--out", default="outputs/limits_2d")
    ap.add_argument("--publish", default="docs/results/limits")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref = load_ref(args.table)
    cfg = json.load(open(args.tuning))["best"]
    cal = calibrations(args.meas)
    nets = load_nets(args.models) if os.path.isdir(args.models) else {}
    rows = []
    for npy in sorted(glob.glob(os.path.join(args.bands, "*.npy"))):
        meta = json.load(open(npy[:-4] + ".json"))
        if meta["series"] == "CMd_0.23_2mths":       # development series of the ridge tuning
            continue
        gray = np.load(npy)
        net = nets.get(meta["series"])
        for res in RESOLUTIONS:
            rng = np.random.default_rng(int(res * 10) + 7 * meta["stage"])
            img = degrade(gray, res, PSF, NOISE, rng)
            f = NATIVE_PX_PER_MM / res
            mask = segment_ridge(img, res, cfg)
            ys, xs = np.nonzero(skeletonize(mask))
            pts = np.stack([xs, ys], 1).astype(np.float64)
            normals = local_normals(pts, max(3.5, 0.5 * res)) if len(pts) else pts
            c = cal[(meta["series"], res)]
            for name, x1, y1, x2, y2 in meta["lines"]:
                key = (meta["series"], meta["image"], meta["stage"], name)
                if key not in ref:
                    continue
                centre = ((x1 + x2) / 2 / f, ((y1 + y2) / 2 - meta["row0"]) / f)
                w = gridline_widths(img, pts, normals, centre, res, net) if len(pts) else None
                r = dict(series=meta["series"], image=meta["image"], stage=meta["stage"], gridline=name,
                         px_per_mm=res, ref_um=ref[key], hit=w is not None)
                if w is not None:
                    r.update(fwhm_cal=apply_cal(c["fwhm"], w["fwhm"]), rect_cal=apply_cal(c["rect"], w["rect"]),
                             photo_cal=apply_cal(c["photo"], w["photo"]),
                             fusion_cal=(apply_cal(c["fusion"], w["fwhm"], w["photo"])
                                         if np.isfinite(w["fwhm"]) else apply_cal(c["photo"], w["photo"])),
                             cnn=w["cnn"])
                rows.append(r)
            print(f"{os.path.basename(npy)} @ {res:g} px/mm: {len(meta['lines'])} gridlines", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "gridline_measurements_2d.csv"), index=False)
    out = []
    for res, g in df.groupby("px_per_mm"):
        for m in ("fwhm_cal", "rect_cal", "photo_cal", "fusion_cal", "cnn"):
            if m not in g:
                continue
            ok = g[m].notna()
            e = (g.loc[ok, m] - g.loc[ok, "ref_um"])
            # cluster bootstrap over bands (image x stage)
            cl = (g.loc[ok, "series"] + g.loc[ok, "image"].astype(str) + "_" + g.loc[ok, "stage"].astype(str)).to_numpy()
            rng = np.random.default_rng(0)
            u = np.unique(cl)
            boots = []
            for _ in range(1000):
                pick = rng.choice(u, len(u))
                ee = np.concatenate([np.abs(e.to_numpy()[cl == p]) for p in pick])
                boots.append(ee.mean())
            out.append(dict(px_per_mm=res, method=m, n=len(g), hit_rate=round(float(g.hit.mean()), 3),
                            measured_rate=round(float(ok.mean()), 3), mae_um=round(float(e.abs().mean()), 1),
                            ci_lo=round(float(np.percentile(boots, 2.5)), 1),
                            ci_hi=round(float(np.percentile(boots, 97.5)), 1),
                            bias_um=round(float(e.mean()), 1)))
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(args.out, "summary_2d.csv"), index=False)
    print(summ.pivot(index="px_per_mm", columns="method", values="mae_um").sort_index(ascending=False).to_string())
    if args.publish:
        os.makedirs(args.publish, exist_ok=True)
        shutil.copy(os.path.join(args.out, "summary_2d.csv"), os.path.join(args.publish, "summary_2d.csv"))


if __name__ == "__main__":
    main()
