"""
width_test_chart.py
===================
(1) Generate a printable A4 chart of dark lines with known widths and four corner
    fiducials, and (2) analyse photographs of the printed chart: recover scale from
    the fiducials, measure every line with the project's sub-pixel width routine and
    report measured-vs-nominal width as a function of the photo's px/mm.

This gives a REAL-camera, independent-reference version of the resolution study
(the krkCMd study uses re-imaged scans with a manual reference).

Printing notes (important for validity)
---------------------------------------
* Print at 100 % scale ("actual size", no fit-to-page) on a laser printer at the
  highest quality; verify the 100 mm scale bar with a ruler.
* Printer dot gain changes thin-line widths. Verify the real printed widths with a
  calibrated microscope / comparator card and use THOSE as the reference (put them in
  a JSON file and pass --printed-widths). Without that the nominal widths are only
  accurate to roughly +-0.03 mm and lines <= 0.1 mm are unreliable.

Usage
-----
  python scripts/width_test_chart.py make --out docs/width_test_chart
  python scripts/width_test_chart.py analyze --photo IMG.jpg --layout docs/width_test_chart.json
  python scripts/width_test_chart.py selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.crack_width_measure import PREBLUR_FWHM_PX, PREBLUR_SIGMA_PX, fwhm_from_profile  # noqa: E402

PAGE_W, PAGE_H = 210.0, 297.0
FID_CENTRES = {"TL": (20.0, 20.0), "TR": (190.0, 20.0), "BL": (20.0, 277.0), "BR": (190.0, 277.0)}
FID_SIZE = {"TL": 14.0, "TR": 10.0, "BL": 10.0, "BR": 10.0}   # TL larger: orientation cue
WIDTHS_MM = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50]
LINE_LEN = 110.0
LINES_PER_GROUP = 3
GROUP_PITCH = 20.0
LINE_SPACING = 4.0
X0 = 50.0
Y0 = 50.0


def layout() -> dict:
    lines = []
    for gi, w in enumerate(WIDTHS_MM):
        for k in range(LINES_PER_GROUP):
            y = Y0 + gi * GROUP_PITCH + k * LINE_SPACING
            lines.append(dict(id=f"w{w:.2f}_{k}", nominal_mm=w, x1=X0, x2=X0 + LINE_LEN, y=y))
    return dict(page_mm=[PAGE_W, PAGE_H], fid_centres=FID_CENTRES, fid_size=FID_SIZE, lines=lines)


def make(out_base: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    lay = layout()
    c = canvas.Canvas(out_base + ".pdf", pagesize=A4)
    c.setFillGray(0)
    for k, (cx, cy) in FID_CENTRES.items():
        s = FID_SIZE[k]
        c.rect((cx - s / 2) * mm, (PAGE_H - cy - s / 2) * mm, s * mm, s * mm, stroke=0, fill=1)
    for ln in lay["lines"]:
        h = ln["nominal_mm"]
        c.rect(ln["x1"] * mm, (PAGE_H - ln["y"] - h / 2) * mm, (ln["x2"] - ln["x1"]) * mm, h * mm, stroke=0, fill=1)
    # 100 mm scale bar (verify with a ruler after printing)
    c.rect(50 * mm, (PAGE_H - 262) * mm, 100 * mm, 0.5 * mm, stroke=0, fill=1)
    c.setFont("Helvetica", 7)
    c.drawString(50 * mm, (PAGE_H - 258) * mm, "100 mm scale bar - print at 100 % (actual size) and verify with a ruler")
    for gi, w in enumerate(WIDTHS_MM):
        c.drawString(165 * mm, (PAGE_H - (Y0 + gi * GROUP_PITCH + 4.0) - 1) * mm, f"{w:.2f} mm")
    c.showPage()
    c.save()
    json.dump(lay, open(out_base + ".json", "w"), indent=1)
    print("wrote", out_base + ".pdf", "and", out_base + ".json")


def render_chart(px_per_mm: float, gray_bg: int = 235) -> np.ndarray:
    """Rasterise the chart (for self-tests and previews). Black on near-white."""
    lay = layout()
    W, H = int(round(PAGE_W * px_per_mm)), int(round(PAGE_H * px_per_mm))
    img = np.full((H, W), gray_bg, np.float32)
    S = 8  # supersample to get exact sub-pixel line widths
    big = np.full((H * 1, W * 1), 0.0, np.float32)
    cov = np.zeros((H * S, W * S), np.uint8)
    for k, (cx, cy) in FID_CENTRES.items():
        s = FID_SIZE[k]
        x0, x1 = int((cx - s / 2) * px_per_mm * S), int((cx + s / 2) * px_per_mm * S)
        y0, y1 = int((cy - s / 2) * px_per_mm * S), int((cy + s / 2) * px_per_mm * S)
        cov[y0:y1, x0:x1] = 1
    for ln in lay["lines"]:
        h = ln["nominal_mm"]
        y0, y1 = (ln["y"] - h / 2) * px_per_mm * S, (ln["y"] + h / 2) * px_per_mm * S
        x0, x1 = int(ln["x1"] * px_per_mm * S), int(ln["x2"] * px_per_mm * S)
        a, b = int(np.floor(y0)), int(np.ceil(y1))
        for yy in range(a, b):
            frac = min(yy + 1, y1) - max(yy, y0)
            if frac > 0:
                cov[yy, x0:x1] = 1 if frac >= 0.5 else cov[yy, x0:x1]
    frac = cov.reshape(H, S, W, S).mean(axis=(1, 3))
    del big
    return (gray_bg * (1 - frac) + 20.0 * frac).astype(np.float32)


def _find_fiducials(gray: np.ndarray):
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 200:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if 0.7 < w / h < 1.4 and a / (w * h) > 0.7:
            cand.append((a, (x + w / 2, y + h / 2)))
    cand.sort(reverse=True)
    if len(cand) < 4:
        raise RuntimeError("could not find the 4 corner fiducials")
    pts = np.array([c[1] for c in cand[:4]], np.float32)
    areas = [c[0] for c in cand[:4]]
    return pts, areas


def analyze(photo_path: str, layout_path: str, printed_widths: dict | None = None):
    lay = json.load(open(layout_path))
    img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(photo_path)
    pts, areas = _find_fiducials(img)
    big = int(np.argmax(areas))                     # TL fiducial is the largest square
    tl = pts[big]
    others = [p for i, p in enumerate(pts) if i != big]
    # order the remaining three by angle around the centroid relative to TL
    cen = pts.mean(axis=0)
    ang = lambda p: np.arctan2(p[1] - cen[1], p[0] - cen[0])  # noqa: E731
    ordered = sorted(pts.tolist(), key=ang)
    k = [i for i, p in enumerate(ordered) if np.allclose(p, tl)][0]
    ordered = ordered[k:] + ordered[:k]             # clockwise (image coords) starting at TL: TL, TR, BR, BL
    src = np.array(ordered, np.float32)
    fc = lay["fid_centres"]
    dst = np.array([fc["TL"], fc["TR"], fc["BR"], fc["BL"]], np.float32)
    Hm, _ = cv2.findHomography(dst, src)            # chart mm -> photo px
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), PREBLUR_SIGMA_PX)
    from scipy.ndimage import map_coordinates

    def to_px(x, y):
        v = Hm @ np.array([x, y, 1.0])
        return v[:2] / v[2]

    rows = []
    for ln in lay["lines"]:
        xs = np.linspace(ln["x1"] + 10, ln["x2"] - 10, 40)
        widths = []
        for x in xs:
            p = to_px(x, ln["y"])
            # local scale and the line's normal direction in the photo
            px_x = to_px(x + 1.0, ln["y"]) - p      # px per mm along the line
            px_y = to_px(x, ln["y"] + 1.0) - p      # px per mm across the line
            scale = float(np.linalg.norm(px_y))
            n = px_y / (scale + 1e-12)
            R = max(1.5 * scale * max(ln["nominal_mm"], 0.2) + 4.0, 6.0)
            t = np.arange(-R, R + 1e-6, 0.5, dtype=np.float32)
            row = map_coordinates(g, [p[1] + n[1] * t, p[0] + n[0] * t], order=1, mode="nearest")
            w = fwhm_from_profile(row, t)
            if w is not None:
                widths.append(np.sqrt(max(w ** 2 - PREBLUR_FWHM_PX ** 2, 0.0)) / scale)   # -> mm
        ref = (printed_widths or {}).get(ln["id"], ln["nominal_mm"])
        rows.append(dict(id=ln["id"], nominal_mm=ln["nominal_mm"], reference_mm=ref,
                         n_profiles=len(widths),
                         measured_mm=float(np.median(widths)) if widths else None,
                         px_per_mm=float(np.linalg.norm(to_px(ln["x1"] + 50, ln["y"] + 1.0) - to_px(ln["x1"] + 50, ln["y"])))))
    return rows


def selftest():
    """Render the chart, simulate a photo (perspective, blur, noise, lower resolution) and analyse it."""
    ppm_native = 40.0
    chart = render_chart(ppm_native)
    rng = np.random.default_rng(0)
    out = {}
    for target_ppm in (20.0, 10.0, 5.0):
        f = ppm_native / target_ppm
        g = cv2.GaussianBlur(chart, (0, 0), 0.7 * f)
        g = cv2.resize(g, (int(chart.shape[1] / f), int(chart.shape[0] / f)), interpolation=cv2.INTER_AREA)
        h, w = g.shape
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[0.03 * w, 0.02 * h], [0.97 * w, 0.0], [0.99 * w, 0.99 * h], [0.0, 0.97 * h]])
        M = cv2.getPerspectiveTransform(src, dst)
        g = cv2.warpPerspective(g, M, (w, h), borderValue=200)
        g = np.clip(g + rng.normal(0, 3, g.shape), 0, 255).astype(np.uint8)
        path = os.path.join(os.environ.get("TEMP", "."), f"chart_sim_{int(target_ppm)}.png")
        cv2.imwrite(path, g)
        lay = os.path.join(os.environ.get("TEMP", "."), "chart_layout.json")
        json.dump(layout(), open(lay, "w"))
        rows = analyze(path, lay)
        by = {}
        for r in rows:
            if r["measured_mm"] is not None:
                by.setdefault(r["nominal_mm"], []).append(r["measured_mm"])
        out[target_ppm] = {w: round(float(np.median(v)), 3) for w, v in by.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--out", default="docs/width_test_chart")
    a = sub.add_parser("analyze"); a.add_argument("--photo", required=True)
    a.add_argument("--layout", default="docs/width_test_chart.json")
    a.add_argument("--printed-widths", default=None, help="JSON {line_id: measured_width_mm}")
    a.add_argument("--csv", default=None)
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "make":
        make(args.out)
    elif args.cmd == "analyze":
        pw = json.load(open(args.printed_widths)) if args.printed_widths else None
        rows = analyze(args.photo, args.layout, pw)
        import pandas as pd
        df = pd.DataFrame(rows)
        print(df.groupby("nominal_mm").agg(measured_mm=("measured_mm", "median"), px_per_mm=("px_per_mm", "median")).round(3))
        if args.csv:
            df.to_csv(args.csv, index=False)
    else:
        res = selftest()
        for ppm, d in res.items():
            print(f"{ppm:5.1f} px/mm:", d)


if __name__ == "__main__":
    main()
