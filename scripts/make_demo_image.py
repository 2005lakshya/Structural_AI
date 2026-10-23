"""
make_demo_image.py
==================
Writes a test photo for trying the app's crack-width measurement ("Certified measurement" panel on
the Detection page) on a fresh machine, and prints the scale to enter in the UI.

  --real (recommended): one real krkCMd crack scan (CC BY 4.0, Zenodo 10.5281/zenodo.14568863),
         re-imaged to phone-like quality at --px-per-mm; prints its reference widths. The band is
         fetched once with scripts/fetch_krkCMd_band.py (~150 MB of range requests, needs internet)
         into krk_bands/ (git-ignored).
  default: a synthetic dark line of --width-mm. Smoke test only: an ideal slot is not a real crack,
         and the calibration (made on real cracks) does not have to reproduce its width.

Usage:
    python scripts/make_demo_image.py --real --px-per-mm 12.5      # -> outputs/demo/demo_crack_real.png
    python scripts/make_demo_image.py --width-mm 0.3 --px-per-mm 8  # -> outputs/demo/demo_crack_synthetic.png
Then in the app: Detection -> upload the PNG -> Reference Object: "Reference size in image (px)" =
the printed px/mm value, "Known physical size (mm)" = 1 -> Calculate Crack Width from Image.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BAND = ("CMd_0.28_20mths", 1, 1)


def real(px_per_mm: float, out_dir: str) -> str:
    from scripts.evaluate_end_to_end_krkCMd import degrade
    series, image, stage = BAND
    bands = os.path.join(ROOT, "krk_bands")
    stem = os.path.join(bands, f"{series}_img{image}_stage{stage}")
    if not os.path.exists(stem + ".npy"):
        print("fetching one krkCMd band from Zenodo (one-off) ...", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fetch_krkCMd_band.py"),
                        series, str(image), str(stage), bands], check=True)
    img = degrade(np.load(stem + ".npy"), px_per_mm, 0.7, 4.0, np.random.default_rng(1))
    path = os.path.join(out_dir, "demo_crack_real.png")
    cv2.imwrite(path, img)
    table = os.path.join(ROOT, "data", "krkCMd_raw", "krkCMd_table.csv")
    if os.path.exists(table):
        from scripts.evaluate_end_to_end_krkCMd import load_ref
        ref = load_ref(table)
        meta = json.load(open(stem + ".json"))
        w = np.array([v for v in (ref.get((series, image, stage, ln[0])) for ln in meta["lines"]) if v is not None])
        print(f"reference widths of this crack: 95th percentile {np.percentile(w, 95) / 1000:.3f} mm, "
              f"maximum {w.max() / 1000:.3f} mm")
    else:
        print("reference widths (krkCMd table): 95th percentile 0.287 mm, maximum 0.393 mm")
    return path


def synthetic(width_mm: float, px_per_mm: float, out_dir: str) -> str:
    from scripts.width_information_limit import slot_template
    n = int(40 * px_per_mm)
    x = (np.arange(n) - n / 2) / px_per_mm * 1000
    pitch = 1000 / px_per_mm
    Q, _, _ = slot_template(width_mm * 1000, 0.0, x - pitch / 2, x + pitch / 2, 0.7 * pitch)
    img = np.clip(np.tile(170 - 160 * Q, (n, 1)) + np.random.default_rng(0).normal(0, 4, (n, n)), 0, 255)
    path = os.path.join(out_dir, "demo_crack_synthetic.png")
    cv2.imwrite(path, img.astype(np.uint8))
    print(f"synthetic line {width_mm} mm wide (smoke test only, not a real crack)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--px-per-mm", type=float, default=12.5)
    ap.add_argument("--width-mm", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "demo"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = real(a.px_per_mm, a.out) if a.real else synthetic(a.width_mm, a.px_per_mm, a.out)
    print(f"wrote {path}")
    print(f"in the app: Reference Object -> 'Reference size in image (px)' = {a.px_per_mm:g}, "
          f"'Known physical size (mm)' = 1")


if __name__ == "__main__":
    main()
