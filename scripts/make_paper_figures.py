"""Build the paper figures from the saved result tables (no recomputation)."""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "docs", "results")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

s1 = pd.read_csv(os.path.join(R, "summary_by_resolution.csv"))
s2 = pd.read_csv(os.path.join(R, "2d", "summary_2d.csv"))
mdl = json.load(open(os.path.join(R, "resolution_error_model.json")))
noskill = mdl["phone_like"]["noskill_mae_um"]

# Fig 1: MAE vs resolution, 1-D study (3 methods) + fitted model + no-skill line
fig, ax = plt.subplots(figsize=(3.5, 2.7))
d = s1[s1.scenario == "phone_like"]
for m, st, lab in (("subpixel_fwhm (ours)", "o-", "sub-pixel FWHM"), ("threshold_count (baseline)", "s--", "threshold count"),
                   ("gradient_edges (baseline)", "^:", "gradient edges")):
    g = d[d.method == m].sort_values("px_per_mm")
    ax.plot(g.px_per_mm, g.mae_um, st, ms=3.5, lw=1, label=lab)
p = np.logspace(np.log10(1000 / 252), np.log10(400), 100)
pm = mdl["phone_like"]
ax.plot(1000 / p, pm["a_um"] + pm["c"] * p ** pm["k"], "k-", lw=0.8, alpha=0.6, label="fitted model")
ax.axhline(noskill, color="r", lw=0.8, ls="-.", label="no-skill (mean width)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("image resolution (px/mm)"); ax.set_ylabel("mean abs. error ($\mu$m)")
ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mae_vs_resolution.pdf")); plt.close(fig)

# Fig 2: 1-D vs 2-D end-to-end (held-out), phone-like
fig, ax = plt.subplots(figsize=(3.5, 2.7))
g = d[d.method == "subpixel_fwhm (ours)"].sort_values("px_per_mm")
ax.plot(g.px_per_mm, g.mae_um, "o-", ms=3.5, lw=1, label="1-D profiles (all 14,198)")
h = s2[(s2.scenario == "phone_like") & (s2.split == "heldout")]
for m, st in (("ridge", "s--"), ("otsu", "^:")):
    q = h[h.method == m].sort_values("px_per_mm")
    ax.plot(q.px_per_mm, q.mae_um, st, ms=3.5, lw=1, label=f"2-D end-to-end, {m} (held-out)")
ax.axhline(noskill, color="r", lw=0.8, ls="-.", label="no-skill")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("image resolution (px/mm)"); ax.set_ylabel("mean abs. error ($\mu$m)")
ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_1d_vs_2d.pdf")); plt.close(fig)
print("figures written to", OUT)
