"""
learned_width_estimator.py
==========================
Resolution-conditioned learned crack-width estimator with quantile outputs, trained on real
krkCMd profiles degraded on the fly to random resolutions (2.5-252 px/mm), blur, noise and
shading. Evaluated leave-one-series-out on exactly the same simulated test images as the
classical estimators (outputs/limits/sim/*.npz written by evaluate_limits_krkCMd.py).

Input  : the pixel profile resampled to a fixed physical grid (+-0.8 mm, 256 points), divided
         by the local background; scalars log(pixel pitch), background/255 and the photometric
         width P (deficit area / background) of the same pixels.
Output : quantiles 5/25/50/75/95 % of z = log(w + 20 um) (pinball loss); the median is the
         point estimate, the 5-95 % band feeds the conformal step (CQR).

The point is not a better network: it is to see how close a flexible estimator gets to the
information limit at each resolution, and whether it beats the physics-based photometric width
where theory says only the deficit area is identifiable.

Usage:
    python scripts/learned_width_estimator.py --csv data/krkCMd_raw/krkCMd_table.csv --sim outputs/limits/sim
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate_limits_krkCMd import coarse_rows, deficit_batch  # noqa: E402
from scripts.evaluate_resolution_krkCMd import NATIVE_PITCH_UM  # noqa: E402

GRID = np.linspace(-800.0, 800.0, 256)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_OFFSET = 20.0


def interp_matrix(k: np.ndarray, pitch_um: float, grid: np.ndarray = GRID) -> np.ndarray:
    """Linear interpolation from pixel centres k*pitch onto `grid` as a (n_pix, n_grid) matrix."""
    x = k * pitch_um
    M = np.zeros((len(x), len(grid)))
    j = np.clip(np.searchsorted(x, grid) - 1, 0, len(x) - 2)
    f = np.clip((grid - x[j]) / (x[j + 1] - x[j]), 0.0, 1.0)
    M[j, np.arange(len(grid))] = 1 - f
    M[j + 1, np.arange(len(grid))] += f
    return M


def make_inputs(rows: np.ndarray, k: np.ndarray, pitch_um: float):
    """Network inputs for pixel rows (N, n_pix) sharing pitch and pixel grid."""
    g = rows.astype(np.float64)
    P, B, _ = deficit_batch(rows, k, pitch_um)
    B = np.maximum(B, 5.0)
    prof = (g @ interp_matrix(k, pitch_um)) / B[:, None] - 1.0
    prof = np.clip(prof, -1.5, 1.5)
    P = np.nan_to_num(P, nan=0.0)
    sc = np.stack([np.full(len(g), np.log(pitch_um / 100.0)), B / 255.0, P / 500.0], axis=1)
    return prof.astype(np.float32), sc.astype(np.float32)


class WidthNet(nn.Module):
    def __init__(self, n_scalar: int = 3, n_q: int = len(QUANTILES), ch: int = 48):
        super().__init__()

        # GroupNorm, not BatchNorm: batch statistics would leak the batch's imaging condition
        def block(ci, co, d):
            return nn.Sequential(nn.Conv1d(ci, co, 5, padding=2 * d, dilation=d), nn.GroupNorm(8, co), nn.GELU())

        self.film = nn.Linear(n_scalar, 2 * ch)
        self.stem = block(1, ch, 1)
        self.body = nn.Sequential(block(ch, ch, 1), nn.MaxPool1d(2), block(ch, ch, 2), nn.MaxPool1d(2),
                                  block(ch, ch, 4), nn.MaxPool1d(2), block(ch, ch, 8))
        self.head = nn.Sequential(nn.Linear(ch * 2 + n_scalar, 128), nn.GELU(), nn.Linear(128, n_q))

    def forward(self, prof, sc):
        h = self.stem(prof[:, None, :])
        gamma, beta = self.film(sc).chunk(2, dim=1)
        h = h * (1 + gamma[:, :, None]) + beta[:, :, None]       # resolution-conditioned features
        h = self.body(h)
        h = torch.cat([h.mean(-1), h.amax(-1), sc], dim=1)
        q = self.head(h)
        # monotone quantiles: median + cumulative positive offsets
        mid = q[:, 2:3]
        lo = mid - torch.cumsum(nn.functional.softplus(q[:, [1, 0]]), dim=1)
        hi = mid + torch.cumsum(nn.functional.softplus(q[:, [3, 4]]), dim=1)
        return torch.cat([lo[:, [1]], lo[:, [0]], mid, hi], dim=1)


def pinball(pred, z):
    q = torch.tensor(QUANTILES, device=pred.device)[None, :]
    e = z[:, None] - pred
    return torch.mean(torch.maximum(q * e, (q - 1) * e))


def random_batch(X, W, idx, rng, batch, n_groups=8):
    """One training batch of random profiles, in `n_groups` groups that each get their own random
    resolution (log-uniform 2.5-252 px/mm), PSF, noise and shading."""
    profs, scs, zs = [], [], []
    for _ in range(n_groups):
        sel = rng.choice(idx, size=batch // n_groups)
        pitch = float(np.exp(rng.uniform(np.log(NATIVE_PITCH_UM), np.log(400.0))))
        psf = float(rng.uniform(0.0, 1.6))
        noise = float(rng.uniform(1.0, 11.0))
        ramp = float(rng.uniform(0.0, 30.0)) if rng.uniform() < 0.5 else 0.0
        rows, k = coarse_rows(X[sel], pitch, psf, noise, rng, ramp)
        prof, sc = make_inputs(rows, k, pitch)
        profs.append(prof)
        scs.append(sc)
        zs.append(np.log(W[sel] + Z_OFFSET).astype(np.float32))
    return np.concatenate(profs), np.concatenate(scs), np.concatenate(zs)


def train_fold(X, W, train_idx, val_idx, device, steps, batch, seed, log_every=500):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = WidthNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    vrng = np.random.default_rng(seed + 999)
    val = [random_batch(X, W, val_idx, vrng, 512) for _ in range(12)]     # fixed validation degradations
    best, best_state = np.inf, None
    t0 = time.time()
    for step in range(1, steps + 1):
        net.train()
        prof, sc, z = random_batch(X, W, train_idx, rng, batch)
        loss = pinball(net(torch.from_numpy(prof).to(device), torch.from_numpy(sc).to(device)),
                       torch.from_numpy(z).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                vl = np.mean([pinball(net(torch.from_numpy(p).to(device), torch.from_numpy(s).to(device)),
                                      torch.from_numpy(zz).to(device)).item() for p, s, zz in val])
            if vl < best:
                best, best_state = vl, {k_: v.detach().clone() for k_, v in net.state_dict().items()}
            print(f"    step {step}/{steps} train {loss.item():.4f} val {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def predict(net, prof, sc, device, bs=4096):
    out = []
    for i in range(0, len(prof), bs):
        out.append(net(torch.from_numpy(prof[i:i + bs]).to(device), torch.from_numpy(sc[i:i + bs]).to(device)).cpu().numpy())
    return np.exp(np.concatenate(out)) - Z_OFFSET


def nested(args, X, W, series, images, sims, device):
    """For every held-out series s and every other series t: train on the remaining two series and
    predict t. These out-of-series residuals calibrate the CQR intervals of fold s, so the calibration
    sees the same kind of specimen shift as the test (the role-'cal' rows of the main run do not)."""
    preds = []
    names = np.unique(series)
    for si, held in enumerate(names):
        for ti, cal_s in enumerate(names):
            if cal_s == held:
                continue
            tr_all = np.where(~np.isin(series, [held, cal_s]) & (W > 0))[0]
            imgs = np.unique(images[tr_all])
            vimg = np.random.default_rng(args.seed + 10 * si + ti).choice(imgs, size=max(1, len(imgs) // 10),
                                                                         replace=False)
            val_idx = tr_all[np.isin(images[tr_all], vimg)]
            train_idx = tr_all[~np.isin(images[tr_all], vimg)]
            print(f"nested: held-out {held}, calibration series {cal_s}, train {len(train_idx)}", flush=True)
            net = train_fold(X, W, train_idx, val_idx, device, args.steps, args.batch, args.seed + 100 + 10 * si + ti,
                             log_every=args.steps)
            idx = np.where(series == cal_s)[0]
            for f in sims:
                z = np.load(f)
                scen, res = os.path.basename(f)[:-4].rsplit("_", 1)
                prof, sc = make_inputs(z["rows"][idx], z["k"], float(z["pitch_um"]))
                d = pd.DataFrame(predict(net, prof, sc, device), columns=[f"cnn_q{int(100*a):02d}" for a in QUANTILES])
                d.insert(0, "role", "nested_cal")
                d.insert(0, "fold", held)
                d.insert(0, "i", idx)
                d.insert(0, "px_per_mm", float(res))
                d.insert(0, "scenario", scen)
                preds.append(d)
    out = pd.concat(preds, ignore_index=True)
    out.to_csv(os.path.join(args.out, "cnn_nested_cal.csv.gz"), index=False)
    print("saved", os.path.join(args.out, "cnn_nested_cal.csv.gz"), out.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/krkCMd_raw/krkCMd_table.csv")
    ap.add_argument("--sim", default="outputs/limits/sim")
    ap.add_argument("--out", default="outputs/limits")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="models/width_net")
    ap.add_argument("--nested", action="store_true",
                    help="train the 12 nested models (held-out s, calibration series t) whose predictions on t are "
                         "shift-aware conformal calibration scores for fold s -> cnn_nested_cal.csv.gz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tab = pd.read_csv(args.csv)
    xcols = [c for c in tab.columns if c.startswith("x") and c[1:].isdigit()]
    X = tab[xcols].to_numpy(dtype=np.float64)
    W = tab.MANwidth.to_numpy(dtype=np.float64)
    series = tab.Series.to_numpy()
    images = (tab.Series + "/" + tab.Image.astype(str)).to_numpy()
    sims = sorted(glob.glob(os.path.join(args.sim, "*.npz")))
    os.makedirs(args.models, exist_ok=True)
    if args.nested:
        nested(args, X, W, series, images, sims, device)
        return
    preds = []
    for fi, held in enumerate(np.unique(series)):
        tr_all = np.where((series != held) & (W > 0))[0]
        # 10 % of training IMAGES for checkpoint selection (never the held-out series)
        imgs = np.unique(images[tr_all])
        vimg = np.random.default_rng(args.seed + fi).choice(imgs, size=max(1, len(imgs) // 10), replace=False)
        val_idx = tr_all[np.isin(images[tr_all], vimg)]
        train_idx = tr_all[~np.isin(images[tr_all], vimg)]
        print(f"fold {fi}: held-out {held}  train {len(train_idx)}  val {len(val_idx)}", flush=True)
        net = train_fold(X, W, train_idx, val_idx, device, args.steps, args.batch, args.seed + fi)
        torch.save(net.state_dict(), os.path.join(args.models, f"width_net_heldout_{held}.pth"))
        te = np.where(series == held)[0]
        for f in sims:
            z = np.load(f)
            scen, res = os.path.basename(f)[:-4].rsplit("_", 1)
            # role "test": the held-out series; role "cal": this fold's validation images (never
            # trained on, never the held-out series) -> conformal calibration scores (CQR)
            for role, rows_idx in (("test", te), ("cal", val_idx)):
                prof, sc = make_inputs(z["rows"][rows_idx], z["k"], float(z["pitch_um"]))
                q = predict(net, prof, sc, device)
                d = pd.DataFrame(q, columns=[f"cnn_q{int(100*a):02d}" for a in QUANTILES])
                d.insert(0, "role", role)
                d.insert(0, "fold", held)
                d.insert(0, "i", rows_idx)
                d.insert(0, "px_per_mm", float(res))
                d.insert(0, "scenario", scen)
                preds.append(d)
    out = pd.concat(preds, ignore_index=True)
    out.to_csv(os.path.join(args.out, "cnn_predictions.csv.gz"), index=False)
    print("saved", os.path.join(args.out, "cnn_predictions.csv.gz"), out.shape)


if __name__ == "__main__":
    main()
