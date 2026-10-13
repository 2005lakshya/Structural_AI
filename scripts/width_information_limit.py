"""
width_information_limit.py
==========================
How much can a digital image say about the width of a crack?  Fisher information and
Cramer-Rao lower bounds (CRLB) for crack-width estimation under optical blur, pixel
integration and sensor noise.

Forward model (one profile taken perpendicular to the crack, all lengths in um)
--------------------------------------------------------------------------------
Surface brightness: a dark slot of width w and contrast h on a background B,

    r(x) = B - h * 1[|x - x0| < w/2]

imaged through a Gaussian PSF (std s_o) and integrated over square pixels of pitch p:

    mu_k = B - h * Q_k(w, x0),
    Q_k  = (1/p) * int_{pixel k} [Phi((x-x0+w/2)/s_o) - Phi((x-x0-w/2)/s_o)] dx

which has a closed form through Psi(u) = u*Phi(u) + phi(u) (Psi' = Phi).  Pixel values are
mu_k + Gaussian noise (std `noise`, grey levels).  Unknowns: theta = (w, h, x0, B).

Main result (checked numerically by `asymptotics` and `monte_carlo` below)
--------------------------------------------------------------------------
For w << s_eff (s_eff^2 = s_o^2 + p^2/12) the image depends on w and h, to leading order,
only through the deficit area A = w*h:

    mu_k = B - A*q_k - (A*w^2/24)*q''_k + O(w^4),

so the two Jacobian columns for w and h become collinear. The information left about w
once h is unknown comes from the w^2 term only, giving

    sd(w_hat)/w  >=  C * (noise/h) * sqrt(p/s_eff) * (s_eff/w)^3     (contrast unknown)
    sd(w_hat)/w  ~   C'* (noise/h) * sqrt(p*s_eff) / w               (contrast known)

The relative error with unknown contrast grows as the CUBE of (blur/width), and the
smallest width measurable to a given relative precision scales as s_eff * SNR^(-1/3).
Better sensors therefore buy little resolution. Below the resolution limit a width is
only obtainable with contrast knowledge, and then the floor is the relative spread of
the contrast (Bayesian CRLB with a prior on h, `crlb(..., h_prior_cv=...)`).

CLI
---
    python scripts/width_information_limit.py figures --out docs/results/limits
    python scripts/width_information_limit.py montecarlo --out docs/results/limits
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.special import ndtr  # standard normal CDF

SQRT2PI = np.sqrt(2.0 * np.pi)


def _phi(u):
    return np.exp(-0.5 * u * u) / SQRT2PI


def _Psi(u):
    return u * ndtr(u) + _phi(u)


# ─────────────────────────────────────────────────────────────────────────────
#  Forward model and analytic Jacobian
# ─────────────────────────────────────────────────────────────────────────────

def pixel_edges(pitch: float, half_window: float, phase: float = 0.0):
    """Left/right edges (um) of the pixels covering [-half_window, half_window].
    `phase` in [0, 1) shifts the grid by a fraction of a pixel."""
    R = int(np.ceil(half_window / pitch))
    k = np.arange(-R, R + 1)
    a = (k - 0.5 + phase) * pitch
    return a, a + pitch


def slot_template(w, x0, a, b, s_o):
    """Q_k and its derivatives dQ/dw, dQ/dx0 for a unit-contrast slot. Broadcasts over w, x0."""
    s = max(float(s_o), 1e-6)
    p = b - a
    aL, bL = (a - x0 + w / 2) / s, (b - x0 + w / 2) / s
    aR, bR = (a - x0 - w / 2) / s, (b - x0 - w / 2) / s
    Q = (s / p) * (_Psi(bL) - _Psi(aL) - _Psi(bR) + _Psi(aR))
    dQ_dw = (ndtr(bL) - ndtr(aL) + ndtr(bR) - ndtr(aR)) / (2 * p)
    dQ_dx0 = (-ndtr(bL) + ndtr(aL) + ndtr(bR) - ndtr(aR)) / p
    return Q, dQ_dw, dQ_dx0


def render_profile(w, h, B, x0, pitch, s_o, half_window, phase=0.0, noise=0.0, rng=None, quantise=False):
    a, b = pixel_edges(pitch, half_window, phase)
    Q, _, _ = slot_template(w, x0, a, b, s_o)
    mu = B - h * Q
    if noise > 0:
        rng = rng or np.random.default_rng()
        mu = mu + rng.normal(0, noise, mu.shape)
    if quantise:
        mu = np.clip(np.round(mu), 0, 255)
    return mu, 0.5 * (a + b)


def fisher(w, h, x0, pitch, s_o, noise, half_window, phase=0.0):
    """Fisher information matrix for theta = (w, h, x0, B), Gaussian noise of std `noise`."""
    a, b = pixel_edges(pitch, half_window, phase)
    Q, dQw, dQx = slot_template(w, x0, a, b, s_o)
    J = np.stack([-h * dQw, -Q, -h * dQx, np.ones_like(Q)], axis=1)
    return J.T @ J / noise ** 2


def crlb(w, h, pitch, s_o, noise, half_window=None, h_known=False, h_prior_cv=None, n_phase=16):
    """Phase-averaged CRLB on w (variance, um^2).

    h_known    : contrast exactly known (drop it from theta).
    h_prior_cv : Gaussian prior on h with relative std h_prior_cv (Bayesian / Van Trees bound;
                 the prior information 1/(cv*h)^2 is added to F_hh).
    """
    if half_window is None:
        half_window = max(3.0 * w, 6.0 * np.hypot(s_o, pitch), 3.0 * pitch)
    out = []
    for ph in (np.arange(n_phase) + 0.5) / n_phase:
        F = fisher(w, h, 0.0, pitch, s_o, noise, half_window, ph)
        if h_known:
            F = np.delete(np.delete(F, 1, 0), 1, 1)
        elif h_prior_cv is not None:
            F = F.copy()
            F[1, 1] += 1.0 / (h_prior_cv * h) ** 2
        try:
            out.append(np.linalg.inv(F)[0, 0])
        except np.linalg.LinAlgError:
            out.append(np.inf)
    out = np.asarray(out)
    return float(np.mean(out[np.isfinite(out)])) if np.isfinite(out).any() else np.inf


def crlb_area(w, h, pitch, s_o, noise, half_window=None, n_phase=16):
    """Phase-averaged CRLB on the deficit area A = w*h (delta method), um*grey squared."""
    if half_window is None:
        half_window = max(3.0 * w, 6.0 * np.hypot(s_o, pitch), 3.0 * pitch)
    g = np.array([h, w, 0.0, 0.0])
    out = []
    for ph in (np.arange(n_phase) + 0.5) / n_phase:
        F = fisher(w, h, 0.0, pitch, s_o, noise, half_window, ph)
        out.append(g @ np.linalg.solve(F, g))
    return float(np.mean(out))


# ─────────────────────────────────────────────────────────────────────────────
#  Small-width asymptotics
# ─────────────────────────────────────────────────────────────────────────────

def asymptotic_rel_sd(w, h, pitch, s_o, noise, known=False):
    """Leading-order relative sd(w)/w from the continuum approximation (pixel sums -> integrals,
    pixel box folded into s_eff). Unknown contrast: the w-information comes from the w^2*q''
    term after projecting out span{q, q', 1} (q = LSF); known contrast: from q itself."""
    s = np.hypot(s_o, pitch / np.sqrt(12.0))
    if known:
        # sum_k q_k^2 ~ (1/p) int G^2 = 1/(2 sqrt(pi) s p), then subtract the B-projection (negligible
        # for windows >> s). Var(w) = noise^2 / (h^2 * sum q^2).
        info = h ** 2 / noise ** 2 * (1.0 / (2.0 * np.sqrt(np.pi) * s * pitch))
        return np.sqrt(1.0 / info) / w
    # Column for w orthogonal to h: -(h w^2/12) q''. ||P q''||^2 with q'' projected off q (q' and 1 are
    # orthogonal to q'' by parity / integration): int G''^2 = 3/(8 sqrt(pi) s^5); <G'',G> = -1/(4 sqrt(pi) s^3);
    # int G^2 = 1/(2 sqrt(pi) s)  ->  ||P G''||^2 = 3/(8 sqrt(pi) s^5) - (1/(16 pi s^6)) * 2 sqrt(pi) s
    #                                           = (3/8 - 1/8)/(sqrt(pi) s^5) = 1/(4 sqrt(pi) s^5).
    perp = 1.0 / (4.0 * np.sqrt(np.pi) * s ** 5) / pitch
    info = (h * w ** 2 / 12.0) ** 2 * perp / noise ** 2
    return np.sqrt(1.0 / info) / w


def min_resolvable_width(pitch, s_o, noise, h, rel_tol=0.25, known=False, h_prior_cv=None,
                         w_grid=None):
    """Smallest w whose (exact, phase-averaged) CRLB relative sd is <= rel_tol."""
    w_grid = np.geomspace(1.0, 3000.0, 160) if w_grid is None else w_grid
    for w in w_grid:
        v = crlb(w, h, pitch, s_o, noise, h_known=known, h_prior_cv=h_prior_cv, n_phase=8)
        if np.isfinite(v) and np.sqrt(v) / w <= rel_tol:
            return float(w)
    return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
#  Monte-Carlo check of the bound with a maximum-likelihood estimator
# ─────────────────────────────────────────────────────────────────────────────

def ml_fit(y, centres, pitch, s_o, half_window, h_known=None, w0=None):
    """Least-squares (= ML for Gaussian noise) fit of (w, h, x0, B) by variable projection:
    grid over (w, x0), closed-form (B, h) for each, then local refinement."""
    from scipy.optimize import least_squares
    a, b = centres - pitch / 2, centres + pitch / 2
    wg = np.geomspace(2.0, max(4.0 * half_window, 10.0), 90)
    xg = np.linspace(-pitch, pitch, 9)
    best = (np.inf, None)
    for w in wg:
        for x0 in xg:
            Q, _, _ = slot_template(w, x0, a, b, s_o)
            if h_known is None:
                Z = np.c_[np.ones_like(Q), -Q]
                c, *_ = np.linalg.lstsq(Z, y, rcond=None)
                if c[1] <= 0:
                    continue
                r = y - Z @ c
            else:
                Bc = np.mean(y + h_known * Q)
                c = np.array([Bc, h_known])
                r = y - (Bc - h_known * Q)
            rss = float(r @ r)
            if rss < best[0]:
                best = (rss, (w, c[1], x0, c[0]))
    if best[1] is None:                  # no dark dip at any template: the crack is not detected
        return float(wg[0])
    w, h, x0, B = best[1]

    def resid(t):
        hh = t[1] if h_known is None else h_known
        Q, _, _ = slot_template(t[0], t[2], a, b, s_o)
        return y - (t[3] - hh * Q)

    lo = [0.1, 0.0, -2 * pitch, 0.0]
    hi = [8 * half_window, 400.0, 2 * pitch, 400.0]
    t0 = np.clip([w, h, x0, B], np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    sol = least_squares(resid, t0, bounds=(lo, hi), x_scale=[max(w, 1.0), 50.0, pitch, 50.0])
    return float(sol.x[0])


def monte_carlo(out_dir: str, n_rep: int = 150, seed: int = 0):
    """RMSE of the ML width estimate vs the CRLB, contrast unknown and known."""
    rng = np.random.default_rng(seed)
    h, B, noise = 160.0, 170.0, 4.0
    rows = []
    for pitch in (40.0, 125.0, 278.0):            # 25, 8, 3.6 px/mm
        s_o = 0.7 * pitch
        hw = max(800.0, 6 * pitch)
        for w in (25.0, 50.0, 100.0, 200.0, 400.0, 800.0):
            for known in (False, True):
                est = []
                for _ in range(n_rep):
                    y, c = render_profile(w, h, B, rng.uniform(-pitch / 2, pitch / 2), pitch, s_o, hw,
                                          phase=rng.uniform(), noise=noise, rng=rng)
                    est.append(ml_fit(y, c, pitch, s_o, hw, h_known=h if known else None))
                est = np.asarray(est)
                bound = np.sqrt(crlb(w, h, pitch, s_o, noise, half_window=hw, h_known=known))
                rows.append(dict(pitch_um=pitch, px_per_mm=round(1000 / pitch, 2), w_um=w,
                                 contrast="known" if known else "unknown",
                                 ml_rmse_um=round(float(np.sqrt(np.mean((est - w) ** 2))), 2),
                                 ml_bias_um=round(float(np.mean(est - w)), 2),
                                 ml_mad_um=round(float(np.median(np.abs(est - w))), 2),
                                 crlb_sd_um=round(float(bound), 2)))
                print(rows[-1], flush=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rows, open(os.path.join(out_dir, "crlb_montecarlo.json"), "w"), indent=1)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def figures(out_dir: str, h: float = 160.0, noise: float = 4.0, psf_frac: float = 0.7,
            contrast_cv: float = 0.3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    # (a) relative CRLB vs w / s_eff at a phone-like pitch, unknown / prior / known contrast
    pitch = 125.0
    s_o = psf_frac * pitch
    s_eff = np.hypot(s_o, pitch / np.sqrt(12))
    ws = np.geomspace(5, 2000, 70)
    cur = {k: [] for k in ("unknown", "prior", "known", "asym_unknown", "asym_known")}
    for w in ws:
        cur["unknown"].append(np.sqrt(crlb(w, h, pitch, s_o, noise)) / w)
        cur["prior"].append(np.sqrt(crlb(w, h, pitch, s_o, noise, h_prior_cv=contrast_cv)) / w)
        cur["known"].append(np.sqrt(crlb(w, h, pitch, s_o, noise, h_known=True)) / w)
        cur["asym_unknown"].append(asymptotic_rel_sd(w, h, pitch, s_o, noise))
        cur["asym_known"].append(asymptotic_rel_sd(w, h, pitch, s_o, noise, known=True))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    x = ws / s_eff
    ax.loglog(x, cur["unknown"], "k-", label="contrast unknown")
    ax.loglog(x, cur["asym_unknown"], "k:", label=r"asymptote $\propto (s_{eff}/w)^3$")
    ax.loglog(x, cur["prior"], "C1-", label=f"contrast prior (CV {contrast_cv:.0%})")
    ax.loglog(x, cur["known"], "C0-", label="contrast known")
    ax.loglog(x, cur["asym_known"], "C0:", lw=1)
    ax.axhline(0.25, color="0.6", lw=0.8, ls="--")
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel(r"crack width / effective blur  $w/s_{eff}$")
    ax.set_ylabel(r"CRLB relative sd  $\sigma_w / w$")
    ax.set_title(f"p = {pitch:.0f} um ({1000/pitch:.0f} px/mm), PSF 0.7 px, SNR h/noise = {h/noise:.0f}", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_crlb_regimes.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "fig_crlb_regimes.pdf"))
    plt.close(fig)
    summary["regimes"] = dict(pitch_um=pitch, s_eff_um=float(s_eff), w_um=ws.tolist(),
                              **{k: [float(v) for v in vals] for k, vals in cur.items()})

    # (b) minimum width measurable to +-25 % vs resolution and SNR (cube-root law)
    res = np.array([252, 100, 50, 25, 12.5, 8, 5, 3.6, 2.5])
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    mins = {}
    for snr, style in ((10, "C3-"), (40, "k-"), (160, "C2-")):
        m = [min_resolvable_width(1000 / r, psf_frac * 1000 / r, h / snr, h) for r in res]
        mins[f"unknown_snr{snr}"] = m
        ax.loglog(res, m, style, marker="o", ms=3, label=f"contrast unknown, SNR {snr}")
    # (a contrast prior is not shown: its floor equals the contrast CV, so a 25 % target is either
    #  reached by the image alone or never)
    ax.loglog(res, 2000 / res, color="0.5", lw=0.8, ls=":", label="2 px rule")
    ax.set_xlabel("image resolution (px/mm)")
    ax.set_ylabel("min. width with CRLB sd <= 25 % (um)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.25)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_min_resolvable_width.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "fig_min_resolvable_width.pdf"))
    plt.close(fig)
    summary["min_resolvable_width_um"] = dict(px_per_mm=res.tolist(), **mins)

    # (c) cube-root law: fitted exponent of w_min vs noise at fixed pitch
    noises = np.geomspace(0.5, 32, 8)
    wm = np.array([min_resolvable_width(125.0, 87.5, n, h) for n in noises])
    ok = np.isfinite(wm)
    slope = float(np.polyfit(np.log(noises[ok]), np.log(wm[ok]), 1)[0])
    summary["wmin_vs_noise"] = dict(pitch_um=125.0, noise=noises.tolist(), w_min_um=wm.tolist(),
                                    loglog_slope=slope)
    json.dump(summary, open(os.path.join(out_dir, "crlb_summary.json"), "w"), indent=1)
    print(f"w_min ~ noise^{slope:.3f} (theory 1/3)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["figures", "montecarlo"])
    ap.add_argument("--out", default="docs/results/limits")
    ap.add_argument("--reps", type=int, default=150)
    a = ap.parse_args()
    if a.cmd == "figures":
        figures(a.out)
    else:
        monte_carlo(a.out, n_rep=a.reps)


if __name__ == "__main__":
    main()
