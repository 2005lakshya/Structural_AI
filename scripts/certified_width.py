"""
certified_width.py
==================
The five linked stages of the certified photo measurement, as used by the image pipeline
(scripts/crack_width_measure.py) and the API:

  1. measurability limit w_c of THIS photo (Cramer-Rao bound from its pixel pitch, blur model and
     the noise/texture measured on the background next to the crack)
  2. resolution-robust widths: photometric (deficit area / background), FWHM + photometric fusion,
     and the resolution-conditioned network (ensemble of the leave-one-series-out models)
  3. 90 % conformal interval per estimator, calibrated per resolution on specimens the estimator
     was not fitted on
  4. three-way decision CONFORMS / DOES NOT CONFORM / INCONCLUSIVE against the code limit, at the
     95th-percentile location and for the element maximum
  5. re-capture guidance from the decidability map when the decision is inconclusive
  +  optional gauge anchoring: one gauge reading at installation corrects later photos of the
     same location, w* = w - beta (w1 - g1)

Calibration tables: docs/results/limits/photometric_calibration.json (photo, fusion) and
docs/results/limits/app_calibration.json (w_c look-up table, network, anchoring, element maximum),
written by scripts/conformal_width.py --export and scripts/export_app_calibration.py.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os

import numpy as np

from scripts.conformal_width import EPS_UM, z, zinv
from scripts.photometric_width import window_half_um

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMITS_DIR = os.path.join(ROOT, "docs", "results", "limits")
ANCHOR_FILE = os.path.join(ROOT, "outputs", "anchors.json")
# decides when available (the patent's decidability results are the network's); photometric otherwise
PRIMARY_ORDER = ("cnn", "photo")
METHOD_LABELS = {"photo": "Photometric width", "fusion": "Edge + photometric fusion",
                 "cnn": "Resolution-conditioned network"}
MIN_NOISE_GREY = 2.0

_TABLES: dict = {}
_NETS = None


def _json(name):
    if name not in _TABLES:
        p = os.path.join(LIMITS_DIR, name)
        _TABLES[name] = json.load(open(p)) if os.path.exists(p) else None
    return _TABLES[name]


def _interp_table(table, px_per_mm, get):
    """Linear interpolation of get(row) in log(px/mm), clamped to the table range."""
    rows = sorted(table, key=lambda r: r["px_per_mm"])
    grid = np.log([r["px_per_mm"] for r in rows])
    return float(np.interp(np.log(px_per_mm), grid, [get(r) for r in rows]))


def _decide(lo, hi, L):
    return "conforms" if hi <= L else ("does_not_conform" if lo > L else "inconclusive")


def _mm(v_um):
    return round(float(v_um) / 1000.0, 4)


# ── stage 1 ─────────────────────────────────────────────────────────────────
def measurability_limit_um(pitch_um: float, snr: float) -> float:
    """w_c from the look-up table (bilinear in log pitch, log SNR, clamped to the table)."""
    lut = _json("app_calibration.json")["measurability"]
    P, S, W = np.log(lut["pitch_um"]), np.log(lut["snr"]), np.log(np.array(lut["wc_um"]))
    p, s = np.clip(np.log(pitch_um), P[0], P[-1]), np.clip(np.log(snr), S[0], S[-1])
    i = int(np.clip(np.searchsorted(P, p) - 1, 0, len(P) - 2))
    j = int(np.clip(np.searchsorted(S, s) - 1, 0, len(S) - 2))
    fp, fs = (p - P[i]) / (P[i + 1] - P[i]), (s - S[j]) / (S[j + 1] - S[j])
    v = (W[i, j] * (1 - fp) * (1 - fs) + W[i + 1, j] * fp * (1 - fs) + W[i, j + 1] * (1 - fp) * fs
         + W[i + 1, j + 1] * fp * fs)
    return float(np.exp(v))


def background_noise(rows: np.ndarray, k: np.ndarray, pitch_um: float):
    """Background grey level and noise (sensor noise plus surface texture): standard deviation of
    the pixels outside the integration window of every profile, median over profiles."""
    outer = np.abs(k * pitch_um) > window_half_um(pitch_um)
    if outer.sum() < 4:
        outer = np.zeros(len(k), bool)
        outer[:2] = outer[-2:] = True
    g = rows[:, outer]
    return float(np.median(np.median(g, axis=1))), max(float(np.median(g.std(axis=1))), MIN_NOISE_GREY)


# ── stage 2 ─────────────────────────────────────────────────────────────────
def _nets():
    global _NETS
    if _NETS is None:
        import torch
        from scripts.learned_width_estimator import WidthNet
        _NETS = []
        for p in sorted(glob.glob(os.path.join(ROOT, "models", "width_net", "width_net_heldout_*.pth"))):
            net = WidthNet()
            net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            net.eval()
            _NETS.append(net)
    return _NETS


def cnn_quantiles(rows: np.ndarray, k: np.ndarray, pitch_um: float):
    """5/50/95 % width quantiles (um) per profile: ensemble of the four leave-one-series-out
    networks, averaged in z = log(w + 20 um). None when the weights are missing."""
    nets = _nets()
    if not nets:
        return None
    import torch
    from scripts.learned_width_estimator import make_inputs
    prof, sc = make_inputs(rows, k, pitch_um)
    with torch.no_grad():
        zq = np.mean([n(torch.from_numpy(prof), torch.from_numpy(sc)).numpy() for n in nets], axis=0)
    w = np.clip(np.exp(zq) - EPS_UM, 0, None)
    return w[:, 0], w[:, 2], w[:, 4]


def profile_estimates(rows: np.ndarray, k: np.ndarray, pitch_um: float, px_per_mm: float):
    """Per-profile point estimates (um) of every estimator, plus the network's 5/95 % quantiles."""
    from scripts.evaluate_limits_krkCMd import deficit_batch
    from scripts.evaluate_resolution_krkCMd import measure_ours
    cal = _json("photometric_calibration.json")["table"]
    P, _, _ = deficit_batch(rows, k, pitch_um)
    okP = np.isfinite(P)
    a = _interp_table(cal, px_per_mm, lambda r: r["photo"]["coef"][0])
    b = _interp_table(cal, px_per_mm, lambda r: r["photo"]["coef"][1])
    photo = np.where(okP, np.clip(a * P + b, 0, None), np.nan)
    fw = np.array([np.nan if (v := measure_ours(r, k.astype(np.float64), pitch_um)) is None else v
                   for r in rows.astype(np.float64)])
    c = [_interp_table(cal, px_per_mm, lambda r, i=i: r["fusion"]["coef"][i]) for i in range(3)]
    fusion = np.where(np.isfinite(fw) & okP, np.clip(c[0] * fw + c[1] * P + c[2], 0, None), photo)
    out = dict(photo=photo, fusion=fusion)
    q = cnn_quantiles(rows, k, pitch_um)
    if q is not None:
        out["cnn"], out["cnn_q05"], out["cnn_q95"] = q[1], q[0], q[2]
    return out


# ── stages 3-4 ──────────────────────────────────────────────────────────────
def location_interval(est: dict, name: str, px_per_mm: float):
    """Point estimate and 90 % interval (um) at the 95th-percentile location along the crack."""
    w = est[name]
    ok = np.isfinite(w)
    w95 = float(np.percentile(w[ok], 95))
    if name == "cnn":
        cal = _json("app_calibration.json")["table"]
        zl = _interp_table(cal, px_per_mm, lambda r: r["cnn"]["z_lo"])
        zh = _interp_table(cal, px_per_mm, lambda r: r["cnn"]["z_hi"])
        i = int(np.nanargmin(np.abs(w - w95)))
        lo, hi = zinv(z(est["cnn_q05"][i]) + zl), zinv(z(est["cnn_q95"][i]) + zh)
    else:
        cal = _json("photometric_calibration.json")["table"]
        zl = _interp_table(cal, px_per_mm, lambda r: r[name]["z_lo"])
        zh = _interp_table(cal, px_per_mm, lambda r: r[name]["z_hi"])
        lo, hi = zinv(z(w95) + zl), zinv(z(w95) + zh)
    return float(np.median(w[ok])), w95, max(float(lo), 0.0), float(hi)


def element_interval(est: dict, name: str, px_per_mm: float):
    """Element maximum: max over as many evenly spaced locations as the calibration elements had,
    with the max-statistic conformal offsets."""
    cal = _json("app_calibration.json")["table"]
    n = int(round(_interp_table(cal, px_per_mm, lambda r: r["element"][name]["n_locations"])))
    w = est[name][np.isfinite(est[name])]
    idx = np.unique(np.linspace(0, len(w) - 1, min(n, len(w))).round().astype(int))
    wmax = float(np.max(w[idx]))
    zl = _interp_table(cal, px_per_mm, lambda r: r["element"][name]["z_lo"])
    zh = _interp_table(cal, px_per_mm, lambda r: r["element"][name]["z_hi"])
    return wmax, max(float(zinv(z(wmax) + zl)), 0.0), float(zinv(z(wmax) + zh)), len(idx)


# ── gauge anchoring ─────────────────────────────────────────────────────────
def _anchors():
    try:
        return json.load(open(ANCHOR_FILE))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def anchoring(location_id: str, gauge_mm: float, points_um: dict, px_per_mm: float, L_um: float):
    """Installation (gauge given): store w1 and g1 for the location. Later photo (no gauge):
    w* = w - beta (w1 - g1) with beta and conformal offsets learned on other specimens, and the
    self-healing ratio HR = 1 - w / w1 from the two photos (photo-to-photo, so errors largely cancel)."""
    anchors = _anchors()
    if gauge_mm > 0:
        now = _dt.datetime.now().isoformat(timespec="seconds")
        anchors[location_id] = dict(g1_um=gauge_mm * 1000.0, w1_um=points_um, px_per_mm=px_per_mm, date=now,
                                    history=[dict(date=now, width_um=points_um)])
        os.makedirs(os.path.dirname(ANCHOR_FILE), exist_ok=True)
        json.dump(anchors, open(ANCHOR_FILE, "w"), indent=1)
        return dict(status="installed", location_id=location_id, gauge_mm=gauge_mm,
                    note="Gauge reading stored for this location; later photos of it will be anchored.")
    a = anchors.get(location_id)
    if a is None:
        return dict(status="no_anchor", location_id=location_id,
                    note="No installation gauge reading stored for this location yet.")
    cal = _json("app_calibration.json")["table"]
    out = dict(status="anchored", location_id=location_id, installed=a["date"], gauge_mm=_mm(a["g1_um"]),
               estimators={})
    a.setdefault("history", []).append(dict(date=_dt.datetime.now().isoformat(timespec="seconds"), width_um=points_um))
    json.dump(anchors, open(ANCHOR_FILE, "w"), indent=1)
    out["visits"] = len(a["history"])
    for name, w in points_um.items():
        if name not in a["w1_um"]:
            continue
        beta = _interp_table(cal, px_per_mm, lambda r: r["anchoring"][name]["beta"])
        ws = max(w - beta * (a["w1_um"][name] - a["g1_um"]), 0.0)
        lo = max(float(zinv(z(ws) + _interp_table(cal, px_per_mm, lambda r: r["anchoring"][name]["z_lo"]))), 0.0)
        hi = float(zinv(z(ws) + _interp_table(cal, px_per_mm, lambda r: r["anchoring"][name]["z_hi"])))
        out["estimators"][name] = dict(width_mm=_mm(ws), interval90_mm=[_mm(lo), _mm(hi)], beta=round(beta, 3),
                                       decision=_decide(lo, hi, L_um),
                                       healing_ratio=round(1.0 - w / a["w1_um"][name], 3) if a["w1_um"][name] > 0 else None)
    if abs(np.log(px_per_mm / a["px_per_mm"])) > np.log(1.25):
        out["warning"] = (f"Installation photo was taken at {a['px_per_mm']:.1f} px/mm, this one at "
                          f"{px_per_mm:.1f} px/mm; anchoring assumes a similar set-up.")
    return out


# ── all stages ──────────────────────────────────────────────────────────────
def assess(rows: np.ndarray, k: np.ndarray, pixels_per_mm: float, limit_mm: float,
           location_id: str = "", gauge_mm: float = 0.0, focal_mm: float = 0.0, pixel_um: float = 0.0) -> dict:
    """Certified assessment of one photographed crack from its perpendicular profiles (rows,
    pixel offsets k at 1-pixel steps, 0 at the centre line). focal_mm / pixel_um (lens focal length,
    sensor pixel size, both > 0) turn the required resolution into a maximum camera distance,
    d <= f / (r* s); the photo must then be used at the camera's full resolution, uncropped."""
    out = dict(available=False)
    if _json("photometric_calibration.json") is None or _json("app_calibration.json") is None:
        out["reason"] = "calibration tables not found (run scripts/export_app_calibration.py)"
        return out
    grid = [r["px_per_mm"] for r in _json("photometric_calibration.json")["table"]]
    if pixels_per_mm < min(grid) * 0.95:
        out["reason"] = f"resolution {pixels_per_mm:.2f} px/mm is below the validated range ({min(grid):g} px/mm)"
        return out
    pitch = 1000.0 / pixels_per_mm
    L = limit_mm * 1000.0
    rows = np.asarray(rows, dtype=np.float64)

    # stage 1
    B, sigma = background_noise(rows, k, pitch)
    wc = measurability_limit_um(pitch, B / sigma)

    # stages 2-4
    est = profile_estimates(rows, k, pitch, pixels_per_mm)
    estimators, points = {}, {}
    for name in ("photo", "fusion", "cnn"):
        if name not in est or np.isfinite(est[name]).sum() < 5:
            continue
        med, w95, lo, hi = location_interval(est, name, pixels_per_mm)
        emax, elo, ehi, n_loc = element_interval(est, name, pixels_per_mm)
        points[name] = w95
        estimators[name] = dict(label=METHOD_LABELS[name], median_mm=_mm(med), p95_mm=_mm(w95),
                                interval90_mm=[_mm(lo), _mm(hi)], decision=_decide(lo, hi, L),
                                measurable=bool(w95 >= wc),
                                element_max_mm=_mm(emax), element_interval90_mm=[_mm(elo), _mm(ehi)],
                                element_decision=_decide(elo, ehi, L), element_locations=n_loc)
    primary = next((m for m in PRIMARY_ORDER if m in estimators), None)
    if primary is None:
        out["reason"] = "no usable profiles across the crack"
        return out
    p = estimators[primary]
    out.update(available=True, primary=primary, limit_mm=limit_mm, pixels_per_mm=round(pixels_per_mm, 3),
               n_profiles=int(len(rows)), estimators=estimators,
               measurability=dict(wc_mm=_mm(wc), background_grey=round(B, 1), noise_grey=round(sigma, 2),
                                  snr=round(B / sigma, 1), measurable=p["measurable"],
                                  basis="Cramer-Rao bound, contrast prior 30 %, blur 0.7 px, relative sd <= 50 %"),
               decision=p["decision"], element_decision=p["element_decision"])

    # stage 5
    if p["decision"] == "inconclusive":
        try:
            from scripts.capture_planner import plan
            out["capture_advice"] = plan(p["p95_mm"], limit_mm, pixels_per_mm, method=primary,
                                         focal_mm=focal_mm or None, pixel_um=pixel_um or None)
            out["capture_advice"]["camera_given"] = bool(focal_mm > 0 and pixel_um > 0)
        except (FileNotFoundError, KeyError, ValueError) as e:
            out["capture_advice"] = dict(advice="Re-shoot closer or use a crack gauge.", error=str(e))

    if location_id:
        out["anchoring"] = anchoring(location_id, gauge_mm, points, pixels_per_mm, L)
    return out
