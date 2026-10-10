"""
resolution_error_model.py
=========================
Fits a compact predictive model of the absolute width-measurement error as a
function of image resolution, from the krkCMd re-imaging study, and turns it into a
planning tool ("how close must I photograph a crack of width w?").

Model (per imaging scenario):   MAE(p) = a + c * p**k          [um]
    p : pixel pitch on the surface in um (= 1000 / px_per_mm)
    a : resolution-independent floor (reference/definition offset + noise)
    c, k : pixel-limited growth

It is validated leave-one-series-out: fit on 3 specimen series, predict the 4th.
It describes THIS dataset's cracks (0.1-0.8 mm, laboratory scans re-imaged to lower
resolution); extrapolation to rough field surfaces is not supported.

CLI:
    python scripts/resolution_error_model.py fit  --meas outputs/krkCMd/all_measurements.csv.gz
    python scripts/resolution_error_model.py plan --width-mm 0.2 --tol 0.25 \
           --focal-mm 6.0 --pixel-um 1.4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

DEFAULT_PARAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "docs", "results", "resolution_error_model.json")


def mae_model(p, a, c, k):
    return a + c * np.power(p, k)


def fit(meas_csv: str, scenario: str = "phone_like", method_col: str = "ours_um", calibrated: bool = False):
    r = pd.read_csv(meas_csv)
    r = r[(r.scenario == scenario) & r[method_col].notna()].copy()
    if calibrated:
        # Leave-one-series-out linear calibration per resolution: ref ~ a*meas + b fitted on the
        # other series, applied to the held-out series (so no test data touches its own calibration).
        out = r.copy()
        for res, g in r.groupby("px_per_mm"):
            for held in g.series.unique():
                tr = g[g.series != held]
                a_, b_ = np.polyfit(tr[method_col], tr.ref_um, 1)
                idx = g.index[g.series == held]
                out.loc[idx, method_col] = a_ * g.loc[idx, method_col] + b_
        r = out
    r["abs_err"] = (r[method_col] - r.ref_um).abs()
    r["pitch"] = 1000.0 / r.px_per_mm
    curves = r.groupby(["series", "pitch"], as_index=False).abs_err.mean()
    series = sorted(curves.series.unique())

    def _fit(df):
        pop, _ = curve_fit(mae_model, df.pitch.to_numpy(), df.abs_err.to_numpy(), p0=[50, 0.05, 1.7],
                           bounds=([0, 0, 0.5], [500, 50, 4]), maxfev=20000)
        return pop

    full = _fit(curves)
    loso = []
    for held in series:
        pop = _fit(curves[curves.series != held])
        te = curves[curves.series == held]
        pred = mae_model(te.pitch.to_numpy(), *pop)
        loso.append(dict(held_out=held,
                         mean_abs_pred_error_um=float(np.mean(np.abs(pred - te.abs_err.to_numpy()))),
                         mean_rel_pred_error=float(np.mean(np.abs(pred - te.abs_err.to_numpy()) / te.abs_err.to_numpy()))))
    # No-skill reference: always predict the mean reference width of the other series.
    ns = []
    base = r[r.pitch == r.pitch.min()].drop_duplicates(subset=["series", "i"]) if "i" in r else r
    for held in series:
        m = base[base.series != held].ref_um.mean()
        ns.append(np.abs(base[base.series == held].ref_um - m).to_numpy())
    noskill = float(np.concatenate(ns).mean())
    pooled = r.groupby("pitch").abs_err.mean()
    resid = pooled.to_numpy() - mae_model(pooled.index.to_numpy(), *full)
    return dict(scenario=scenario, a_um=float(full[0]), c=float(full[1]), k=float(full[2]),
                pooled_rmse_um=float(np.sqrt(np.mean(resid ** 2))), noskill_mae_um=noskill, loso=loso,
                pitches_um=[float(x) for x in pooled.index], pooled_mae_um=[float(x) for x in pooled.to_numpy()])


def expected_mae_um(px_per_mm: float, params: dict) -> float:
    return float(mae_model(1000.0 / px_per_mm, params["a_um"], params["c"], params["k"]))


def required_px_per_mm(width_mm: float, tol: float, params: dict):
    """Smallest resolution (px/mm) whose expected MAE is <= tol * width.
    Returns None if the tolerance is below the model's resolution-independent floor."""
    target = tol * width_mm * 1000.0
    if target <= params["a_um"] * 1.0001:
        return None
    # c * p**k = target - a  ->  p
    p = ((target - params["a_um"]) / params["c"]) ** (1.0 / params["k"])
    return 1000.0 / p


def camera_distance_mm(px_per_mm: float, focal_mm: float, pixel_um: float) -> float:
    """Pinhole model: surface sampling = distance * pixel_pitch / focal  ->  px/mm = f / (D * s)."""
    return focal_mm / (px_per_mm * pixel_um / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--meas", default="outputs/krkCMd/all_measurements.csv.gz")
    f.add_argument("--out", default=DEFAULT_PARAMS)
    pl = sub.add_parser("plan")
    pl.add_argument("--width-mm", type=float, required=True)
    pl.add_argument("--tol", type=float, default=0.25, help="allowed relative error, e.g. 0.25")
    pl.add_argument("--focal-mm", type=float, default=6.0)
    pl.add_argument("--pixel-um", type=float, default=1.4)
    pl.add_argument("--params", default=DEFAULT_PARAMS)
    pl.add_argument("--calibrated", action="store_true",
                    help="assume the width estimate has been linearly calibrated on reference data")
    a = ap.parse_args()

    if a.cmd == "fit":
        res = {s: fit(a.meas, s) for s in ("ideal_scan", "phone_like")}
        res["phone_like_calibrated"] = fit(a.meas, "phone_like", calibrated=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        for s, v in res.items():
            print(s, f"MAE(p) = {v['a_um']:.1f} + {v['c']:.4f} * p^{v['k']:.2f} um | pooled RMSE {v['pooled_rmse_um']:.1f} um")
            for l in v["loso"]:
                print("   held-out", l["held_out"], f"pred err {l['mean_abs_pred_error_um']:.1f} um ({100*l['mean_rel_pred_error']:.0f} %)")
    else:
        params = json.load(open(a.params))["phone_like_calibrated" if a.calibrated else "phone_like"]
        need = required_px_per_mm(a.width_mm, a.tol, params)
        print(f"crack width {a.width_mm} mm, tolerance +-{a.tol*100:.0f} % (= +-{a.tol*a.width_mm*1000:.0f} um)")
        if need is None:
            print(f"NOT ACHIEVABLE: tolerance is below the method's resolution-independent error floor "
                  f"(~{params['a_um']:.0f} um mean absolute error on this dataset).")
        else:
            d = camera_distance_mm(need, a.focal_mm, a.pixel_um)
            print(f"required resolution >= {need:.1f} px/mm  ->  camera distance <= {d/10:.1f} cm "
                  f"(focal {a.focal_mm} mm, pixel {a.pixel_um} um, pinhole model)")
            print(f"expected mean abs error at that resolution: {expected_mae_um(need, params):.0f} um")
            ns = params.get("noskill_mae_um")
            if ns and expected_mae_um(need, params) >= ns:
                print(f"WARNING: that is no better than guessing the dataset-average width (~{ns:.0f} um MAE): "
                      "the measurement carries no information at this resolution.")


if __name__ == "__main__":
    main()
