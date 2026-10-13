import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.conformal_width import conformal_quantile, decisions  # noqa: E402
from scripts.evaluate_limits_krkCMd import coarse_rows, deficit_batch  # noqa: E402
from scripts.evaluate_resolution_krkCMd import coarse_row  # noqa: E402
from scripts.photometric_width import deficit_features, rectangle_transform  # noqa: E402
from scripts.width_information_limit import (asymptotic_rel_sd, crlb, pixel_edges,  # noqa: E402
                                             render_profile, slot_template)


def test_slot_template_conserves_area_and_has_exact_derivatives():
    a, b = pixel_edges(125.0, 1500.0, 0.3)
    Q, dw, dx = slot_template(50.0, 10.0, a, b, 87.5)
    assert abs((Q * 125.0).sum() - 50.0) < 1e-6
    eps = 1e-4
    Qw, _, _ = slot_template(50.0 + eps, 10.0, a, b, 87.5)
    Qx, _, _ = slot_template(50.0, 10.0 + eps, a, b, 87.5)
    assert np.abs((Qw - Q) / eps - dw).max() < 1e-6
    assert np.abs((Qx - Q) / eps - dx).max() < 1e-6


def test_unknown_contrast_bound_grows_with_cube_of_blur_over_width():
    p, so, h, n = 125.0, 87.5, 160.0, 4.0
    r = [np.sqrt(crlb(w, h, p, so, n)) / w for w in (10.0, 20.0)]
    assert 6.5 < r[0] / r[1] < 9.0                     # (20/10)^3 = 8
    assert abs(np.log(r[0] / asymptotic_rel_sd(10.0, h, p, so, n))) < 0.2


def test_known_contrast_or_prior_keeps_bound_finite_below_resolution():
    p, so, h, n = 125.0, 87.5, 160.0, 4.0
    unknown = np.sqrt(crlb(20.0, h, p, so, n)) / 20.0
    known = np.sqrt(crlb(20.0, h, p, so, n, h_known=True)) / 20.0
    prior = np.sqrt(crlb(80.0, h, p, so, n, h_prior_cv=0.3)) / 80.0
    assert unknown > 50 and known < 0.5
    assert 0.25 < prior < 0.35                          # floor = contrast CV


def test_photometric_width_is_blur_invariant_and_works_below_one_pixel():
    k = np.arange(-6, 7)
    for w in (30.0, 300.0):                             # 0.24 px and 2.4 px at 125 um pitch
        P = []
        for so in (1.0, 60.0, 150.0):
            mu, _ = render_profile(w, 170.0, 170.0, 0.0, 125.0, so, 6 * 125.0)
            P.append(deficit_features(mu, k, 125.0)["P"])
        assert np.ptp(P) / w < 0.03
        assert abs(np.mean(P) - w) / w < 0.03           # black interior: contrast = background


def test_rectangle_transform_recovers_resolved_black_slot():
    mu, _ = render_profile(400.0, 180.0, 180.0, 0.0, 10.0, 1.0, 800.0)
    assert abs(rectangle_transform(mu, 10.0) - 400.0) / 400.0 < 0.02          # black floor -> b_hat = 0
    grey, _ = render_profile(400.0, 120.0, 180.0, 0.0, 10.0, 1.0, 800.0)      # floor 60 > cap 20
    assert rectangle_transform(grey, 10.0) < 400.0                            # assumed-black bottom under-reads


def test_vectorised_imaging_matches_reference_implementation():
    rng = np.random.default_rng(1)
    X = 170 - 120 * np.exp(-0.5 * ((np.arange(501) - 250) / 12.0) ** 2)[None, :] + rng.normal(0, 3, (3, 501))

    class Fixed:
        def __init__(self, j):
            self.j = j

        def uniform(self, a, b, size=None):
            return self.j if size is None else np.asarray(self.j).reshape(size)

        def normal(self, m, s, size=None):
            return np.zeros(size)

    j = np.array([-0.4, 0.0, 0.3])
    rows, k = coarse_rows(X, 125.0, 0.7, 0.0, Fixed(j[:, None]))
    for i in range(3):
        r1, k1 = coarse_row(X[i], 125.0, 0.7, 0.0, Fixed(j[i]))
        assert np.array_equal(r1, rows[i].astype(float)) and np.array_equal(k1, k)
    P, B, _ = deficit_batch(rows, k, 125.0)
    assert abs(P[0] - deficit_features(rows[0], k, 125.0)["P"]) < 1e-9


def test_conformal_quantile_is_finite_sample_order_statistic():
    s = np.arange(1, 100, dtype=float)                  # n = 99
    assert conformal_quantile(s, 0.95) == 95.0          # ceil(100*0.95) = 95th smallest
    assert conformal_quantile(s, 0.05) == 5.0
    assert conformal_quantile(np.arange(5.0), 0.99) == np.inf


def test_compliance_decision_rule():
    lo = np.array([100.0, 250.0, 320.0])
    hi = np.array([280.0, 350.0, 400.0])
    d = decisions(lo, hi, (lo + hi) / 2, np.array([200.0, 310.0, 360.0]), 300.0)
    assert abs(d["frac_conclusive"] - 2 / 3) < 1e-9
    assert d["false_conform_given_exceed"] == 0.0


def test_pipeline_photometric_width_is_resolution_robust():
    import cv2
    from scripts.crack_width_measure import measure_crack_width
    from scripts.ridge_crack_segmentation import segment_cracks

    rng = np.random.default_rng(0)
    out = {}
    for ppm in (25.0, 4.0):
        n = int(40 * ppm)
        x = (np.arange(n) - n / 2) / ppm * 1000
        Q, _, _ = slot_template(300.0, 0.0, x - 500 / ppm, x + 500 / ppm, 700 / ppm)
        img = np.clip(np.tile(170 - 160 * Q, (n, 1)) + rng.normal(0, 4, (n, n)), 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        r = measure_crack_width(segment_cracks(bgr), bgr, known_length_px=ppm, known_length_mm=1.0)
        out[ppm] = r
    p25, p4 = out[25.0].photometric, out[4.0].photometric
    assert p25["available"] and p4["available"]
    assert abs(p25["median_mm"] - p4["median_mm"]) / p25["median_mm"] < 0.15
    assert p4["decision"] in ("conforms", "does_not_conform", "inconclusive")
    assert p4["interval90_mm"][0] <= p4["p95_mm"] <= p4["interval90_mm"][1]
    # the edge-based width drifts much more than the photometric one over the same change
    assert abs(out[4.0].median_width_mm - out[25.0].median_width_mm) > 2 * abs(p25["median_mm"] - p4["median_mm"])


def test_capture_planner_needs_more_resolution_near_the_limit():
    from scripts.capture_planner import p_conclusive, plan, required_px_per_mm
    import json
    t = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "docs", "results", "limits", "decidability.json")))
    far = required_px_per_mm(t, "cnn", 300.0, 600.0)
    near = required_px_per_mm(t, "cnn", 300.0, 420.0)
    assert far is not None and near is not None and near > far
    assert required_px_per_mm(t, "cnn", 300.0, 300.0) is None          # at the limit: never decidable
    assert p_conclusive(t, "cnn", 300.0, 252.0, 600.0) > p_conclusive(t, "cnn", 300.0, 2.5, 600.0)
    p = plan(0.45, 0.3, 3.6, method="cnn", focal_mm=6.0, pixel_um=1.4)
    assert p["achievable"] and p["closer_factor"] > 1 and p["max_camera_distance_cm"] > 0


def test_gauge_pairs_and_quantised_gauge():
    import pandas as pd
    from scripts.gauge_anchoring_krkCMd import gauge, pairs
    d = pd.DataFrame(dict(series="s", image=1, gridline=["a", "a", "b", "b", "c", "c"], stage=[1, 2, 1, 2, 1, 2],
                          ref_um=[100, 80, 0, 0, 200, 150], est=[110, 90, 5, np.nan, 190, 140]))
    p = pairs(d)
    assert list(p.gridline) == ["a", "c"]                                # healed and unmeasured rows dropped
    g = gauge(np.array([112.0, 137.0]), dict(step=50.0, noise=0.0), np.random.default_rng(0))
    assert list(g) == [100.0, 150.0]


def test_element_table_takes_max_over_locations():
    import pandas as pd
    from scripts.element_decisions_krkCMd import element_table
    d = pd.DataFrame(dict(series="s", image=1, stage=[1, 1, 1, 2, 2], ref_um=[100, 300, 50, 0, 20],
                          est=[120, 250, 400, 10, np.nan]))
    e = element_table(d, "est")
    assert e.loc[("s", 1, 1)].ref_max == 300 and e.loc[("s", 1, 1)].est_max == 400
    assert e.loc[("s", 1, 2)].n == 1
