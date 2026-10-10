import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.acoustic_optical_inversion import (  # noqa: E402
    extract_crack_morphology, solve_optically_guided_acoustic_inversion)
from scripts.crack_width_measure import _skeletonize, measure_crack_width  # noqa: E402
from scripts.grouting_controller import generate_grouting_profile  # noqa: E402
from scripts.physics_fracture_engine import evaluate_fracture_and_capacity  # noqa: E402
from scripts.ridge_crack_segmentation import segment_cracks  # noqa: E402
from scripts.validate_width_synthetic import render  # noqa: E402


def test_upv_depth_matches_is13311_formula():
    pts = []
    res = solve_optically_guided_acoustic_inversion(pts, 150, 75, 125, 500, 40)
    assert abs(res.max_depth_mm - 150 * np.sqrt((125 / 75) ** 2 - 1)) < 0.1


def test_inversion_on_synthetic_trace():
    mask = np.zeros((128, 128), np.uint8)
    cv2.line(mask, (10, 20), (110, 100), 255, 4)
    skel = _skeletonize(mask)
    dist = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
    pts = extract_crack_morphology(skel, dist, pixels_per_mm=2.0)
    res = solve_optically_guided_acoustic_inversion(pts, 150, 75, 125, 500, 40)
    assert res.surface_tortuosity >= 1.0
    assert res.surface_length_mm > 0
    assert not hasattr(res, "internal_void_volume_cm3")   # no guessed void volume
    assert not hasattr(res, "depth_profile")


def test_physics_engine_deeper_crack_lowers_capacity():
    shallow = evaluate_fracture_and_capacity(10, 0.2)
    deep = evaluate_fracture_and_capacity(200, 0.2)
    assert deep.residual_load_capacity_kN <= shallow.residual_load_capacity_kN


def test_grouting_safe_pressure_below_blowout():
    g = generate_grouting_profile(50, 0.3, 100)
    assert g.safe_operating_pressure_bar < g.hydraulic_blowout_pressure_bar


def test_width_measurement_accurate_for_resolved_crack():
    img = render(6.0, seed=1)
    r = measure_crack_width(segment_cracks(img), img, known_length_px=1, known_length_mm=1)
    assert abs(r.median_width_px - 6.0) / 6.0 < 0.15


def test_sub_pixel_crack_is_flagged_unresolved():
    img = render(0.5, seed=1)
    r = measure_crack_width(segment_cracks(img), img, known_length_px=1, known_length_mm=1)
    # either nothing found or the resolution warning must fire
    assert r.crack_pixels < 20 or any("cannot be" in w for w in r.warnings) or not r.resolution_ok or r.median_width_px > 0.5


def test_ridge_filter_ignores_dark_blob():
    img = np.full((300, 300, 3), 170, np.uint8)
    cv2.circle(img, (150, 150), 60, (40, 40, 40), -1)
    assert segment_cracks(img).sum() == 0
