# Reproducibility

Environment: Python 3.11, numpy 2.0, scipy 1.16, scikit-image 0.26, OpenCV 4.12, pandas 2.3, PyTorch 2.5
(CUDA optional; the learned estimator trains in ~20 min on an RTX 3060 laptop GPU). See `requirements.txt`.
All stochastic steps use fixed seeds.

```bash
# 0. data (CC BY 4.0, Zenodo 10.5281/zenodo.14568863), 56 MB
mkdir -p data/krkCMd_raw
curl -L -o data/krkCMd_raw/krkCMd_table.csv https://zenodo.org/api/records/14568863/files/krkCMd_table.csv/content

# 1. information limit: CRLB figures and Monte-Carlo check (~10 min CPU)   -> docs/results/limits/
python scripts/width_information_limit.py figures --out docs/results/limits
python scripts/width_information_limit.py montecarlo --out docs/results/limits --reps 120

# 2. real-data study, all estimators x 9 resolutions x 5 imaging conditions (~3 min)  -> outputs/limits/
python scripts/evaluate_limits_krkCMd.py --out outputs/limits

# 3. learned resolution-conditioned estimator, leave-one-series-out (~20 min GPU)
python scripts/learned_width_estimator.py --steps 10000
#    12 nested models for shift-aware conformal calibration of the learned estimator (~60 min GPU)
python scripts/learned_width_estimator.py --nested --steps 6000

# 4. conformal intervals, compliance decisions, deployment calibration table
python scripts/conformal_width.py
# 5. self-healing (differential) analysis, paired estimator comparison
python scripts/differential_width_krkCMd.py
python scripts/compare_estimators_krkCMd.py

# 5b. whole-element decisions, gauge-anchored monitoring, decidability map / capture planner
python scripts/element_decisions_krkCMd.py
python scripts/gauge_anchoring_krkCMd.py
python scripts/capture_planner.py map

# 6. 2-D end-to-end on held-out specimens (~0.9 GB of range requests, 6 bands)
for spec in "CMd_0.28_20mths 1 1" "CMd_0.28_20mths 1 6" "CMd_0.23_20mths 8 1" "CMd_0.23_20mths 8 6" \
            "CMd_0.28_2mths 3 1" "CMd_0.28_2mths 3 6"; do
  set -- $spec; python scripts/fetch_krkCMd_band.py $1 $2 $3 krk_bands; done
python scripts/evaluate_limits_2d_krkCMd.py --bands krk_bands --out outputs/limits_2d

# 7. figures, tests
python scripts/make_limits_figures.py --res outputs/limits --out paper/figures
python -m pytest tests
```

Earlier resolution study (FWHM only, `docs/RESULTS_krkCMd.md`, `docs/RESULTS_krkCMd_2D.md`):

```bash
python scripts/evaluate_resolution_krkCMd.py --out outputs/krkCMd
python scripts/resolution_error_model.py fit --meas outputs/krkCMd/all_measurements.csv.gz
for spec in "CMd_0.23_2mths 1 1" "CMd_0.23_2mths 1 4" "CMd_0.28_20mths 1 1" "CMd_0.28_20mths 1 4"; do
  set -- $spec; python scripts/fetch_krkCMd_band.py $1 $2 $3 krk_bands; done
python scripts/evaluate_end_to_end_krkCMd.py --bands krk_bands --out outputs/krkCMd_2d
```

## Not reproducible / not done
* The real-camera printed-chart experiment (`scripts/width_test_chart.py`) has **not** been run on physical prints.
* `outputs/` (measurements ~60 MB, simulated pixel rows) is not committed; regenerate with steps 2-6.
* The manuscript itself is not in this repository; `paper/` holds figures, the bibliography and notes.
