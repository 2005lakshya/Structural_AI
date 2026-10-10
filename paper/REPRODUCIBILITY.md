# Reproducibility

Environment: conda, Python 3.11, numpy 2.4, scipy 1.17, scikit-image 0.26, OpenCV 5, pandas 3 (see `requirements.txt`).
All stochastic steps use fixed seeds (`--seed 0`).

```bash
# 0. data (CC BY 4.0, Zenodo 10.5281/zenodo.14568863)
mkdir -p data/krkCMd_raw
curl -L -o data/krkCMd_raw/krkCMd_table.csv https://zenodo.org/api/records/14568863/files/krkCMd_table.csv/content

# 1. 1-D resolution study  (~20 min CPU)           -> outputs/krkCMd/, copied to docs/results/
python scripts/evaluate_resolution_krkCMd.py --out outputs/krkCMd

# 2. error model, calibration, planner             -> docs/results/resolution_error_model.json
python scripts/resolution_error_model.py fit --meas outputs/krkCMd/all_measurements.csv.gz
python scripts/resolution_error_model.py plan --width-mm 0.3 --tol 0.25 --focal-mm 6 --pixel-um 1.4

# 3. 2-D end-to-end (needs ~0.5 GB of image bands via range requests)
for spec in "CMd_0.23_2mths 1 1" "CMd_0.23_2mths 1 4" "CMd_0.28_20mths 1 1" "CMd_0.28_20mths 1 4"; do
  set -- $spec; python scripts/fetch_krkCMd_band.py $1 $2 $3 krk_bands; done
python scripts/evaluate_end_to_end_krkCMd.py --bands krk_bands --out outputs/krkCMd_2d

# 4. figures, tests
python scripts/make_paper_figures.py
python -m pytest tests
```

## Not reproducible / not done
* The real-camera printed-chart experiment (`scripts/width_test_chart.py`) has **not** been run on physical prints.
* The 2-D study uses 4 of 36 images (one image, two stages, two series).
* `outputs/krkCMd/all_measurements.csv.gz` (~100 MB uncompressed) is not committed; regenerate with step 1.
* The manuscript itself is not in this repository; `paper/` holds only figures, the bibliography and notes.
