# StructuralAI — AI-Powered Structural Health Monitoring System

StructuralAI has two parts that share one code base:

1. **An inspection app** (FastAPI backend + React frontend): crack detection and segmentation in
   photos, crack-width measurement against IS 456 limits, structural health / risk / remaining-life
   prediction, SHAP explanations, physics-based engineering models (fracture mechanics, UPV crack
   depth, chloride durability, epoxy grouting) and an LLM inspection report.
2. **A research study on photographic crack-width measurement** (`docs/RESULTS_limits.md`): how much a
   photo can tell about a crack's width, which estimators work at which resolution, and how to give a
   certified "conforms / does not conform / inconclusive" verdict. This is the novel, validated part of
   the project and the basis of the planned paper (`paper/NOVELTY.md`).

## Contents

- [Start here (new contributors)](#start-here-new-contributors)
- [What is validated and what is not](#what-is-validated-and-what-is-not)
- [Research result: limits of photographic crack-width measurement](#research-result-limits-of-photographic-crack-width-measurement)
- [Architecture overview](#architecture-overview)
- [Tech stack](#tech-stack)
- [Datasets](#datasets)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [API reference](#api-reference)
- [Measuring crack width correctly](#measuring-crack-width-correctly)
- [Reproducing the research](#reproducing-the-research)
- [Training the app models](#training-the-app-models)
- [Tests](#tests)
- [Validation details and limits](#validation-details-and-limits)
- [Known issues](#known-issues)
- [Open work / next steps](#open-work--next-steps)
- [License and dataset attribution](#license-and-dataset-attribution)

## Start here (new contributors)

Read in this order:

1. This README, especially [What is validated and what is not](#what-is-validated-and-what-is-not).
2. `docs/RESULTS_limits.md` — the main study: every number, figure and caveat (sections 1-10).
3. `paper/NOVELTY.md` — what is new vs. published work (with verified references), draft abstract,
   honest limitations.
4. `paper/REVIEWER_CRITIQUE.md` — what a reviewer will attack and what is still open.
5. `paper/REPRODUCIBILITY.md` — exact commands to regenerate every result.
6. Earlier, narrower studies: `docs/RESULTS_krkCMd.md` (1-D FWHM resolution study) and
   `docs/RESULTS_krkCMd_2D.md` (first 2-D check).

Rules the project follows (please keep them):

* Every accuracy number is measured **leave-one-series-out** on real data (calibrate on three specimen
  series, test on the fourth); confidence intervals are cluster bootstraps over scanned images.
* Results that are not validated say so, in the code, in API responses (`warnings`) and in the docs.
  Do not report a number from the synthetic tabular data or from simulation as if it were field accuracy.
* Third-party data stays out of git (`data/krkCMd_raw/`, `krk_bands/`, `outputs/` are ignored).

## What is validated and what is not

| Component | Status | Evidence |
|---|---|---|
| Crack-width information limit (Cramér-Rao theory) | Derived, checked by Monte Carlo | `docs/RESULTS_limits.md` §1 |
| Width estimators vs. resolution (FWHM, rectangle transform, photometric, fusion, CNN) | Validated on 14,198 real krkCMd profiles, re-imaged by simulation | §2, §6 |
| End-to-end width on real images (segmentation + width) | Validated on 6 held-out scanner bands | §7 |
| Conformal intervals and three-way compliance decisions | Validated across specimen series | §4, §8 |
| Self-healing tracking, gauge anchoring, capture planner | Validated on krkCMd | §5, §9, §10 |
| Real phone photographs | **Not yet tested** (simulated degradation only) | `scripts/width_test_chart.py` prepared |
| Crack detectors (CrackNet / Swin / PI-CrackNet) and U-Net | Trained on cropped patches; fail on full-scene photos | [Validation details](#validation-details-and-limits) |
| SHI / risk / RUL models | Trained on 350 **synthetic** rows; not predictive of real buildings | same |
| Durability, grouting, fracture, UPV models | Engineering formulas, not validated against field data | same |

## Research result: limits of photographic crack-width measurement

**Question.** At what image resolution can a crack's width still be measured from a photo, by which
method, and when should the software refuse to give a code-compliance verdict?

**Main findings** (krkCMd, Zenodo 10.5281/zenodo.14568863: 14,198 real crack profiles re-imaged at
2.5-252 px/mm, phone-like blur and noise, leave-one-series-out):

* **Theory** (`scripts/width_information_limit.py`): once a crack is narrower than the image blur, only
  its deficit area (width x darkness) is measurable. With unknown contrast the best possible relative
  error grows with the **cube** of blur/width, and the smallest measurable width improves only with the
  cube root of the signal-to-noise ratio. The "2-pixel rule" is this limit at SNR ~10.
* **Edge-based widths** (FWHM, thresholding — the classic approach and this app's original method) fall to
  "no better than guessing the average width" below ~8 px/mm. **Deficit-based widths** (the photometric
  width of `scripts/photometric_width.py`; the rectangle transform of Benz & Rodehorst 2021) do not.
* A **resolution-conditioned 1-D CNN** (`scripts/learned_width_estimator.py`) is the most accurate
  estimator at every resolution, but its advantage over the physics-based estimate vanishes exactly where
  theory says only the deficit is identifiable.
* Real cracks have a **~100 um dark bevelled rim** outside the operator's edges (photometric width =
  0.92 w + 102 um); self-healing shrinks it from 122 to 93 um.
* **Certified decisions** (`scripts/conformal_width.py`): a naive "estimate <= 0.3 mm" check passes
  24-100 % of non-compliant cracks at phone-like resolutions; the conformal rule keeps false passes
  <= 2.4 % and says "inconclusive" instead of guessing.

Mean absolute error (um), phone-like imaging, leave-one-series-out (no-skill baseline 112 um):

| px/mm | sub-pixel FWHM | rectangle transform [Benz 2021] | photometric | FWHM+photometric fusion | learned CNN |
|---|---|---|---|---|---|
| 252 | 37 | 37 | 45 | 37 | 16 |
| 25 | 46 | 41 | 46 | 44 | 23 |
| 8 | 99 | 46 | 52-53 | 45-46 | 38 |
| 3.6 | 112 | 70-76 | 64-68 | 57-58 | 56 |

(The ranges reflect two row sets: all rows each method measures, and rows every method measures; see
`docs/results/limits/summary.csv` and `compare_mae_phone_like.csv`.)

Further results (whole-element verdicts, gauge-anchored monitoring, capture planning, self-healing,
2-D end-to-end) are in `docs/RESULTS_limits.md` §5-§10. Tables: `docs/results/limits/`.
Paper figures: `paper/figures/fig_limits_*`, `fig_crlb_regimes`, `fig_min_resolvable_width`.

## Architecture overview

```
Image upload ─► crack detector (SwinCrackNet / CrackNet / PI-CrackNet) ─► bounding boxes
             ├► U-Net segmentation ─┐
             └► ridge filter (training-free, scripts/ridge_crack_segmentation.py) ─┤
                                    ▼
                     skeleton + local normals ─► profiles across the crack
                        ├► sub-pixel FWHM width (edge-based, crack_width_measure.py)
                        └► photometric width + 90 % conformal interval + decision
                           (photometric_width.py, conformal_width.py, capture_planner.py)
                                    ▼
                     IS 456 severity / compliance, durability model, UPV depth, grouting plan

Structural parameters ─► XGBoost / RandomForest ─► SHI, risk class, RUL ─► SHAP ─► Gemini report + PDF
```

Two frontends talk to the same FastAPI backend: a React SPA (primary) and a legacy Streamlit
dashboard (`app.py`).

## Tech stack

- **Backend:** FastAPI, Uvicorn
- **Computer vision / deep learning:** PyTorch, Torchvision (Swin Transformer backbone), OpenCV,
  scikit-image, Albumentations
- **Statistics / research code:** NumPy, SciPy, pandas, Matplotlib
- **Classical ML:** scikit-learn, XGBoost; **explainability:** SHAP
- **LLM / agent:** LangChain + Google Gemini (`gemini-2.5-flash` by default, set `GEMINI_MODEL`)
- **Reporting:** ReportLab (PDF)
- **Frontend:** React, Axios, Recharts; **legacy dashboard:** Streamlit

## Datasets

The repo ships pre-populated `data/` folders for the app models. **All third-party datasets below are
for research use; check each license before commercial use** (see
[License and dataset attribution](#license-and-dataset-attribution)).

| Dataset | Used for | Location in repo | Source |
|---|---|---|---|
| **krkCMd** (Jakubowski & Tomczak, Sci. Data 2025) — 19,098 brightness profiles across real concrete cracks, 252 px/mm scans, manual reference widths, 6 self-healing stages, 4 specimen series, 36 images. **CC BY 4.0** | The whole crack-width research study | `data/krkCMd_raw/krkCMd_table.csv` (56 MB, **not in git**); image bands in `krk_bands/` (not in git) | [Zenodo 10.5281/zenodo.14568863](https://doi.org/10.5281/zenodo.14568863) — download command in [Reproducing the research](#reproducing-the-research) |
| **Concrete Crack Images for Classification** (Özgenel & Sorguç) — 20,000 + 20,000 227x227 patches | Training `CrackNet` / `SwinCrackNet` classifiers | `data/crack_detection/Positive`, `.../Negative` | [Mendeley Data](https://data.mendeley.com/datasets/5y9wdsg2zt/2) · [Kaggle mirror](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) |
| **SDNET2018** (Utah State University) — 56,000+ cracked/uncracked decks, pavements, walls | Supplementary classifier training data | `data/crack_detection/sdnet2018/{D,P,W}/{C*,U*}` | [USU Digital Commons](https://digitalcommons.usu.edu/all_datasets/48/) |
| **DeepCrack** (Liu et al. 2019) — pixel-level crack masks | Training `TinyUNet`; converted to YOLO boxes | `data/segmentation/deepcrack/` | [GitHub: yhlleo/DeepCrack](https://github.com/yhlleo/DeepCrack) |
| **CrackForest** (Shi et al., IEEE T-ITS 2016) — road crack images | Segmentation ground truth / benchmarking | `data/segmentation/` | [GitHub: cuilimeng/CrackForest-dataset](https://github.com/cuilimeng/CrackForest-dataset) |
| **YOLO-format crack set** — from DeepCrack masks (`scripts/prepare_yolo_dataset.py`) | Legacy YOLOv8 detector | `data/yolo_dataset/`, `data/yolo_dataset_tiny/` | Derived in-repo (`dataset.yaml`) |
| **Synthetic structural tabular data** — 350 rows from a formula + noise | SHI / risk / RUL models | `data/tabular/structural_data.csv` | Generated: `python scripts/generate_tabular_data.py` |

`data/knowledge_base/` is an empty placeholder (not used by any code path).

## Project structure

```
backend/main.py            FastAPI app — all REST endpoints (see API reference)
main.py                    Uvicorn entrypoint (port from STRUCTURALAI_PORT, default 8000)
app.py                     Legacy Streamlit dashboard (same backend)
frontend-react/            React SPA (Detection page = image width measurement + verdict banner)

scripts/  — app models
  cracknet.py, swin_cracknet.py, physics_informed_cracknet.py   classifier architectures
  detect_cracknet.py, detect_swin_cracknet.py, detect_physics_informed_cracknet.py   sliding-window detectors
  train_cracknet.py, train_swin_cracknet.py, train_physics_informed_cracknet.py, train_unet.py, train_ml.py
  ridge_crack_segmentation.py   training-free crack segmentation (multi-scale Sato ridge filter)
  crack_width_measure.py        image pipeline: skeleton, FWHM width, IS 456, photometric verdict
  physics_fracture_engine.py    LEFM fracture mechanics + residual capacity
  acoustic_optical_inversion.py photo trace + one measured UPV depth (IS 13311)
  crack_width_durability_model.py   chloride ingress (Fick) -> service-life impact
  grouting_controller.py        closed-loop epoxy injection controller
  agent.py, generate_report.py, explain.py, extract_features.py   LLM agent, PDF report, SHAP, features
  generate_tabular_data.py      synthetic tabular dataset
  train.py / detect.py / predict.py / prepare_yolo_dataset.py   legacy YOLOv8 pipeline

scripts/  — crack-width research study (docs/RESULTS_limits.md)
  width_information_limit.py    forward model, Fisher information, Cramér-Rao bounds, Monte Carlo
  photometric_width.py          photometric (deficit-area) width + rectangle-transform baseline
  evaluate_limits_krkCMd.py     main study: all estimators x 9 resolutions x 5 imaging conditions
  learned_width_estimator.py    resolution-conditioned 1-D CNN (LOSO folds; --nested for calibration)
  conformal_width.py            conformal intervals, compliance decisions, deployment calibration export
  compare_estimators_krkCMd.py  paired estimator comparison with cluster-bootstrap CIs
  differential_width_krkCMd.py  self-healing / change monitoring
  element_decisions_krkCMd.py   whole-element (max width) certified decisions
  gauge_anchoring_krkCMd.py     one gauge reading at installation + later photos
  capture_planner.py            decidability map, "how much closer to re-shoot"
  certified_width.py            app: all five certified stages (w_c, estimators, intervals, decisions, anchoring)
  export_app_calibration.py     app calibration tables (w_c look-up, CNN, anchoring, element maximum)
  make_demo_image.py            test photo (real krkCMd crack or synthetic) for trying the app
  evaluate_limits_2d_krkCMd.py  2-D end-to-end check on held-out image bands
  make_limits_figures.py        paper figures
  fetch_krkCMd_band.py          downloads one image band from the 38.6 GB Zenodo zip by range requests
  evaluate_resolution_krkCMd.py, evaluate_end_to_end_krkCMd.py, resolution_error_model.py,
  validate_width_synthetic.py, width_test_chart.py, make_paper_figures.py   earlier FWHM studies + test chart

models/                    trained weights (Git LFS), incl. models/width_net/ (4 CNN fold models)
docs/RESULTS_limits.md     main research results;  docs/results/limits/  its tables + calibration files
docs/RESULTS_krkCMd*.md    earlier studies;        docs/results/          their tables
paper/                     NOVELTY.md, REVIEWER_CRITIQUE.md, REPRODUCIBILITY.md, references.bib, figures/
tests/                     test_core.py (app engines), test_limits.py (research code + pipeline)
data/                      datasets (see above);  outputs/  regenerated study outputs (not in git)
```

## Setup

### Quick start on a new laptop (copy-paste)

Needs Python 3.11 or 3.12, Node.js 18+, Git and [Git LFS](https://git-lfs.com) (install Git LFS
**before** cloning, otherwise the model weights arrive as small text pointers). About 3 GB of disk
with PyTorch; the clone itself is ~340 MB.

**Windows (PowerShell)**

```powershell
git lfs install
git clone https://github.com/2005lakshya/Structural_AI.git
cd Structural_AI
git lfs ls-files                      # must list 15 weight files under models/
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env                # optional: Gemini key for the AI report / chat
cd frontend-react; npm ci; cd ..
```

**macOS / Linux**

```bash
git lfs install
git clone https://github.com/2005lakshya/Structural_AI.git
cd Structural_AI
git lfs ls-files                      # must list 15 weight files under models/
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env                  # optional
(cd frontend-react && npm ci)
```

(With an NVIDIA GPU use the CUDA wheel instead of `/whl/cpu`, e.g. `/whl/cu121`; the app itself runs
fine on CPU.)

**Check the install, then run** (two terminals, venv active in the first):

```bash
python -m pytest tests -q             # 26 tests, about 1-2 min, all must pass
python main.py                        # backend on http://127.0.0.1:8000  (GET /health -> {"status":"ok"})
cd frontend-react && npm start        # frontend on http://localhost:3000
```

If port 8000 is taken: start the backend with `STRUCTURALAI_PORT=8010` (PowerShell:
`$env:STRUCTURALAI_PORT=8010; python main.py`) and the frontend with
`REACT_APP_API_URL=http://127.0.0.1:8010` (or put it in `frontend-react/.env`, see
`frontend-react/.env.example`). `.claude/launch.json` already uses 8010 for both.

**Try the certified measurement:**

```bash
python scripts/make_demo_image.py --real --px-per-mm 12.5   # real krkCMd crack, downloads ~150 MB once
python scripts/make_demo_image.py                           # or: synthetic line, no download
```

In the app open **Detection**, upload `outputs/demo/demo_crack_real.png`, choose **Reference Object**,
set "Reference size in image (px)" = 12.5 and "Known physical size (mm)" = 1, and click **Calculate
Crack Width from Image**. The "Certified measurement" panel should show w_c = 0.018 mm, the network at
0.27 mm with a 90 % interval 0.16-0.35 mm (reference 0.29 mm), INCONCLUSIVE at the widest 5 % and
DOES NOT CONFORM for the element maximum (limit 0.2 mm), and "re-shoot about 5.3x closer".

What is **not** in git and when you need it: the krkCMd table and image bands (only for reproducing the
research, see [Reproducing the research](#reproducing-the-research)), `outputs/` (regenerated; also holds
gauge-anchoring records in `outputs/anchors.json`), `.env` (optional API key) and the datasets for
re-training the vision models ([Datasets](#datasets)). Everything needed to run the app and the tests is
in the repo.

### Prerequisites

- Python 3.11+ (tested on 3.11 and 3.12)
- Node.js 18+ and npm (tested on Node 22)
- Git LFS (for the model weights)
- Optional: an NVIDIA GPU (only for re-training the CNN width estimator, ~25 min + ~60 min nested)
- Optional: a Google Gemini API key (AI narrative report and chat; rule-based fallback without it)

### Backend

```bash
conda create -n structural_ai python=3.11
conda activate structural_ai
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # or the CPU build
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is git-ignored; copy `.env.example`):

```
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

If port 8000 is in use, set `STRUCTURALAI_PORT` (backend) and `REACT_APP_API_URL` (frontend), e.g.
`http://127.0.0.1:8010`.

### Model weights (Git LFS)

The trained weights (`models/**/*.pth`, `models/**/*.pkl`, `models/*.pt`, ~170 MB) are stored with
[Git LFS](https://git-lfs.com). They are tracked via `.gitattributes`; do not add them to `.gitignore`.

**On a new machine, install Git LFS before cloning:**

```bash
git lfs install
git clone https://github.com/2005lakshya/Structural_AI.git
```

For an existing clone, run `git lfs install` then `git pull`.

**Check it worked:** `git lfs ls-files` lists the weight files, and `models/ml/shi_regressor.pkl` should
be real binary data. If weights are tiny text files starting with `version https://git-lfs...`, run
`git lfs install` and `git lfs pull`.

A new `*.pth`, `*.pkl` or `models/*.pt` is picked up automatically. Files outside those patterns (e.g.
`yolov8n.pt` in the repo root, downloaded by Ultralytics) are **not** LFS-tracked — keep them out of git.
GitHub's free LFS quota is 1 GB storage / 1 GB bandwidth per month, so avoid committing many versions.

### Frontend

```bash
cd frontend-react
npm install
```

Default backend URL is `http://localhost:8000` (`src/services/api.js`); override with
`frontend-react/.env` → `REACT_APP_API_URL=...`.

## Running the app

```bash
python main.py                          # backend (or: uvicorn backend.main:app --reload --port 8000)
cd frontend-react && npm start          # frontend, then open http://localhost:3000
streamlit run app.py                    # optional legacy dashboard
```

`.claude/launch.json` defines the same two servers (`backend`, `frontend`) for the Claude Code preview.

## API reference

| Method | Route | What it does |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/detect` | Crack detection (Swin preferred, CrackNet fallback) |
| POST | `/analyze_image` | Detection + U-Net segmentation |
| POST | `/analyze_thermal` | Thermal anomaly detection on an IR image |
| POST | `/measure_crack_width_image` | Width from a photo (U-Net mask): FWHM widths, IS 456 check, **`photometric` verdict**, durability impact |
| POST | `/measure_crack_width_ridge` | Same with the training-free ridge filter (`roi`, `exclude` boxes); use for full-scene photos |
| POST | `/calculate_crack_width` | Design crack width per IS 456 Annex F from structural inputs |
| POST | `/calculate_durability_impact` | Width -> chloride-ingress corrosion initiation / service-life impact |
| POST | `/calculate_crack_depth` | Penetration depth, rebar exposure, grout volume estimate |
| POST | `/calculate_upv_depth` | UPV crack depth (IS 13311 Part 1 / BS 1881-203) |
| POST | `/predict_risk`, `/explain_risk` | SHI / risk / RUL prediction and SHAP plot (synthetic-data models) |
| POST | `/analyze_structure`, `/chat`, `/generate_report` | LLM agent, chat, PDF report |
| GET | `/model_performance` | Stored evaluation metrics |
| POST | `/api/unified_invention_pipeline`, `/api/pump/simulate_step` | Combined acoustic-optical / grouting demo (API only, not in the UI) |

**The `photometric` field** (both width endpoints) — the resolution-robust result:

```json
{
  "available": true,
  "median_mm": 0.221, "p95_mm": 0.2355,
  "interval90_mm": [0.0945, 0.427],
  "limit_mm": 0.2,
  "decision": "conforms | does_not_conform | inconclusive",
  "n_profiles": 300,
  "basis": "how the number was obtained",
  "capture_advice": {"advice": "Re-shoot about 4.2x closer (>= 15.3 px/mm).", "required_px_per_mm": 15.3, "...": "..."}
}
```

`available: false` comes with a `reason` (no calibration table, resolution below 2.5 px/mm, too few
centre-line points). `capture_advice` appears only when the decision is inconclusive. The calibration
and conformal offsets come from `docs/results/limits/photometric_calibration.json`; the planner uses
`docs/results/limits/decidability.json`. Both are regenerated by `scripts/conformal_width.py` and
`scripts/capture_planner.py map`.

**The `certified` field** (both width endpoints) — all five stages of the certified measurement
(`scripts/certified_width.py`), shown in the UI as the "Certified measurement" panel:

```json
{
  "available": true, "primary": "photo", "limit_mm": 0.2, "pixels_per_mm": 8.0, "n_profiles": 203,
  "measurability": {"wc_mm": 0.0075, "background_grey": 169.5, "noise_grey": 2.22, "snr": 76.2, "measurable": true},
  "estimators": {
    "photo":  {"p95_mm": 0.19, "interval90_mm": [0.076, 0.339], "decision": "inconclusive",
               "element_max_mm": 0.198, "element_interval90_mm": [0.103, 0.285], "element_decision": "inconclusive", "...": "..."},
    "fusion": {"...": "..."}, "cnn": {"...": "..."}
  },
  "decision": "inconclusive", "element_decision": "inconclusive",
  "capture_advice": {"advice": "..."},
  "anchoring": {"status": "installed | anchored | no_anchor", "estimators": {"photo": {"width_mm": 0.2, "...": "..."}}}
}
```

1. **Measurability limit** `wc_mm`: smallest width this photo can determine (Cramer-Rao bound, contrast
   prior 30 %, blur 0.7 px, relative sd <= 50 %) from its pixel pitch and the noise/texture of the
   background next to the crack; replaces the fixed 2-pixel rule (`min_resolvable_width_mm`).
2. **Estimators**: the CNN (ensemble of the four leave-one-series-out models in `models/width_net/`;
   primary, it decides), the photometric width (primary if the weights are missing) and the FWHM +
   photometric fusion. The photometric/fusion intervals are too wide to certify a crack as exceeding
   (their 0.3 mm map is ~0 above the limit at every resolution); the CNN decides most cracks >= 0.4 mm at
   >= 25 px/mm.
3. **90 % interval** per estimator (photometric/fusion offsets from `photometric_calibration.json`, CNN CQR
   offsets from `app_calibration.json`).
4. **Decision** at the 95th-percentile location and for the **element maximum** (max-statistic offsets).
5. **Capture advice** when inconclusive; with the lens focal length and sensor pixel size
   (`lens_focal_mm`, `pixel_size_um`, photo used uncropped at full resolution) also the maximum camera
   distance d <= f / (r* s).

Gauge anchoring: pass `location_id` and, on the first visit only, a gauge reading (`anchor_gauge_mm` on
`/measure_crack_width_image`, `gauge_reading_mm` on `/measure_crack_width_ridge`). It is stored in
`outputs/anchors.json`; later photos of the same `location_id` get `w* = w - beta (w1 - g1)` with its own
interval and decision, plus the self-healing ratio 1 - w / w1 between this photo and the installation
photo. `docs/results/limits/app_calibration.json` (w_c look-up table, CNN offsets,
anchoring beta/offsets, element offsets) is regenerated by `python scripts/export_app_calibration.py`.

## Measuring crack width correctly

- **A real scale is required for real mm values.** In the UI, click the two ends of an object of known
  length (ruler, coin, gauge) and enter its length, or use scanner DPI. The camera-distance mode is a
  rough guess and can be off by several times.
- **Resolution decides what is measurable.** Below ~8 px/mm the FWHM width (the large "severity" numbers)
  over-reads narrow cracks badly; trust the **photometric verdict** banner instead. If it says
  *inconclusive*, follow its re-shoot advice or measure with a crack gauge.
- **Hairline cracks** (e.g. 0.04 mm) need >= 50 px/mm (macro lens or microscope) or the gauge itself.
- **Calibration caveat:** the photometric calibration reproduces the krkCMd operator's edge definition
  on laboratory concrete with dark crack interiors. Wet, filled, painted or light-coloured cracks, and
  other concretes, need checking with gauge readings (`lab_reading_mm` / `gauge_reading_mm` inputs).
- The U-Net can flag ink marks, aggregate edges and background. Crop to the concrete, or use
  `/measure_crack_width_ridge` with an `roi`.

## Reproducing the research

```bash
# data (CC BY 4.0), 56 MB
mkdir -p data/krkCMd_raw
curl -L -o data/krkCMd_raw/krkCMd_table.csv https://zenodo.org/api/records/14568863/files/krkCMd_table.csv/content

python scripts/width_information_limit.py figures --out docs/results/limits     # theory (~1 min)
python scripts/width_information_limit.py montecarlo --out docs/results/limits  # bound check (~10 min)
python scripts/evaluate_limits_krkCMd.py --out outputs/limits                   # main study (~3 min)
python scripts/learned_width_estimator.py --steps 10000                         # CNN folds (~25 min GPU)
python scripts/learned_width_estimator.py --nested --steps 6000                 # nested models (~60 min GPU)
python scripts/conformal_width.py                                               # intervals, decisions, app calibration
python scripts/compare_estimators_krkCMd.py
python scripts/differential_width_krkCMd.py
python scripts/element_decisions_krkCMd.py
python scripts/gauge_anchoring_krkCMd.py
python scripts/capture_planner.py map
python scripts/export_app_calibration.py                                        # app tables (~30 s)
python scripts/make_limits_figures.py --res outputs/limits --out paper/figures
```

The 2-D check needs ~0.9 GB of image bands fetched by range requests (`remotezip`); the exact loop is
in `paper/REPRODUCIBILITY.md`. All steps use fixed seeds. `outputs/` holds the large intermediate files
(~60 MB measurement table, simulated pixel rows); only small tables are copied to `docs/results/limits/`.

## Training the app models

All training scripts read from `data/` and write weights to `models/`.

```bash
python scripts/train_cracknet.py        # CrackNet CNN classifier
python scripts/train_swin_cracknet.py   # SwinCrackNet transformer classifier
python scripts/train_physics_informed_cracknet.py
python scripts/train_unet.py            # TinyUNet segmentation
python scripts/train_ml.py              # SHI / risk / RUL tabular models
python scripts/generate_tabular_data.py # regenerate synthetic tabular dataset
```

The legacy YOLOv8 pipeline (no longer called from the backend) can be trained with
`python scripts/train.py`.

## Tests

```bash
pip install pytest
python -m pytest tests          # ~1-2 min, 23 tests
```

`tests/test_core.py` covers the engineering engines, ridge filter and FWHM routine.
`tests/test_limits.py` covers the forward model and its exact derivatives, the cube law, blur
invariance of the photometric width, the rectangle transform, the vectorised imaging simulation,
conformal quantiles and decisions, the capture planner, gauge anchoring, element tables and the
image-pipeline photometric verdict. The planner and pipeline tests read the committed JSON tables in
`docs/results/limits/`.

## Validation details and limits

**Crack width (research study).** See `docs/RESULTS_limits.md`, section "What this does NOT show":
simulated resolutions from scanner images (no real phone photos yet), one dataset / one lab / four
specimen series, one operator's reference (5-operator RMS 16.6 um), dark-interior contrast assumption,
conformal coverage 0.85-0.95 by series around a pooled 0.90, element-level coverage fragile for one series.

**Earlier FWHM studies.** `docs/RESULTS_krkCMd.md` and `docs/RESULTS_krkCMd_2D.md`;
`scripts/resolution_error_model.py plan` (FWHM-based camera-distance rule, superseded by
`scripts/capture_planner.py`). `scripts/width_test_chart.py` generates a printable known-width chart and
analyses photos of it — the real-camera experiment has **not** been run.

**Segmentation.** The learned U-Net and the sliding-window detectors were trained on tightly cropped
concrete patches and fail on full-scene photos (they flag clothing, shadows, hands). Responses include
`box_coverage_fraction` and a `warning` when boxes cover > 50 % of the frame.

**Risk / SHI / RUL models** are trained on 350 rows generated from a formula plus noise. Their accuracy
shows they learned that formula, not that they predict real buildings. The repair cost is a fixed
formula. The risk model does not recover the "Safe" class on the held-out split.

**Reported U-Net Dice (0.7345)** is the value logged at training time; it cannot be recomputed because
the segmentation dataset is not in the repo. `/model_performance` labels it as such.

**Acoustic-optical stage** (`scripts/acoustic_optical_inversion.py`) combines the photo's crack trace
and widths with ONE measured UPV depth (IS 13311). It does not estimate a depth profile or void volume.
The grouting plan only runs when you supply `void_volume_ml`. Default UPV times (`t1_us`, `t2_us`) are
placeholders; enter your own readings.

## Known issues

- The image datasets for the app models are not all reproducible from the repo; retraining the vision
  models requires downloading them first.
- The CrackNet checkpoint is from epoch 2 (val acc 0.66); the Swin checkpoint is from the first
  head-only phase. The PI-CrackNet checkpoint (val acc 0.888) is the one used.
- `TinyUNet` is defined in three places (`backend/main.py`, `scripts/train_unet.py`,
  `scripts/extract_features.py`).
- Saved sklearn/XGBoost models were pickled with scikit-learn 1.9.0 and warn on newer versions;
  re-run `python scripts/train_ml.py` to regenerate them.
- The app decides with the **CNN**; the photometric width and the fusion are shown beside it. The CNN
  is an ensemble of the four leave-one-series-out fold models; its offsets are pooled over those folds, not
  from a model trained on all four series.
- On the real 0.04 mm core photo (`roi=170,540,135,150`, assumed 4 px/mm) every estimator reads about
  0.36-0.39 mm and the 90 % intervals exclude 0.04 mm: the krkCMd calibration does not transfer to that
  photo (edge damage and dirt read as crack). Gauge anchoring is the intended remedy for such locations.
- The FWHM "severity level" and the photometric verdict can disagree at low resolution; the
  photometric one is the validated one.
- Chat / LLM report need `GEMINI_API_KEY` (untested without a key).
- `reports/latest_annotated.jpg` is overwritten by local runs; do not commit run-generated versions.
- `/api/unified_invention_pipeline` and `/api/pump/simulate_step` are API-only (not in the UI).

## Open work / next steps

1. **Real-camera validation (highest priority for the paper):** print the chart from
   `scripts/width_test_chart.py`, photograph it with 2-3 phones at several distances, and photograph real
   cracks with a crack-comparator card in frame; compare against the simulated predictions.
2. **Single deployment CNN:** the app uses the four fold models as an ensemble; train one model on all
   four series and calibrate it from the nested models.
3. **Second dataset** with physical (gauge/microscope) widths to test generalisation.
4. **Paper:** the study has ~10 contributions; consider two papers — (a) limits + estimators + certified
   decisions, (b) monitoring (self-healing, gauge anchoring, capture planning). See `paper/NOVELTY.md`.
5. Add a code license file before any public release.

## License and dataset attribution

This repository's own code has no license file yet — add one before any public release.

Third-party datasets retain their original licenses and citation requirements:

- **krkCMd** — CC BY 4.0. Cite Jakubowski & Tomczak, *"Dataset for developing deep learning models to
  assess crack width and self-healing progress in concrete,"* Scientific Data 12, 165 (2025),
  doi:10.1038/s41597-025-04485-z, and the Zenodo record 10.5281/zenodo.14568863.
- **CrackForest Dataset** — non-commercial research use only. Cite Shi et al., *"Automatic Road Crack
  Detection Using Random Structured Forests,"* IEEE T-ITS, 2016.
- **DeepCrack** — cite Liu et al., *"DeepCrack: A Deep Hierarchical Feature Learning Architecture for
  Crack Segmentation,"* Neurocomputing, 2019.
- **SDNET2018** and **Concrete Crack Images for Classification** — check the source pages above for
  current license terms before commercial use.

Bibliography for the research study (all DOIs verified): `paper/references.bib`.
