# Information limits of photographic crack-width measurement (krkCMd)

This is the study behind the paper contribution described in `paper/NOVELTY.md`. All numbers below are
reproducible with the commands in `paper/REPRODUCIBILITY.md`; tables are in `docs/results/limits/`,
figures in `paper/figures/fig_limits_*` and `docs/results/limits/fig_*`.

**Data.** krkCMd (Zenodo 10.5281/zenodo.14568863, CC BY 4.0): 19,098 brightness profiles across real
concrete cracks, scanner pitch 3.97 um (252 px/mm), 3,183 gridlines x 6 self-healing stages (0, 2, 4, 7,
14, 28 days), 4 specimen series, 36 images. Reference: one trained operator marking the steepest edges
(5-operator RMS 16.6 um per the dataset paper, i.e. an MAE floor of about 13 um). 14,198 profiles have an
open crack (reference > 0).

**Protocol.** Each profile is re-imaged at 2.5-252 px/mm: Gaussian PSF (0.7 px "phone-like"; 0 "ideal
scan"; 1.5 px "blurry"), area integration over pixels with random sub-pixel phase, Gaussian noise
(4 grey; 10 "noisy"), 8-bit; "shading" adds a +-25 grey ramp. Every calibration is leave-one-series-out
(LOSO): fitted on three specimen series, tested on the fourth. 95 % CIs are cluster bootstraps over the
36 scanned images. Code: `scripts/evaluate_limits_krkCMd.py`.

## 1. Theory: what a pixel profile can determine (`scripts/width_information_limit.py`)

Model: dark slot (width w, contrast h) on background B, Gaussian PSF s_o, pixel pitch p, noise sigma_n;
closed-form pixel means and exact derivatives; phase-averaged Cramér-Rao bound on w.

* For w << s_eff (s_eff^2 = s_o^2 + p^2/12) the profile depends on (w, h) only through the deficit area
  A = w*h. With contrast unknown the relative bound is

      sd(w)/w >= 12 (sigma_n/h) * sqrt(p) * (4 sqrt(pi))^(1/2) * s_eff^(5/2) / w^3   (cube law)

  and the exact bound follows it within 10 % (`fig_crlb_regimes`). Halving the width multiplies the
  bound by 8.
* The smallest width measurable to +-25 % scales as s_eff x SNR^(-1/3) (fitted exponent 0.38 on the exact
  bound): a 16x better SNR shrinks it only ~2.8x (`fig_min_resolvable_width`, `crlb_summary.json`). The
  practitioners' "2-pixel rule" coincides with this bound at SNR (contrast/noise) ~ 10, i.e. it is the
  information limit of a fairly noisy photo, not a universal constant.
* With the contrast known the bound stays finite (sd ~ sqrt(p s_eff) sigma_n/h); with a contrast prior of
  relative spread CV the bound tends to CV. Below the blur scale a width is therefore only obtainable
  from the deficit area plus contrast knowledge, and its floor is the contrast spread.
* Monte Carlo (`crlb_montecarlo.json`, 120 noisy profiles per cell): the maximum-likelihood estimate
  attains the bound wherever it is finite (e.g. 8 px/mm, w = 100 um, contrast known: RMSE 6.2 vs bound
  6.1 um; 25 px/mm, w = 50 um, unknown: 12.6 vs 12.2 um). Where the bound diverges the ML estimate is
  biased and its error is several times the width.

## 2. Real data: the two regimes (phone-like imaging, MAE in um, LOSO, 95 % CI in `summary.csv`)

| px/mm | sub-pixel FWHM | threshold count | rectangle transform [Benz 2021] | photometric (ours) | FWHM+photometric fusion (ours) | contrast oracle |
|---|---|---|---|---|---|---|
| 252 | 37 | 36 | 37 | 45 | 37 | 2 |
| 50 | 39 | 39 | 39 | 46 | 39 | 4 |
| 25 | 46 | 44 | 41 | 46 | 44 | 8 |
| 12.5 | 72 | 62 | 44 | 49 | 47 | 15 |
| 8 | 99 | 89 | 46 | 53 | 45 | 23 |
| 5 | 110 | 112 | 58 | 57 | 52 | 39 |
| 3.6 | 112 | 108 | 76 | 64 | 57 | 43 |
| 2.5 | 113 | 107 | 98 | 70 | 66 | 48 |

No-skill baseline (always predict the other series' mean width): 112 um. Dataset DLM at native
resolution: 14.9 um (CI 12.2-18.5), but that network was trained on this dataset, so it is not a
held-out number. All rows use LOSO linear calibration of the raw estimator.

* **Edge-based estimators (FWHM, threshold) fall to the no-skill level below ~8 px/mm**, deficit-based
  ones (rectangle transform, photometric) do not. At 3.6 px/mm the fusion is 57 um (CI 53-61) vs FWHM
  112 um (CI 99-126).
* **All estimators collapse onto one curve of relative error vs w/s_eff** (`fig_limits_collapse`):
  FWHM breaks down below w/s_eff ~ 1, close to where the contrast-unknown bound rises through 25 %
  (w/s_eff ~ 1.6); deficit-based estimators follow the shape of the contrast-prior bound. Above
  w/s_eff ~ 2 every estimator sits at a 10-20 % floor set by how the real cracks and the operator's edge
  definition differ from the slot model, not by the image.
* **The published rectangle transform is a strong baseline once calibrated**: it is as good as or better
  than our photometric width down to 8 px/mm; the photometric width is better at <= 5 px/mm (64 vs 76 um at
  3.6, 70 vs 98 um at 2.5 px/mm), because the rectangle transform sets its integration level from the
  observed minimum, which blur raises.
* **What limits the deficit estimators is contrast, not pixels**: given each crack's true contrast
  (oracle) the error at 8 px/mm would be 23 um instead of 53 um.
* Sensitivity (`summary.csv`, scenarios blurry / noisy / shading at 25, 8, 3.6 px/mm): the ordering is
  unchanged; at 3.6 px/mm FWHM stays at ~112 um, the fusion is 66 / 62 / 58 um.
* Ideal scan (no optical blur): same picture; at 3.6 px/mm FWHM 111, rectangle 86, photometric 61,
  fusion 54 um.

Median relative error by true width (phone-like, `error_by_width_bin.csv`): 200-400 um cracks stay at
13-18 % down to 3.6 px/mm with the photometric and fusion estimators (FWHM 14-25 %); 50-100 um cracks
are 52-57 % at 3.6 px/mm (FWHM 199 %); < 50 um cracks have median relative errors of >= 100 % at
<= 25 px/mm for every estimator.

## 3. Crack contrast on real concrete (`contrast_by_group.csv`, `fig_limits_contrast`)

* Crack interiors are essentially black in these scans (median interior grey 0), so the background is a
  usable contrast estimate.
* Photometric width at native resolution: **P = 0.92 w + 102 um**. The deficit extends ~100 um beyond the
  operator's edges: a darker bevelled rim, which the operator excludes by definition. This is the origin
  of the +55 um bias of the FWHM estimator reported in `RESULTS_krkCMd.md`.
* **Self-healing shrinks the rim**: the per-stage offset falls from 122 um (day 0) to 93 um (day 14-28)
  while the slope stays 0.90-0.94. Healing products fill the bevelled edges first.

## 4. Certified compliance decisions (`scripts/conformal_width.py`)

Split-conformal 90 % intervals in log(w + 20 um), calibrated per resolution, nested LOSO (calibration
scores never include the held-out series). Learned estimator: conformalised quantile regression.

* **Coverage under specimen-series shift** (`conformal_coverage.csv`): pooled 0.900-0.905 for the
  photometric intervals at every resolution, 0.89-0.90 rectangle, 0.875-0.92 fusion, 0.89-0.93 FWHM; by
  series 0.85-0.95 (photometric) and 0.74-0.98 (rectangle at 2.5 px/mm). Exchangeability across
  specimens is approximately, not exactly, satisfied.
* **Decisions at the 0.3 mm limit** (`compliance_decisions.csv`): the naive rule "point estimate <= limit"
  declares compliant 24 % (25 px/mm) to 34 % (3.6 px/mm, fusion) of the cracks that actually exceed
  0.3 mm, and 97-100 % with FWHM at <= 8 px/mm. The conformal rule keeps this at <= 2.4 % for every
  estimator and resolution, while still deciding 46-49 % of cracks (fusion, 3.6-25 px/mm); the rest are
  flagged inconclusive. At 3.6 px/mm FWHM is (correctly) never conclusive.
* Same at 0.1 and 0.2 mm: false acceptance <= 2.6 % with the conformal rule; 9-17 % with the naive rule
  (fusion and photometric, 3.6 px/mm).
* The learned estimator's intervals need shift-aware calibration to keep these guarantees (section 6).

## 5. Self-healing monitoring (`scripts/differential_width_krkCMd.py`)

Stage s vs stage 1 of the same gridline, each stage imaged independently.

* Per-crack change is not easier than absolute width once calibrated: the change MAE is close to the
  absolute MAE (fusion 45 vs 45 um, learned 39 vs 37 um at 8 px/mm). Random error, not the shared bias,
  dominates. Sign of changes >= 50 um: 93-94 % correct at 8 px/mm (fusion, learned), 75 % with FWHM.
* **The healing ratio per image and stage** (the quantity self-healing studies report;
  `healing_ratio.csv`) is recovered by the photometric width with r = 0.94 (252 px/mm), 0.92 (8 px/mm),
  0.91 (5 px/mm), 0.84 (3.6 px/mm); MAE 6.5-12 percentage points. **FWHM does not follow healing:**
  r = 0.50 at 252 px/mm and ~0 at <= 50 px/mm, with a -10 to -28 point bias; the rectangle transform
  reaches r = 0.61-0.70 at >= 8 px/mm and fails below. A likely reason: healing changes the rim
  (section 3), which the half-depth FWHM partly includes, differently from the core.

## 6. Learned, resolution-conditioned estimator (`scripts/learned_width_estimator.py`)

1-D CNN (61 k parameters, GroupNorm, FiLM conditioning on log pixel pitch, inputs: profile on a fixed
+-0.8 mm grid divided by the background, plus the photometric width), trained with a pinball loss on
real profiles degraded on the fly (2.5-252 px/mm, PSF 0-1.6 px, noise 1-11 grey, shading). Four models,
each trained without one specimen series and tested on it (same simulated test images as the classical
estimators). Paired comparison on the rows where every estimator returns a value
(`compare_pairs_phone_like.csv`, cluster-bootstrap CIs):

| px/mm | learned (ours) | fusion (ours) | rectangle [Benz 2021] | FWHM | learned - fusion, 95 % CI |
|---|---|---|---|---|---|
| 252 | 15.6 | 37.0 | 36.8 | 37.0 | -21.4 (-24.3, -18.5) |
| 25 | 22.6 | 44.1 | 40.6 | 46.4 | -21.5 (-24.3, -18.7) |
| 12.5 | 30.4 | 47.1 | 43.5 | 72.3 | -16.7 (-19.4, -13.7) |
| 8 | 37.7 | 45.6 | 45.7 | 98.8 | -7.9 (-10.3, -5.4) |
| 5 | 46.9 | 53.1 | 54.9 | 109.9 | -6.1 (-8.6, -3.5) |
| 3.6 | 55.6 | 58.4 | 70.4 | 111.5 | -2.8 (-5.3, -0.5) |
| 2.5 | 65.1 | 67.8 | 92.3 | 112.6 | -2.7 (-5.4, -0.4) |

* Held out by specimen series, it matches the dataset's own network at native resolution (15.6 vs
  14.9 um, the latter trained on this data) and is the best estimator at every resolution.
* **Its advantage vanishes where theory says only the deficit area is identifiable**: 21 um better
  than the physics-based fusion at >= 25 px/mm, under 3 um at <= 3.6 px/mm. A flexible learner does not
  get past the information limit; below the blur scale it converges to the photometric estimate.
* **Its intervals must be calibrated across specimens, not within them.** Calibrated by CQR on withheld
  images of the training series ("same-series"), the 90 % intervals cover only 0.79-0.85 on the held-out
  series, and at 0.3 mm the certified rule wrongly passes 7-10 % of exceeding cracks at <= 5 px/mm.
  Calibrated shift-aware (12 nested models; each fold's scores come from a series that its calibration
  models never saw) coverage is 0.89-0.94 and false acceptance at 0.3 mm is <= 2.2 % at every resolution
  (`conformal_coverage.csv`, methods `cnn_samecal` vs `cnn`). Per-series coverage still ranges 0.75-0.98.
* With shift-aware intervals the learned estimator decides more cracks at 0.3 mm than any classical one:
  77-80 % at >= 25 px/mm, 67 % at 8 px/mm and 48 % at 3.6 px/mm (fusion: 49 %, 49 %, 46 %), at <= 1.5 %
  false acceptance.
* Self-healing: healing-ratio correlation 0.95 (252), 0.90 (8), 0.85 (3.6 px/mm), similar to the
  photometric width, with a -7 to -14 point bias (it under-reads healing).

## 7. End-to-end on real images of held-out specimens (`scripts/evaluate_limits_2d_krkCMd.py`)

Six scanner image bands (3 specimens from the three series not used for ridge-filter tuning, stages 1
and 6; 638 gridline measurements per resolution), degraded to phone-like images, segmented by the frozen
ridge filter, measured along the skeleton normal at each annotated gridline. Calibrations come from the
1-D study without the image's own series; the learned model is the fold that never saw that series.
MAE in um (band-level bootstrap CI, `summary_2d.csv`); errors are over detected gridlines only:

| px/mm | detected | FWHM | rectangle | photometric | fusion | learned |
|---|---|---|---|---|---|---|
| 25 | 0.98 | 42.5 (35-51) | 37.4 (33-44) | 38.9 (34-44) | 39.5 (34-46) | **28.9 (27-33)** |
| 12.5 | 0.93 | 52.1 (45-59) | 38.2 (32-45) | 41.3 (35-48) | 39.6 (34-45) | 37.4 (34-42) |
| 8 | 0.87 | 68.3 (63-73) | 40.7 (35-48) | 43.6 (36-52) | 39.9 (32-48) | 42.2 (38-48) |
| 5 | 0.79 | 77.0 (73-81) | 45.5 (42-53) | 51.4 (42-59) | 49.2 (40-57) | 49.2 (44-58) |
| 3.6 | 0.70 | 79.9 (71-85) | 46.9 (43-52) | 50.3 (42-57) | 48.2 (40-56) | 49.0 (45-59) |

* The 1-D conclusion holds end to end: FWHM error grows to ~80 um while the deficit-based and learned
  estimators stay at 40-50 um (1.6-1.7x better at 3.6-8 px/mm) and are statistically tied with each other
  below 12.5 px/mm.
* Detection, not width, becomes the bottleneck at the coarse end (70 % of gridlines found at 3.6 px/mm);
  the MAE is conditional on detection, which favours wider cracks, so 2-D values are lower than 1-D.
* Transferring the 1-D calibration to 2-D profiles adds a positive bias to the photometric estimate
  (+11 to +33 um; FWHM +13 to +66 um); the rectangle transform and the learned model stay within +-10 um.
* Band-level healing ratio (stage 6 vs 1, three specimens, reference 0.25 / 0.17 / 0.38): at 8 px/mm the
  photometric width gives 0.09 / 0.20 / 0.40 and FWHM -0.06 / 0.05 / 0.15; anecdotal (n = 3) but
  consistent with section 5.

## 8. Whole-element decisions (`scripts/element_decisions_krkCMd.py`)

Codes limit the widest crack on a member. Element = one scanned image at one stage (216 elements,
~88 gridlines each, maximum reference width 0-814 um). Rules at 0.3 / 0.4 / 0.5 mm, LOSO
(`element_decisions.csv`):

* **Max-statistic conformal** (score on the element maximum itself, calibrated on other series'
  elements) decides far more elements than location-based rules. With the learned estimator it
  decides 72 % of elements at 25 px/mm, 56 % at 8 and 18 % at 3.6 px/mm (0.3 mm limit); Bonferroni over
  locations decides <= 4 %, and the 90 % interval at the worst-looking spot 50 % / 18 % / 6 %. False
  acceptance of the max-statistic rule is <= 3.5 % over all estimators, resolutions and limits.
* **But element-level exchangeability is fragile**: coverage of the element interval is 0.80-0.87 pooled,
  because one series (CMd_0.28_2mths, many healed and sparsely sampled images) covers only 0.61-0.73
  while the other three cover 0.96-1.00. Element-level certification therefore needs calibration
  elements from the structure being inspected.
* The worst-spot rule (what a single per-photo interval does) is safe with the deficit-based and learned
  estimators (false acceptance <= 4.2 %) but not with FWHM, which under-reads the widest crack at coarse
  resolution (bias -139 um at 3.6 px/mm) and wrongly passes 44 % of elements at the 0.5 mm limit.
* Error of the estimated element maximum (MAE): learned 32 / 49 / 73 um at 25 / 8 / 3.6 px/mm; fusion
  72 / 70 / 78; FWHM 119 / 192 / 184.

## 9. Gauge-anchored photo monitoring (`scripts/gauge_anchoring_krkCMd.py`)

The error of a photo-based width at a given location is persistent: its correlation with the same
gridline's error at the other stages is 0.66-0.83 at every resolution, for every estimator
(`error_persistence_r` in `gauge_anchoring.csv`). One gauge
reading per location at installation (day 0), then photos only: w_s* = w_s - beta (w_1 - g_1), beta from
other series (`gauge_anchoring.csv`, comparator-card gauge in 0.05 mm steps):

| px/mm | photometric | anchored | fusion | anchored | learned | anchored |
|---|---|---|---|---|---|---|
| 252 | 45.1 | 37.1 | 36.3 | 31.9 | 15.0 | 14.6 |
| 25 | 46.3 | 38.2 | 44.0 | 37.1 | 22.1 | 20.8 |
| 8 | 52.5 | 42.8 | 45.2 | 39.0 | 36.6 | 32.7 |
| 3.6 | 64.1 | 48.6 | 57.4 | 48.6 | 52.4 | 45.2 |

* 12-24 % lower error for the physics-based estimators (largest at the coarse end), 3-14 % for the learned
  one; a card with 0.05 mm steps loses < 1 um against an exact reading, adding 15 um gauge noise < 2 um.
* Conformal coverage is kept (0.89-0.91), but intervals change by only -8 to +2 % and decided fractions
  rise by 0-3 points: anchoring tightens typical errors more than the tails.

## 10. Decision-driven capture planning (`scripts/capture_planner.py`)

Decidability map D(r, w) = P(certified decision is conclusive | resolution, true width), from the
LOSO intervals (`decidability.json`, `fig_limits_decidability_*`). Around a limit there is a dead band
of widths that no photo at that resolution decides with >= 80 % probability. At 0.3 mm, learned
estimator: 262-362 um at 252 px/mm, 238-388 at 25, 212-512 at 8, 112-588 at 3.6 px/mm (upper edges
>= 588 um are censored by the data); photometric: 138-588 um already at 25 px/mm. The band is
asymmetric: at low resolution wide cracks become hard to certify as exceeding, because the intervals
are multiplicative. The planner inverts the map: e.g. an inconclusive 0.45 mm estimate at 3.6 px/mm needs
>= 15.3 px/mm, i.e. re-shoot 4.2x closer; a 0.235 mm estimate against a 0.2 mm limit cannot be decided by
any photo (photometric) and is sent to a gauge. The app adds this advice to inconclusive results.
The planner uses the estimate as a proxy for the true width, so the advice is approximate.

## What this does NOT show

* Lower resolutions are simulated from scans, not real phone photographs. The printed-chart experiment
  (`scripts/width_test_chart.py`) is still to be run.
* One dataset, one laboratory, four specimen series; the calibration reproduces one operator's edge
  definition. Cracks with light-coloured fillings, wet or painted surfaces break the dark-interior
  contrast assumption.
* The deployed calibration (`docs/results/limits/photometric_calibration.json`, used by the app) is fitted
  on krkCMd. On other concretes its offsets and its coverage must be re-checked with gauge readings.
