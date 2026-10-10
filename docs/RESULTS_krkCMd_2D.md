# End-to-end (2-D) real-image study on krkCMd

Code: `scripts/evaluate_end_to_end_krkCMd.py` (bands fetched with `scripts/fetch_krkCMd_band.py`).
Tables: `docs/results/2d/`.

## What was run
Real scanner images (252 px/mm) were re-imaged at 3.6 – 50 px/mm (area-averaged, optional Gaussian
PSF, sensor noise, 8-bit), then segmented, skeletonised and measured at every annotated gridline
with the same sub-pixel FWHM routine. A gridline counts as a *miss* when no skeleton point lies within
0.3 mm of the annotated crack centre. Two segmentations were compared: the training-free multi-scale
ridge filter and plain background-flattened Otsu thresholding.

* **Development split** (hyper-parameters tuned here only): series CMd_0.23_2mths, Image 1, stages 1 and 4 (161 gridlines).
* **Held-out split**: series CMd_0.28_20mths, Image 1, stages 1 and 4 (254 gridlines).
  Different specimen series from the tuning data; hyper-parameters frozen before it was touched.

Image bands were verified against the dataset's stored profiles (correlation 1.000 for the same gridline and stage).

## Results (phone-like imaging, mean absolute error in µm; hit rate in brackets)

| px/mm | ridge, held-out | Otsu, held-out | ridge, dev |
|---|---|---|---|
| 50 | 97 (1.00) | 118 (1.00) | 117 (1.00) |
| 25 | 96 (1.00) | 103 (1.00) | 116 (1.00) |
| 12.5 | 115 (1.00) | 114 (1.00) | 129 (0.99) |
| 8 | 174 (1.00) | 186 (1.00) | 166 (0.99) |
| 5 | 273 (0.98) | 308 (1.00) | 255 (0.97) |
| 3.6 | 405 (0.88) | 475 (1.00) | 381 (0.93) |

## What this supports
* **The resolution dependence seen in the 1-D study holds end to end.** Error is roughly flat
  above ~12 px/mm and grows quickly below ~8 px/mm; at 3.6 px/mm the mean error (≈ 0.4 mm) is
  about 3.6 times the no-skill baseline (112 µm).
* **On clean laboratory surfaces segmentation is not the bottleneck.** Hit rates are ≈ 100 % down to 8 px/mm
  for both segmenters; the error is the width estimate itself.

## What it does not support
* **The ridge filter is not better than Otsu here.** The two are within noise at ≥ 8 px/mm; Otsu has the
  higher hit rate at 3.6 px/mm (but a larger error). The ridge filter was designed for cluttered
  full-scene photos, which this dataset does not contain, so its intended advantage is untested.
* **Errors are almost entirely a positive bias** (bias ≈ MAE), larger than the 1-D average (~55 µm).
  These two images (stages 1 and 4 of a single image per series) are wider-crack and more biased
  than the dataset as a whole; per-stage 1-D bias ranges from +72 µm (stage 1) to +45 µm (stage 6).
  This is a definitional offset between half-depth FWHM and the operator's steepest-edge marking, not noise.
* **Small sample:** 415 gridlines from 2 images × 2 stages × 2 series. Results are indicative; no
  confidence intervals are claimed for the 2-D study.
* Only lab-scanner re-imaging; no real phone photographs, no rough core surfaces, no lighting variation.
