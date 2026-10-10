# Real-data resolution study (krkCMd)

**Data.** krkCMd, Zenodo [10.5281/zenodo.14568863](https://doi.org/10.5281/zenodo.14568863)
(CC BY 4.0; Scientific Data 2025, [10.1038/s41597-025-04485-z](https://doi.org/10.1038/s41597-025-04485-z)).
19,098 brightness profiles across real concrete cracks (scanner pixel pitch 3.97 µm, ~252 px/mm),
with a manual reference width from one trained operator. 14,198 profiles with width > 0 were used
(the rest are fully healed cracks). 4 specimen series (w/c 0.23 and 0.28, 2 and 20 months).

**Code.** `scripts/evaluate_resolution_krkCMd.py`; the width routine is the same
`fwhm_from_profile` used by the image pipeline (`scripts/crack_width_measure.py`).

**Protocol.** Each real profile is re-imaged at 2.5 – 252 px/mm: optional Gaussian PSF (σ = 0.7 px
in the "phone-like" scenario), integration over square pixels with random sub-pixel phase, Gaussian
sensor noise (2 / 4 grey levels), 8-bit quantisation. Fixed ±0.8 mm measurement window (independent of
the true width, so no label leakage). Three measurements per simulated profile: our sub-pixel FWHM
(0.8 px pre-blur with quadrature correction), a threshold-and-count baseline, and a steepest-gradient
edge baseline.

## Results (phone-like imaging, MAE in µm, 95 % bootstrap CI over profiles in `mae_bootstrap_ci.csv`)

| px/mm | sub-pixel FWHM (ours) | threshold count | gradient edges |
|---|---|---|---|
| 252 | 56 | 55 | 69 |
| 50 | 57 | 59 | 71 |
| 25 | 61 | 65 | 82 |
| 12.5 | 86 | 91 | 135 |
| 8 | 141 | 142 | 238 |
| 5 | 271 | 279 | 454 |
| 3.6 | 423 | 482 | 707 |
| 2.5 | 701 | 887 | 1153 |

Median relative error by true width (phone-like): see `fig_error_heatmap.png`.
For example a 200–400 µm crack stays within ~14 % down to 12.5 px/mm and ~25 % at 8 px/mm; 100–200 µm
needs ≥ 25 px/mm for ~24 %; 50–100 µm is ~48 % even at 252 px/mm; < 50 µm is > 130 % at every resolution.

## What this supports
1. **Error is flat above ~25 px/mm and rises steeply below ~12 px/mm** (roughly MAE ∝ 1/resolution
   at the coarse end). A typical phone photo of a core (~3–4 px/mm) gives MAE ≈ 0.4 mm on cracks
   whose true widths are mostly 0.1–0.4 mm.
2. **Hairline cracks (≈ 0.04 mm) are not measurable by profile-width methods even at 250 px/mm** with
   this reference (median relative error 132 %; detection 99.9 %). The result is dominated by a
   systematic offset, see below, plus the limits of a single-operator reference at this scale.
3. **Sub-pixel interpolation helps only modestly and only at the coarse end:** −21 % MAE vs threshold
   counting at 2.5 px/mm, −12 % at 3.6, ≈ 0 at ≥ 8 px/mm.

## What this does NOT support (be explicit in any paper)
* **Our method is not more accurate than existing ones.** The dataset's own deep-learning metasensor
  reaches MAE 14.9 µm (median relative error 4.6 %) and its threshold edge detector 38.0 µm, both at
  native resolution; ours is ~56 µm uncalibrated. We make no state-of-the-art claim.
* **There is a systematic +55 µm bias at every resolution ≥ 25 px/mm** for all three profile-based
  methods. The reference marks the steepest edges and excludes bevelled crack edges, which our
  half-depth FWHM does not. A leave-one-series-out linear calibration reduces MAE to 37 µm at native
  and 47 µm at 25 px/mm, but only to ~98 µm at 8 px/mm and ~110 µm at 3.6 px/mm
  (`calibration_leave_one_series_out.csv`). So calibration removes the offset, not the resolution limit.
* **Per-series spread is large at 8 px/mm** (MAE 110–154 µm, within-50 % 39–64 %); results are not uniform
  across specimens.
* It tests only the profile-width step on flat scanner images simulated at lower resolution:
  not segmentation, rough core surfaces, lighting or perspective, and not real phone photographs.
  The reference is one operator's manual reading, not a physical gauge.
* The dataset is 4 specimen series from one lab; generalisation to other concretes is untested.

## Reproduce
```bash
curl -L -o data/krkCMd_raw/krkCMd_table.csv https://zenodo.org/api/records/14568863/files/krkCMd_table.csv/content
python scripts/evaluate_resolution_krkCMd.py --csv data/krkCMd_raw/krkCMd_table.csv --out outputs/krkCMd
```
(about 20 min on a laptop CPU; `--max-profiles 400` for a quick check). Seeds are fixed.
