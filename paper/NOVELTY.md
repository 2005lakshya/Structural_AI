# Novelty statement and positioning

Working title: **"How much can a photograph tell about a crack's width? Information limits,
contrast-limited estimation and certified compliance decisions for image-based crack measurement"**

Suggested venues (measurement-science framing): *Measurement* (Elsevier), *Measurement Science and
Technology* (IOP), *Automation in Construction*, *Computer-Aided Civil and Infrastructure Engineering*,
*Structural Health Monitoring*.

## Draft abstract

Photographs are increasingly used to measure crack widths against code limits, yet the resolution at
which such measurements stop carrying information has only been described by rules of thumb. We derive
the Fisher information of crack-width estimation for a blurred, pixel-integrated, noisy image of a dark
slot and show that below the blur scale width and contrast are not separately identifiable: with unknown
contrast the Cramér-Rao bound grows with the cube of blur over width, and the smallest measurable width
scales only with the cube root of the signal-to-noise ratio. Only the brightness deficit remains
measurable, so width below the blur scale requires contrast knowledge, and its error floor is the
contrast spread. We test this on 14,198 real crack profiles (krkCMd) re-imaged at 2.5-252 px/mm under
five imaging conditions, with leave-one-series-out validation. Edge-based estimators fall to no-skill
below ~8 px/mm, while the photometric (deficit-based) estimator keeps a mean absolute error of 45-70 um
down to 2.5 px/mm, and all
estimators collapse onto a single curve of error versus width over blur. A resolution-conditioned network
matches the dataset's in-sample deep model at native resolution (15.6 um) on held-out specimens, but its
advantage over the physics-based estimate shrinks to 3 um exactly where theory predicts only the deficit
is identifiable. Real cracks carry a ~100 um dark rim outside the operator's edges that self-healing
shrinks. Split-conformal intervals calibrated across specimen series give 90 % coverage and a three-way
compliance decision that keeps false acceptance at the 0.3 mm limit below 2.4 %, against 24-100 % for
point-estimate checks at phone-like resolutions. Deficit-based widths also recover the self-healing ratio
(r = 0.92 at 8 px/mm) where edge-based widths lose it (r ~ 0).

## What is already published (do not claim these)

| Idea | Prior work | Verified reference |
|---|---|---|
| Sub-pixel width from the area of the dark dip ("equal-area rectangle") | Rectangle transform, Benz & Rodehorst, MVA 2021; Equal Area method, Li & Su, CACAIE 2025 | `benz2021rectangle`, `li2025equalarea` |
| Partial-area sub-pixel crack edges | Zheng et al., Buildings 2024 (tested at 110-170 px/mm only) | `zheng2024partialarea` |
| Mixed-pixel error model of threshold widths vs ground resolution | Cho, Yoon & Jung, Sensors 2018 (thresholding only, no blur, crack intensity known) | `cho2018ground` |
| Zernike-moment width of thin cracks | Ni, Zhang & Chen, CACAIE 2019 | `ni2019zernike` |
| Unbiased line width under Gaussian smoothing | Steger, TPAMI 1998 | `steger1998unbiased` |
| Cramér-Rao bounds for the width of blurred tubular structures | Wörz & Rohr, JMIV 2008 (3-D vessels, medical imaging) | `worz2008crb` |
| CNN crack-width "metasensor" on krkCMd profiles | Jakubowski & Tomczak, CBM 2024 (native resolution only) | `jakubowski2024metasensor` |
| Pixel-level uncertainty of thin cracks near the GSD limit (UAV) | Lee, Lee & Kim, Sensors 2026 (segmentation only, no width estimation) | `lee2026uav` |
| Split-conformal intervals for crack width | Only for tabular member-level crack-width *prediction* (design formulas), not image measurement | — |
| Super-resolution before crack segmentation | Several (2022-2026) | — |

Searches that found nothing for the image-based crack problem (October 2026): Fisher information /
Cramér-Rao bound for crack width; conformal prediction for image-based crack width; ISO 14253-style
decision rules for crack-width code compliance from photographs; resolution dependence of self-healing
(healing-ratio) measurement.

## Contributions of this work

1. **Information limit of photographic crack-width measurement** (`scripts/width_information_limit.py`).
   Closed-form forward model (slot x Gaussian PSF x pixel integration x noise), exact Fisher
   information and phase-averaged CRLB. Main result: for w << s_eff the image depends on width and
   contrast only through the deficit area; with unknown contrast the relative bound grows as
   (s_eff/w)^3 and the smallest measurable width scales as s_eff x SNR^(-1/3) (better sensors buy little).
   A contrast prior turns the divergence into a floor equal to the contrast spread. Checked by Monte
   Carlo: maximum likelihood attains the bound wherever it is finite.
   *Relation to Wörz & Rohr (2008):* same estimation-theory tools, new object (2-D dark slot with pixel
   integration), and the new consequence that is specific to cracks: the width/contrast
   non-identifiability and the cube law, which explain both the empirical "2-pixel rule" and why
   equal-area estimators work below it.
2. **Real-data confirmation of the regimes** on 14,198 krkCMd profiles re-imaged at 2.5-252 px/mm under
   five imaging conditions, leave-one-series-out, cluster-bootstrap CIs. Every estimator collapses onto
   one curve of relative error vs w/s_eff; edge-based estimators fall to no-skill below ~8 px/mm
   (w/s_eff ~ 1), deficit-based ones do not. First across-resolution comparison of the published
   equal-area estimator with edge-based ones on real cracks.
3. **Crack contrast characterised on real concrete**: photometric width P = 0.92 w + 102 um, i.e. a ~100 um
   dark rim (bevelled edges) outside the operator's edges; the rim shrinks from 122 to 93 um during
   28 days of self-healing while the slope stays ~0.92. This explains the earlier +55 um FWHM bias and
   why FWHM cannot follow healing.
4. **Certified compliance decisions.** First distribution-free (split-conformal / CQR) intervals for
   image-based crack width, calibrated per resolution, validated under specimen-series shift, and turned
   into a three-way rule (conforms / does not conform / inconclusive) in the spirit of ISO 14253-1 and
   JCGM 106. Shows that naive point-estimate checks against 0.3 mm accept 24-100 % of non-compliant
   cracks at phone-like resolutions.
5. **Self-healing monitoring at phone resolution.** Per-image healing ratio is recovered with r = 0.92 at
   8 px/mm and 0.84 at 3.6 px/mm by the photometric width, while edge-based widths lose it (r ~ 0 at
   <= 50 px/mm).
6. **A resolution-conditioned learned estimator** benchmarked against the bound and the physics-based
   estimators under the same protocol. Held out by specimen series it reaches 15.6 um at native resolution,
   but its lead over the physics-based fusion shrinks from 21 um to < 3 um below the blur scale, exactly where
   theory says only the deficit is identifiable. Its conformal intervals hold coverage only with
   shift-aware (nested, cross-series) calibration: same-series calibration undercovers (0.79-0.85) and lets
   7-10 % of non-compliant cracks pass at 0.3 mm; nested calibration restores 0.89-0.94 and <= 2.2 %.
7. **Whole-element certification.** A conformal test on the element maximum (the quantity codes limit)
   decides up to 72 % of inspected elements where location-wise Bonferroni decides <= 4 %, with <= 3.5 %
   false acceptance; it also shows that element-level guarantees are fragile across specimens (one of four
   series covers 0.61-0.73) and that a per-photo worst-spot interval is unsafe with edge-based widths
   (44 % false acceptance at 0.5 mm).
8. **Gauge-anchored monitoring.** Photo errors at a location persist over time (r = 0.66-0.83), so one
   comparator-card reading at installation lowers later photo errors by 12-24 % (physics-based) while
   keeping conformal coverage.
9. **Decision-driven capture planning.** A decidability map P(conclusive | resolution, width) with an
   asymmetric dead band around the limit; inverted, it tells the inspector how much closer to re-shoot
   or that only a gauge can decide.
10. **Open implementation** in the StructuralAI app: the image endpoints return the photometric width,
   its 90 % interval, the three-way decision and, when inconclusive, the re-shoot advice.

## Honest limitations (state them in the paper)

* Lower resolutions are simulated from 252 px/mm scanner images (area integration, Gaussian PSF, noise,
  shading); no real phone photographs yet (`scripts/width_test_chart.py` prepares that experiment).
* One dataset, one laboratory, four specimen series; the calibration maps to one operator's edge
  definition (inter-operator RMS 16.6 um per the dataset paper, i.e. an MAE floor of ~13 um).
* The contrast model assumes dark crack interiors in diffuse light. Wet, filled, painted or
  light-coloured crack deposits violate it; the conformal coverage then has to be re-checked.
* Conformal coverage is exact only under exchangeability; across series it ranges 0.85-0.95 (photometric)
  around a pooled 0.90.
* The 2-D end-to-end check uses 6 image bands (3 specimens x 2 stages).
