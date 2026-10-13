# What a skeptical reviewer will say (and where it stands)

1. **"Resolution was simulated from scanner images, not real cameras."** STILL OPEN.
   Run the printed-chart study (`scripts/width_test_chart.py`) with 2-3 phones at 5 distances; verify the printed line
   widths with a microscope or comparator card. Also photograph real cracks with a comparator card in frame.
   Partly mitigated: five imaging conditions (blur, noise, shading) give the same ordering (`docs/RESULTS_limits.md`).
2. **"The reference is one operator's manual measurement."** PARTLY ANSWERED.
   The dataset paper reports a 5-operator RMS of 16.6 um (MAE floor ~13 um); this is now stated as the floor. The
   calibration still reproduces one operator's edge definition.
3. **"The estimator is not better than the dataset's own deep model."** ANSWERED DIFFERENTLY.
   The paper is now about information limits and decisions. The dataset DLM was trained on this data (no held-out
   number). A resolution-conditioned learned estimator is evaluated leave-one-series-out under the same degradations
   (section 6 of `docs/RESULTS_limits.md`).
4. **"Only 4 of 36 images in 2-D; no confidence intervals; one lab, four series."** PARTLY ANSWERED.
   All 1-D results use cluster-bootstrap CIs over the 36 images and leave-one-series-out calibration. 2-D now uses 6
   held-out bands (3 specimens x 2 stages) with band-level bootstrap CIs. Still one laboratory.
5. **"Field relevance: rough surfaces, shadows, clutter, perspective, motion."** STILL OPEN (simulated blur, noise,
   shading only).
6. **"The +55 um offset is a definition mismatch."** ANSWERED.
   It is a ~100 um dark bevelled rim outside the operator's edges (P = 0.92 w + 102 um), which self-healing shrinks
   (122 -> 93 um). Calibration absorbs it; its change with healing is quantified.
7. **(new) "Equal-area width estimation is already published (Benz & Rodehorst 2021; Li & Su 2025)."**
   Correct and cited; the rectangle transform is re-implemented and is the strongest classical baseline down to
   8 px/mm. Our claims are the information-limit theory that explains why such estimators work and where they fail,
   their first across-resolution evaluation on real cracks, the contrast characterisation, certified decisions and
   healing monitoring, not the equal-area idea itself.
8. **(new) "Conformal coverage assumes exchangeability."** Tested, not assumed: leave-one-series-out coverage per series
   is reported (0.85-0.95 for the photometric intervals at a nominal 0.90).
