# What a skeptical reviewer will say (and what would answer it)

1. **"Resolution was simulated from scanner images, not real cameras."**
   Run the printed-chart study (`scripts/width_test_chart.py`) with 2-3 phones at 5 distances; verify the printed line
   widths with a microscope or comparator card. Report measured-vs-nominal against px/mm next to the simulation.
2. **"The reference is one operator's manual measurement."**
   Quantify reference uncertainty (second operator on a 200-profile subset) or re-measure with a calibrated microscope;
   state inter-operator spread as a floor. Without it, conclusions below ~0.1 mm are weak.
3. **"The estimator is not better than the dataset's own deep model (MAE 15 vs 56 um)."**
   Correct, and the paper says so. Either reposition fully as a measurement-science/characterisation paper (current text),
   or add a learned estimator trained on the dataset and evaluated under the same degradation protocol.
4. **"Only 4 of 36 images in 2-D; no confidence intervals; one lab, four series."**
   Fetch all 36 images (~4 GB in bands), add bootstrap CIs by gridline and by specimen, and add a second dataset if one with
   physical widths is found (check the 2023 narrow-crack Scientific Data set's metadata).
5. **"Field relevance: rough surfaces, shadows, clutter, perspective, motion."**
   Add a sensitivity study (blur sigma, noise, illumination gradient, tilt) and at least a small set of real photographs of
   concrete with a crack comparator card in frame.
6. **"The +55 um offset is a definition mismatch; single linear calibration is not general."**
   Show calibration transfer across stages and series; report per-stage offsets; propose a standard definition for width.
