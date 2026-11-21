# Eye-Tracking Foundations & Signal Analysis: Research Methods and Evidence

This document separates the recorded execution of Notebook 01 on original GazeBase recordings from the still-unexecuted scientific stages 02–07. The companion [README](README.md) provides the entry point and notebook order.

## 1. Research Objective and Scope

Study how recorded gaze coordinates can be inspected, quality-assured and transformed into transparently derived event intervals, features, descriptive analyses and visualisations. This is a bounded methodological and educational study, not a model estimating cognition, reading comprehension or clinical outcomes.

## 2. Data Sources, Cohort and Target Definitions

The intended primary source is [GazeBase](https://doi.org/10.6084/m9.figshare.12912257), described in [Griffith et al., Scientific Data (2021)](https://doi.org/10.1038/s41597-021-00959-y). Source columns and record identity are visible in the committed Notebook 01 execution. The original-data run is not independently reproduced in this review. The initial script selects up to ten complete participants with complete `FXS`, `HSS`, `RAN` and `TEX` records in Round 1, Session 1. This is deterministic bounded source inspection, not a representative sample. The source reader selects up to ten complete participants in sorted ID order and reports how many are actually present. The committed notebook records 10 participants, 40 task files and 2,778,008 source rows. The exact CSV contents and complete stage-owned outputs are not committed, so those figures have not been independently recomputed.

Time is `n` in milliseconds; `x/y` are degrees of visual angle; `val` marks invalid/missing samples; `dP` is an uncalibrated area measure. Where available, vendor sample labels `1`, `2` and `-1` represent parser fixation, saccade and blink categories. Absent vendor annotations, especially in some viewing tasks, are not converted to false negatives. The event detectors produce derived *algorithm labels*, not clinical or behavioural targets.

## 3. Data Preparation and Representations

Notebook 01 preserves original record/sample identity, timestamps and source validity, while reporting observed quality. Notebook 02 streams the current large Notebook 01 sample CSV by contiguous full record and applies a centred 5-sample rolling median only *within contiguous valid segments*, with a 5-ms maximum adjacency gap; these are starting parameters for education, not optimised or validated. Original `x/y` values remain available, and missing gaze is neither bridged nor assigned invented eye movements. Velocity is computed from consecutive smoothed valid coordinates and timestamps in degrees of visual angle per second. First samples of each segment have undefined adjacent velocity. The median uses centred neighbouring samples and therefore represents offline rather than causal real-time processing. The streaming implementation uses the unchanged record-level calculation, checks row counts and record identities against Notebook 01's manifest and quality table, and writes a complete replacement file only after those checks succeed.

## 4. Experimental Design and Evaluation

Notebook 03 computes I-VT and fixation-only I-DT assignments on exactly the prepared sample set. I-VT uses an initial threshold of 30 dva/s, minimum fixation duration 80 ms and minimum saccade duration 6 ms. I-DT uses a 1-dva sum-of-axis-range dispersion threshold and 80-ms fixation minimum. Non-fixation I-DT samples remain unknown rather than being presented as saccades or blinks. Vendor comparison uses only records/samples with compatible available reference labels, reporting denominator and coverage. It is conditional agreement, not an independently adjudicated event-level accuracy estimate.

Notebook 06 evaluates a fixed, exploratory 20/30/40-dva/s I-VT threshold grid on Notebook 02's stored motion representation. The change in label fractions and reference agreement is sensitivity analysis, not outcome-driven parameter selection. There is no fitted model, training split, held-out Test set or generalisation claim in the first S1 programme.

## 5. Methods, Models and Configurations

The maintained modules are `gaze_source.py`, `gaze_quality.py`, `gaze_preprocessing.py`, `eye_events.py`, `gaze_features.py` and `gaze_statistics.py` under `code/src/`, with `stage_outputs.py` for explicit reuse of current outputs. Notebook 04 computes per-record fixation counts/duration, saccade velocity, an approximate fixation-centroid scanpath length and pupil summaries where valid measurements exist. It does not map gaze to word regions without verified stimulus geometry. Notebook 05 first aggregates repeated records at participant-by-task level, then calculates exploratory participant-resampled percentile intervals for available features using 500 draws and seed 17; this cannot compensate for a small independent cohort.

All notebooks use numbered Markdown/code sections, non-destructive Colab checkout and source-code imports; external Human Movement infrastructure, when copied, must retain byte-exact code with provenance rather than being silently rewritten. Their scientific event-analysis modules are independent of Human Movement's EMG/IMU methods.

## 6. Results and Comparative Analysis

**Notebook 01 executed:** The committed notebook output reports 10 participants and 40 recordings with a total of 2,778,008 sample rows. Per-record valid gaze fractions range from approximately 67.35% to 100% (the minimum and maximum printed in the notebook). These observed source-audit values are descriptive and not reproduced independently here. No actual derived event labels, velocity summaries, uncertainty intervals or detector-reference comparisons are available from executed Notebooks 02–07; the synthetic tests assess software contracts only.

## 7. Robustness, Uncertainty and Additional Analyses

Planned real-data checks include task-wise missingness, timing gaps, optional pupil coverage, conditional EyeLink agreement, velocity-threshold sensitivity and participant-level variability. Pupil area is uncalibrated; device labels are reference annotations; viewing conditions and stimulus differences can confound task comparisons. Bootstrap uncertainty from a very small participant subset remains exploratory, not population-level certainty. Any substantive threshold or methodology changes after real-data inspection must be recorded and evaluated independently of favourable scores.

## 8. Current Outputs and Reproducibility

The seven notebooks write stage-prefixed current CSV files and plots to `output/`. Each downstream notebook directly consumes named producer files; to continue in a new Colab runtime, an explicit Notebook ZIP created by the unchanged `download_outputs_from_colab` helper can restore those current files. No historical output search, approval token, hidden retraining, or automatic source re-download occurs. Notebook 07 reads and reconciles summaries but does not execute earlier stages.

Static syntax/notebook validity checks and a complete synthetic pipeline exercise are part of the development verification, not independent replication. Notebook 01's Colab execution and saved output cells are visible in its committed notebook, but the generated CSV tables are not published, and no independent reproduction has occurred here. Stage 02 remains to be executed in Colab on the full 10-participant source, and subsequent stages are pending. The original Notebook 01 CSV is reused for Notebook 02; bounded ZIP extraction and a chunked reader limit peak RAM without reducing the dataset or changing preprocessing settings. The exact release, environment versions, accessible files and any necessary licence restrictions should be recorded at first genuine execution.

## 9. Discussion, Limitations and Future Work

This implementation has completed source inspection for Notebook 01 but is not a scientifically completed S1 study. Main limitations are the uninspected real source subset, provisional smoothing and detector thresholds, possible event taxonomy mismatch, limited sample representativeness and lack of expert adjudication. The next validation task is to review Notebook 01's full quality outputs, then execute Notebook 02 on its current saved sample CSV and inspect timebase boundaries and velocity before proceeding to Notebook 03. All subsequent stages should use the named current files for consistent scientific provenance. Deep learning, EEG fusion and cognitive inference remain independent later-study responsibilities.

### References

- Griffith et al. (2021). GazeBase, a large-scale, multi-stimulus, longitudinal eye movement dataset. Scientific Data. https://doi.org/10.1038/s41597-021-00959-y
- GazeBase Figshare data repository: https://doi.org/10.6084/m9.figshare.12912257
