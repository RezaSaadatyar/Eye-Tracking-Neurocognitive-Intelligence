# Eye-Tracking Foundations & Signal Analysis | GazeBase v2.0: Research Methods and Evidence

**Study:** Eye-Tracking Foundations & Signal Analysis (S1), Eye-Tracking & Neurocognitive Intelligence. **Maintained implementation:** [Notebooks 01–07](code/notebooks/). This document provides the scientific definitions, preprocessing and detector configurations, original-data observations, participant-level descriptive uncertainty, sensitivity analysis, output provenance and interpretive limits. For a concise overview and the complete seven-notebook figure, see the [study README](README.md), [study-wide workflow](images/study_overview.svg) and [execution guidance](README.md#4-running-and-reviewing-the-study).

**Evidence status:** All seven notebooks have recorded executions on a selected **GazeBase v2.0 Round 1 / Session 1** subset: 10 participants, 40 task recordings and 2,778,008 source sample rows. Notebook 07's saved summary consistency checks completed without a recorded error. This is an **offline, within-source exploratory analysis**; it is not an independent reproduction, a formally validated eye-event detector or a predictive experiment.

## 1. Research Objective and Scope

The study investigates a bounded methodological question: **Can a documented sequence of source audit, gap-aware preprocessing, provisional eye-event detection, record-level feature construction and participant-aware descriptive analysis produce source-aligned, interpretable evidence from selected GazeBase recordings?** It also asks how fixed I-VT velocity thresholds affect assigned event classes and eligibility for comparison with EyeLink parser annotations.

The seven notebooks address distinct, linked responsibilities: source identity and validity (01), motion representation and extreme-value inspection (02), I-VT/I-DT events and vendor comparisons (03), detector-derived gaze features (04), exploratory participant-level uncertainty (05), sensitivity and eligible-reference denominators (06), and cross-notebook consistency (07).

**Units matter:** a gaze sample is not an independent participant; a detector-generated interval is not an independently validated eye movement; the same person contributes all four task recordings; and the sum of event counts reported by two detectors is not a count of distinct, non-overlapping physiological events across methods.

There is **no supervised cognitive target, learned model, Training/Validation/Test split, held-out person-independent Test or externally validated classifier** in this first study. Reading comprehension, attention diagnosis, EEG fusion, prospective sensing and clinical applications are outside the evaluated scope.

## 2. Data Sources, Cohort and Target Definitions

### 2.1 Original source and acquisition context

The source is [GazeBase](https://doi.org/10.6084/m9.figshare.12912257), described by [Griffith, Lohr, Abdulin and Komogortsev (2021)](https://doi.org/10.1038/s41597-021-00959-y). The original publication states that GazeBase data are distributed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). This repository contains a **selected 40-CSV subset** of the original archive; the original data are not re-licensed under the repository software's MIT licence. Retain source credit, the licence link and an indication of subset selection when redistributing. The implemented source reader uses the `GazeBase_v2_0` directory layout, selecting original CSV files by round, session, task and participant. This analysis uses **Round 1, Session 1 only**. The published GazeBase data collection is larger; the selection here must not be described as the full dataset or a random sample.

Relevant source fields are:

| Field | Meaning and treatment |
| --- | --- |
| `n` | Original time coordinate in **milliseconds**; actual adjacent differences are checked rather than treating the nominal recording frequency as a universal guarantee |
| `x`, `y` | Recorded gaze positions in **degrees of visual angle (dva)** |
| `val` | Source validity flag; in the inspected records **`val = 0` corresponds to nominally valid** gaze |
| `dP` | Available pupil-area measurement, not a calibrated physiological pupil-diameter endpoint |
| `lab` | EyeLink parser labels where available: recognised fixation (`1`), saccade (`2`) and blink (`-1`) codes; an instrument-derived reference, not independent expert ground truth |

Source coordinates, task identity, timestamp and sample index are retained for provenance. Missing or unrecognised vendor labels are not interpreted as evidence that a true event did not occur.

### 2.2 Implemented subset and independence units

Notebook 01 deterministically selected up to **ten complete participants in source-ID order**, requiring one available FXS, HSS, RAN and TEX recording in the specified round/session. The executed selection contains source subjects **1001–1010** (internally identified as `001`–`010`), yielding **40 records and 2,778,008 source sample rows**.

| Scope dimension | Observed value |
| --- | ---: |
| Source participants | 10 |
| Task protocols per person | 4: FXS, HSS, RAN, TEX |
| Distinct selected recordings | 40 |
| Raw sample rows | 2,778,008 |
| Task records contributing to Notebook 05 summaries | 10 per task |
| Independent persons used for uncertainty | 10 |

This subset is neither representative of all GazeBase rounds/sessions nor independent across tasks: the **same participants** appear in FXS, HSS, RAN and TEX. Downstream sample rows, event intervals and task records must not be treated as 2,778,008 or 21,728 independent participants.

### 2.3 Analytical quantities, not cognitive labels

The primary observations are source validity, sampling continuity, derived velocity, algorithm-assigned fixation/saccade intervals, event-dependent feature summaries and conditional parser-label agreement. Notebook 05 uses four prespecified per-record metrics: `valid_fraction`, `mean_fixation_ms`, `scanpath_length_dva` and `mean_saccade_peak_dva_s`.

Neither reading comprehension nor subject-level neurocognitive status is observed as an evaluated target. The HSS/RAN/TEX task names indicate the source recording protocol; they do not imply that a reading-related feature has been validated as a cognitive endpoint.

## 3. Data Preparation and Representations

### 3.1 Source audit and validity preservation

[Notebook 01](code/notebooks/01_source_audit_and_gaze_quality.ipynb) discovers complete source records, preserves source record and sample identifiers, collects the original fields, records row counts and computes validity/timing quality per record. Its saved numerical outputs are the source manifest, sample-level gaze table and per-record quality table.

The recorded valid-gaze fraction ranges from **0.6735225356 to 1.000000** across the 40 records. The Notebook 07 integrated summary reports an **unweighted mean across records of 0.956931**. These are recording-quality observations, not percentages of people meeting a clinical or quality threshold.

### 3.2 Gap-aware offline motion representation

[Notebook 02](code/notebooks/02_gap_aware_gaze_preprocessing.ipynb) streams the previously saved source sample table in bounded chunks of **80,000 rows**, assembling complete source records for processing. The implementation applies a **centred five-sample rolling median** only within contiguous valid segments and breaks continuity when adjacent time differences exceed **5 ms**.

The calculation retains original `x`/`y`, source flags and identities. It does not fill missing gaze across gaps or interpolate unsupported eye movements. Velocity is derived from adjacent smoothed coordinates and observed time differences, reported in **dva/s**; a segment's first sample has no preceding within-run velocity.

Because the median uses samples on both sides of a point, it is **offline/non-causal** and cannot be described as a demonstrated real-time pipeline. The executed saved motion representation retained all **2,778,008** source rows, and the processed sample CSV occupied **623.4 MiB**.

### 3.3 Peak-window diagnostic and consequential limitation

Notebook 02 separately reviewed the three largest per-record velocity maxima, examining source and smoothed gaze, neighbouring `val` flags, timestamp differences and run continuity. Reported maxima include **27,819.1 dva/s** from `S_1007_S1_TEX` and **24,946.5 dva/s** from `S_1002_S1_TEX`. For the three investigated peak windows, adjacent source flags were `val = 0` and adjacent time differences were **1 ms**; numerically reconstructed speeds matched the saved summaries.

These checks establish **calculation consistency at the selected samples**, not physiological plausibility, tracker correctness or a valid event interpretation. The cause of the very large coordinate changes remains unresolved. The observations were neither deleted nor used to retrospectively alter the detector settings. This limitation propagates to detector-derived speed and event features.

### 3.4 Event-labelled and feature representations

[Notebook 03](code/notebooks/03_ivt_idt_event_detection.ipynb) saves one sample-label row per prepared source row, a method-labelled event-interval table and per-record vendor comparisons. [Notebook 04](code/notebooks/04_gaze_features_and_scanpaths.ipynb) uses the saved motion and event outputs to construct **40 per-record feature rows**, a feature dictionary and an example of a path through **I-VT fixation centroids**.

The centroid sequence is a detector-dependent spatial summary. Without source-stimulus coordinates and verified word/region mapping, it is not a validated reading path or a measurement of comprehension.

## 4. Experimental Design and Evaluation

### 4.1 Exploratory, non-predictive analysis design

No data-driven model is fitted and no held-out Test group is defined. The defined analyses are: a fixed selected source subset; per-record source and motion quality summaries; two provisional detector definitions; participant-aware descriptive statistics; an exploratory, **prespecified 20/30/40-dva/s I-VT threshold grid**; and consistency checks between named producer and consumer files.

Notebook 06 applies its three thresholds to the **same 40 source recordings**, using the unchanged Notebook 02 motion representation and the same run-duration criteria. These are within-source sensitivity conditions, not independent test cohorts or prospectively validated operating points.

### 4.2 Sample-level event and parser comparison

I-VT assigns `fixation`, `saccade` or `unknown`. I-DT is deliberately **fixation-only**, so its remaining samples are `unknown` and **must not be reported as I-DT saccades**. The EyeLink comparison is computed only where a recognised parser reference (fixation/saccade) is available **and** the predicted I-VT sample is evaluable (fixation/saccade).

For each record/threshold the implementation reports the **jointly eligible sample count**, vendor-label coverage, fraction of all samples jointly evaluable, and conditional fraction with matching labels. Notebook 06 records both an unweighted **mean across recording-level agreement values** and an **eligible-sample-weighted** conditional mean; they differ in record weighting, but neither makes the comparison denominator invariant across thresholds or converts parser agreement into expert-ground-truth accuracy.

### 4.3 Independent-person descriptive uncertainty

Notebook 05 aggregates eligible data within each **participant–task** before estimating task-feature means and sample standard deviations. It resamples participants with replacement **500 times** using seed **17**, reporting **2.5th and 97.5th percentile** summaries.

All four tasks contain values from ten people. The bootstrap intervals are exploratory with small independent sample support; the Notebook did **not** calculate paired between-task difference intervals, a task-effect significance test or population transportability. Taskwise intervals cannot be interpreted as independent person groups, because the same ten people completed all four tasks.

## 5. Methods, Models and Configurations

The study uses established analytical algorithms and descriptive methods, **not trained predictive models**. The implementation is in [`code/src/`](code/src/) with Colab utilities under the shared [`python_modules/core/`](../python_modules/core/) directory.

| Analysis / notebook | Implemented configuration or operation | Recorded role | Maintained implementation |
| --- | --- | --- | --- |
| Source audit — 01 | Round 1, Session 1; FXS/HSS/RAN/TEX; first ten complete people | Select and quantify 40 source records | [`gaze_source.py`](code/src/gaze_source.py), [`gaze_quality.py`](code/src/gaze_quality.py) |
| Gap-aware preprocessing — 02 | Centred five-sample median; max 5-ms gap; chunked full-record processing | Preserve and annotate all source rows | [`gaze_streaming.py`](code/src/gaze_streaming.py), [`gaze_preprocessing.py`](code/src/gaze_preprocessing.py) |
| Extreme velocity review — 02 | Three record maxima; original `val` and local `Δt` inspection | Diagnose, not exclude or tune | [`gaze_diagnostics.py`](code/src/gaze_diagnostics.py) |
| I-VT event assignment — 03 | **30 dva/s**, fixation ≥80 ms, saccade ≥6 ms | Provisional fixation/saccade/unknown labels | [`eye_events.py`](code/src/eye_events.py) |
| Fixation-only I-DT — 03 | **1-dva sum-of-axis-range** dispersion limit, fixation ≥80 ms | Provisional fixation/unknown labels | [`eye_events.py`](code/src/eye_events.py) |
| Record features — 04 | Event durations, I-VT peak speed, centroid path and available pupil summaries | 40 algorithm-dependent record features | [`gaze_features.py`](code/src/gaze_features.py) |
| Participant descriptives — 05 | Four prespecified metrics; 500 participant bootstrap draws; seed 17 | 16 task-feature statistics and percentile intervals | [`gaze_statistics.py`](code/src/gaze_statistics.py) |
| Fixed sensitivity grid — 06 | 20/30/40 dva/s; unchanged Notebook 03 run durations | 120 record-threshold comparisons | [`eye_events.py`](code/src/eye_events.py), [`gaze_evidence.py`](code/src/gaze_evidence.py) |
| Evidence integration — 07 | Source-key, row-count, validity, participant-mean and threshold-table correspondence | 40-row alignment; one-row bounded summary | [`gaze_integration.py`](code/src/gaze_integration.py) |

Record-level preprocessing and classification computations are implemented in the scientific modules; the notebooks orchestrate source restoration, display, current-file writing and plots. Colab/VS Code execution is recorded in the committed notebook outputs. This record does **not** establish precise environment pinning or independent re-execution in a second computing environment.

## 6. Results and Comparative Analysis

### 6.1 Source and motion quality: Notebooks 01–02

The source selection yielded **10 participants, 40 recordings and 2,778,008 rows**, with record validity fractions spanning **67.35–100%**. Gap-aware preprocessing kept the full row set and generated a **623.4-MiB** source-keyed motion table.

Extreme calculated speed values of at least **27,819.1** and **24,946.5 dva/s** were recorded. The local peak inspection reconstructed the original calculation but did not resolve whether the coordinates represented reliable eye movement. Subsequent detector-dependent summaries must therefore be interpreted cautiously.

### 6.2 Algorithm-derived event evidence: Notebook 03

| Method and output | Sample/interval class | Recorded count |
| --- | --- | ---: |
| I-VT sample labels | Fixation | 1,627,761 |
| I-VT sample labels | Saccade | 261,354 |
| I-VT sample labels | Unknown | 888,893 |
| I-DT sample labels | Fixation | 2,412,239 |
| I-DT sample labels | Unknown | 365,769 |
| I-VT event intervals | Fixation | 7,015 |
| I-VT event intervals | Saccade | 9,304 |
| I-DT event intervals | Fixation | 5,409 |

Each detector's sample-label counts separately sum to **2,778,008 source rows**. The combined event table contains **21,728 method-labelled intervals** (= 7,015 + 9,304 + 5,409), but I-VT and I-DT are **two analyses of the same gaze recordings**; the combined number must not be interpreted as 21,728 distinct, independent physiological eye movements. There is no independent event-level adjudication.

See [Notebook 03's executed output](code/notebooks/03_ivt_idt_event_detection.ipynb) and its [event-evidence diagram](images/03_ivt_idt_event_detection.svg).

### 6.3 Feature coverage: Notebook 04

Notebook 04 recorded **40 source-aligned feature rows**, including all ten people and four tasks. The saved feature coverage display showed **no missing entries among its reported feature columns**. That observation does not prove every physiological input is complete or that the event-dependent measures are unbiased.

The example [fixation-centroid path](code/notebooks/04_gaze_features_and_scanpaths.ipynb) is empirical in the sense of being derived from saved detector events, but it is **not** a task-normalised, stimulus-aligned or expert-validated reading scanpath.

### 6.4 Participant-balanced task summaries: Notebook 05

Notebook 05 produced **40 participant–task rows** and **16 task-feature statistics** (four features per task); for each statistic the reported independent-person support was **n=10**. The saved feature table matched Notebook 01 source-record quality on `record_id`, participant, task and valid fraction; the maximum saved valid-fraction difference was **0.0**.

| Task | Mean valid gaze fraction | Exploratory bootstrap percentiles (2.5th–97.5th) | Independent people |
| --- | ---: | ---: | ---: |
| FXS | 0.970420 | 0.959669–0.979730 | 10 |
| HSS | 0.942191 | 0.863920–0.992464 | 10 |
| RAN | 0.974758 | 0.947405–0.992207 | 10 |
| TEX | 0.940355 | 0.889566–0.979774 | 10 |

Values are means of participant-level records for this selected subset. The percentile ranges were obtained from 500 draws at the participant level; they are **not** formal paired task-comparison confidence statements. Other recorded statistics include I-VT fixation duration, approximate centroid-path length and mean I-VT saccade peak speed, whose interpretation inherits the unresolved large-velocity issue.

See [Notebook 05](code/notebooks/05_participant_balanced_statistics.ipynb) and its [observed-interval diagram](images/05_participant_balanced_statistics.svg).

### 6.5 I-VT sensitivity and conditional vendor-reference coverage: Notebook 06

The fixed grid produced **120 record–threshold rows**. Across the **40 recordings per setting**, the recorded unweighted means and eligible sample support were:

| Threshold (dva/s) | Fixation fraction | Unknown fraction | Jointly eligible EyeLink samples | Record-mean agreement | Eligible-sample-weighted agreement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.285257 | 0.622422 | 1,015,044 | 0.909743 | 0.947745 |
| 30 | 0.612272 | 0.305744 | 1,888,902 | 0.981307 | 0.980780 |
| 40 | 0.786466 | 0.137801 | 2,357,954 | 0.987844 | 0.986325 |

All **40 records** supplied defined conditional agreement at each threshold. Mean recognised vendor coverage was **0.999708** across records, while the mean predicted/joint evaluable fraction rose from **0.377475** to **0.694083** and **0.862009** at the successive thresholds.

The **30-dva/s record-level vendor metrics matched Notebook 03 for all 40 records**. The increasing eligible denominator is central: the apparent agreement rise is a **conditional comparison over changing sets of samples**, not a fixed-test accuracy gain or proof that 40 dva/s is preferable.

See [Notebook 06](code/notebooks/06_detector_sensitivity_and_vendor_reference.ipynb) and its [denominator-aware scientific figure](images/06_detector_sensitivity_and_vendor_reference.svg).

### 6.6 Source-aligned integration: Notebook 07

The final Notebook read **ten compact current producer tables** and reported successful consistency checks for the 40-record source scope. Its one-row study summary contains:

| Integrated quantity | Executed observation |
| --- | ---: |
| Original sample rows | 2,778,008 |
| Source records aligned | 40 |
| Independent people | 10 |
| Unweighted mean record-valid fraction | 0.956931 |
| Detector-labelled event intervals, summed across methods | 21,728 |
| Record-level feature rows / participant–task rows | 40 / 40 |
| Task–feature descriptive summaries | 16 |
| Fixed I-VT threshold conditions | 3 |
| Maximum saved validity discrepancy (Notebook 01 vs 04) | 0.0 |
| Independent full-pipeline reproduction | **Not performed** |

The alignment check compares record identities, source sample counts, validity fractions, participant support, saved task-feature means, 30-dva/s vendor values and the detailed/summary threshold tables. It does **not** recompute the original gaze preprocessing or validate the physical truth of event assignments.

The saved [Notebook 07 three-panel overview](code/notebooks/07_integrated_evidence_and_reproducibility.ipynb) separately plots participant-mean valid gaze with exploratory bootstrap percentiles, record-mean I-VT class fractions and recorded algorithm-derived event intervals. The first panel explicitly uses a **zoomed y-axis**.

## 7. Robustness, Uncertainty and Additional Analyses

### 7.1 Temporal continuity and anomalous speed checks

Gap-aware preprocessing keeps original samples and segment boundaries, restricting smoothing and velocity comparisons to contiguous valid runs. The three largest observed velocity peaks were investigated at adjacent-sample level, with matching recalculated magnitudes, `val = 0` at both relevant samples and `Δt = 1 ms`. This protects the *traceability* of the calculations but does **not** determine whether gaze jumps were plausible, calibration errors or other artefacts. No post hoc samples were excluded or thresholds retuned to improve scores.

### 7.2 Small-cohort participant-level uncertainty

The 500-draw bootstrap resamples people rather than millions of rows or individual eye events. With only **ten independent people**, the percentile intervals have limited inferential precision; taskwise bootstrap ranges should not be read as independent comparisons across the same participant cohort. No paired task-effect interval, hypothesis test, separate cohort or external validation was run.

### 7.3 Conditional vendor comparison and threshold sensitivity

The 20/30/40-dva/s analysis changes the proportions of fixation, saccade and unknown assignments and the sample population eligible for parser comparison. The additional support analysis makes that selection visible and the 30-dva/s baseline comparison checks internal implementation continuity. Neither is a comparison against independently adjudicated eye movements or a calibrated physiological error rate.

### 7.4 Source alignment and reproduction limits

Notebook 07 reconciles saved producer-level quantities, not raw-source bytes or independently reproduced detector outputs. Its successful checks are evidence that the named compatible tables agree on the tested keys and summaries. A clean rerun in a separately provisioned environment, explicit environment/version records and independent biological annotation would require additional work and are **not** silently inferred from a completed Colab notebook sequence.

## 8. Current Outputs and Reproducibility

The notebooks own and overwrite their numbered current outputs under `eye_tracking_foundations/output/`. The large numerical CSVs and generated PNGs are **ignored by Git by default**, while executed notebook displays and curated scientific SVGs are tracked. The source subset includes selected GazeBase task CSVs already tracked in the public repository. The original publication reports **CC BY 4.0** data licensing; appropriate source and licence attribution plus notice of subset selection are required. The repository's MIT licence covers its own code and documentation, not relicensing the upstream dataset.

| Scientific evidence or artefact | Producing notebook | Current named file / access route |
| --- | --- | --- |
| Source records and validity | 01 | `01_source_manifest.csv`, `01_quality_by_record.csv` and large `01_gaze_samples.csv`; recorded display in [Notebook 01](code/notebooks/01_source_audit_and_gaze_quality.ipynb) |
| Prepared motion and anomaly review | 02 | `02_preprocessed_samples.csv`, `02_preprocessing_summary.csv`, `02_velocity_extreme_review.csv` and two plots; [Notebook 02](code/notebooks/02_gap_aware_gaze_preprocessing.ipynb) |
| Event and vendor evidence | 03 | `03_sample_labels.csv`, `03_events.csv`, `03_vendor_agreement.csv`, `03_event_class_summary.csv`; [Notebook 03](code/notebooks/03_ivt_idt_event_detection.ipynb) |
| Record features | 04 | `04_features_by_record.csv`, `04_feature_dictionary.csv` and scanpath PNG; [Notebook 04](code/notebooks/04_gaze_features_and_scanpaths.ipynb) |
| Participant-level statistics | 05 | `05_participant_features.csv`, `05_descriptive_statistics.csv` and statistics PNG; [Notebook 05](code/notebooks/05_participant_balanced_statistics.ipynb) |
| Threshold sensitivity and eligible support | 06 | `06_sensitivity.csv`, `06_sensitivity_summary.csv`, `06_reference_support.csv` and plot; [Notebook 06](code/notebooks/06_detector_sensitivity_and_vendor_reference.ipynb) |
| Cross-notebook summary and provenance | 07 | `07_source_alignment.csv`, `07_evidence_summary.csv`, `07_study_overview.png`; [Notebook 07](code/notebooks/07_integrated_evidence_and_reproducibility.ipynb) |

The large Notebook 01 sample table is approximately **511.9 MiB** and Notebook 02's processed sample table **623.4 MiB** in the recorded execution. Use the [Getting Started](../GETTING_STARTED.md) and [data/source guide](../data/README.md) to acquire current inputs and restore each notebook's explicitly named output ZIP in Colab. Output ZIPs are private transfer artefacts and are not promised as publicly downloadable GitHub assets.

The shared environment setup uses the maintained [`python_modules/core/`](../python_modules/core/) helpers and study-specific [`code/src/`](code/src/) modules. GitHub Actions exercises **synthetic code/data contracts**, including source/participant alignment and threshold consistency, and checks the separate **50,000,000-byte Git blob ceiling**. Passing those tests does not constitute execution of the original scientific pipeline.

If source selection, preprocessing settings, event definitions or sample labels change, regenerate every affected downstream current file rather than mixing summaries from incompatible executions. The current saved results have **not** been independently reproduced from scratch in this review.

## 9. Discussion, Limitations and Future Work

The seven notebooks fulfil the agreed S1 methodological objective **for a particular ten-person, four-task GazeBase subset**: source identity and validity were audited, original sample rows were preserved through offline motion processing, two provisional event procedures were applied, gaze features and participant-level descriptive uncertainty were recorded, a three-setting I-VT sensitivity study was performed, and the resulting named tables passed specified cross-notebook consistency checks.

The results are **exploratory and source-bounded**, not a claim that eye-event detection is physiologically validated. The principal unresolved methodological issue is the presence of **extremely high calculated gaze speeds even at source-flagged valid, 1-ms-adjacent samples**. Checks established numerical continuity but not cause or correctness. The choice of I-VT/I-DT parameters also remains provisional, EyeLink parser codes provide a limited comparison rather than independent expert annotation, and the ten ordered participants cannot support population-wide claims.

A useful scientific next step, if a stronger event-validity claim is desired, would be to adjudicate or characterise the anomalous coordinate transitions against source acquisition conditions, specify an event-validation protocol and then re-run dependent outputs after any justified methodological correction. A separate wider-source/participant study or verified cognitive target would be **new research**, not retrospective evidence from S1.

Accordingly, S1 may be presented as an **executed, internally reconciled educational/exploratory pipeline with clearly stated scientific limitations**. The evidence does not establish real-time performance, independent reproducibility, expert ground truth, cognitive state estimation, causal effects or clinical deployment. Closing S1's current presentation does **not** close the wider multi-study Eye-Tracking & Neurocognitive Intelligence programme.

**Original-source references:** [Griffith et al. (2021), *Scientific Data*](https://doi.org/10.1038/s41597-021-00959-y); [GazeBase Figshare dataset](https://doi.org/10.6084/m9.figshare.12912257). The upstream GazeBase data are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) according to the source publication; [the repository MIT licence](../LICENSE) applies to original repository code and documentation, not to upstream data.
