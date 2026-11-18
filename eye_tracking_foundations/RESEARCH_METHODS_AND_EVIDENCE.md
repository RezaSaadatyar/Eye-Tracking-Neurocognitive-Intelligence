# Eye-Tracking Foundations & Signal Analysis: Research Methods and Evidence

This document records the initial scientific design and the evidence that remains to be established for S1. The [study README](README.md) provides the concise overview and starting point. **Status:** Planning and scaffold only; no original data have been analysed and no experiments have been executed in this repository.

## 1. Research Objective and Scope

**Research question:** How can continuous eye-position and optional pupil recordings be transformed into reliable, interpretable and appropriately qualified eye-movement events and features?

The initial purpose is educational and methodological: evaluate input quality, explain event definitions and build reusable representations before predictive modelling. Relevant analytical units include gaze samples nested within trials, sessions and participants. S1 does not define a comprehension, disease or neurocognitive prediction target.

The planned comparisons concern transparent event-detection methods, agreement with appropriately documented reference annotations where available, and sensitivity to signal quality and algorithm choices. These are proposed analyses, not findings.

## 2. Data Sources, Cohort and Target Definitions

### Candidate sources and provenance

A bounded [GazeBase](https://doi.org/10.6084/m9.figshare.12912257) subset is the primary candidate; see its [source publication](https://doi.org/10.1038/s41597-021-00959-y). A small [EMTeC](https://doi.org/10.3758/s13428-025-02677-4) raw-and-processed example is a possible complementary source for reading-related corrections. Exact release packages, licences, source files and parser-label availability remain unverified locally.

The Blueprint describes GazeBase as a longitudinal eye-tracking corpus, but source publication counts are not the analysed cohort. No S1 inclusion, exclusion or sampling counts have yet been observed.

### Analytical units and measurements

The future contract will identify participant/session/trial/task and stimulus keys, timestamp units, sample intervals, gaze coordinate frames and units, validity indicators and eye identity where present. Optional pupil, calibration, target-position and reference-event information must be inspected before use.

The intended derived outputs are documented intervals such as fixation, saccade and unknown/invalid segments, followed by event- and trial-level features. Vendor-derived event assignments can support agreement analysis but are not presumed independent ground truth. There is no supervised cognitive outcome in S1.

## 3. Data Preparation and Representations

The proposed source audit will inspect actual timestamps and sample gaps rather than assuming the nominal sampling rate establishes every interval. It will check missingness, invalid gaze, coordinate conventions, plausible ranges, duplicate identifiers and available calibration or pupil fields. Raw, cleaned and event-derived representations must remain distinguishable.

Preprocessing options include carefully justified smoothing and bounded treatment of blinks or signal loss. Smoothing can alter saccade velocities and boundaries, and missing samples must not be interpolated into fabricated valid movements. No concrete filtering threshold, interpolation window or coordinate conversion is approved before data inspection.

Source-to-derived records will retain keys and provenance needed to trace transformations and prevent accidental mixing of trials or incompatible measurement frames.

## 4. Experimental Design and Evaluation

This is initially a signal-processing and method-comparison study, not a prediction benchmark. Planned checks include valid sample coverage, event-duration and boundary distributions, sensitivity to detector thresholds, and event agreement with supported reference labels.

Comparisons will state sample-level versus event-level denominators, temporal tolerance, event taxonomy, units and source/task coverage. Participant and session dependence matters for any uncertainty estimate or inferential comparison; repeated samples and events must not be treated as independent people. A participant-disjoint protocol would be needed before claiming unseen-participant generalisation, but it is not imposed on purely within-record descriptive analysis.

Specific tasks, records, reference annotations, event-matching rules and statistical tests await the data contract. There is no Train/Validation/Test division for S1 unless a separately justified fitted detector or supervised comparison requires one.

## 5. Methods, Models and Configurations

The intended educational methods include velocity-threshold event detection (I-VT) and dispersion-threshold detection (I-DT), with explicit units, time windows, missing-data treatment and deterministic parameter reporting. Adaptive or established detectors such as REMoDNaV or appropriate `pymovements` implementations are candidates for external reference comparison, subject to compatible event definitions.

Planned representations include fixation duration/count, saccade amplitude and velocity, gaze/scanpath geometry, and supported pupil or area-of-interest measures. Reading regressions and AOI transitions require credible stimulus geometry and cannot be inferred from horizontal gaze displacement alone.

No detector has been implemented; no thresholds, model parameters, random seeds, package versions or hardware configurations have been measured or selected. The study does not require deep learning or an artificial classification target to justify its scope.

## 6. Results and Comparative Analysis

**Not yet executed.** The repository contains no S1-derived cohorts, event tables, quality metrics, statistical results or generated scientific plots. The planned output filenames in the README are expected artefacts, not available results. No effectiveness, agreement or method-superiority claim is supported at this stage.

## 7. Robustness, Uncertainty and Additional Analyses

Relevant planned checks are detector-threshold sensitivity, sample-gap and invalid-data handling, differences between available task types, and per-participant/session variation. Where reference events exist, the study should distinguish annotation disagreement from detector error and evaluate event-boundary tolerance.

No robustness study, resampling interval or inferential test has been performed. Statistical choices will be determined after observing the real records and relevant independent units.

## 8. Current Outputs and Reproducibility

The created repository scaffold consists of this document, the [study README](README.md), reserved [notebook](code/notebooks/), [source-code](code/src/), [curated-images](images/) and [output](output/) directories, plus the [repository-level data guide](../data/README.md). There are no numerical outputs, trained models, runnable research notebooks or scientific verification records yet.

The planned stage sequence is (1) source audit and gaze quality, (2) preprocessing and event detection, and (3) features, statistics and scanpaths. Once implemented, each notebook will own stable named outputs and downstream stages will consume the appropriate current files. Source versions, software dependencies, actual compute resources and regeneration instructions will be documented when verified; no installation or execution command is claimed to have been tested.

Original datasets and derived reconstruction caches remain local by default, under source-appropriate paths described in the data guide. The Git repository does not contain participant-level recordings.

## 9. Discussion, Limitations and Future Work

S1 currently establishes a bounded scientific plan, not a validated method or empirical finding. Its immediate dependencies are confirming an accessible licensed GazeBase subset, inspecting representative records and their timing/coordinate conventions, and establishing whether comparison annotations exist for the chosen tasks.

Future authorised implementation will produce source-aware quality summaries, transparent detector comparisons, derived feature tables and scientifically labelled visualisations. Limitations will depend on device characteristics, annotation provenance, pupil availability, source coverage and the definition of a valid event. Neither fixation categories nor pupil changes establish cognitive interpretation on their own; downstream ML and EEG–eye-tracking research will require separate targets and evaluation designs.

### References

- GazeBase dataset: https://doi.org/10.6084/m9.figshare.12912257
- GazeBase source publication: https://doi.org/10.1038/s41597-021-00959-y
- EMTeC source publication: https://doi.org/10.3758/s13428-025-02677-4
