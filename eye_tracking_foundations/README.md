# Eye-Tracking Foundations & Signal Analysis

This first study investigates how continuous gaze samples can be inspected, quality-assured and converted into interpretable eye-movement events, features and visualisations. It is designed as a self-contained methodological introduction before predictive modelling or EEG–eye-tracking fusion.

**Status:** Initial structure and documentation created. The dataset and experimental subset are not finalised, and no S1 notebook, reusable analysis module or research output has been implemented or executed.

**Detailed scientific account:** [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md)

## 1. Dataset and Experimental Protocol

The primary candidate is a bounded, representative subset of [GazeBase](https://doi.org/10.6084/m9.figshare.12912257), documented in its [dataset publication](https://doi.org/10.1038/s41597-021-00959-y). A small [EMTeC](https://doi.org/10.3758/s13428-025-02677-4) example could later complement it if source-level correction or reading-layout analysis warrants a second source. Neither the exact subset nor the accompanying event annotations have been selected or inspected in this repository.

The planned input contract includes source and release identifiers; participant, session, trial and task keys; sample timestamps; gaze coordinates and their units; validity information; and optional pupil, calibration, stimulus and event fields when genuinely present.

S1 is an analytical and event-detection study. Its derived fixation and saccade intervals must not be treated as directly observed cognitive states or as independent ground-truth diagnoses.

## 2. Experimental Programme

Three integrated notebooks are planned, subject to verifying real source files and the study contract. They are not yet present.

| Planned notebook | Main purpose | Intended current outputs (not created) |
| --- | --- | --- |
| `01_source_audit_and_gaze_quality.ipynb` | Source inventory, timebase, missingness, data quality and initial gaze plots | `01_source_manifest.csv`, `01_quality_by_record.csv`, `01_gaze_quality.png` |
| `02_preprocessing_and_eye_event_detection.ipynb` | Defensible preprocessing, gap handling, I-VT/I-DT and reference-detector comparison | `02_events.csv`, `02_detector_comparison.csv`, `02_threshold_sensitivity.png` |
| `03_features_statistics_and_scanpaths.ipynb` | Gaze-event features, scanpaths, descriptive statistics and appropriate uncertainty | `03_feature_dictionary.csv`, `03_feature_table.csv`, `03_summary_statistics.csv`, `03_scanpath_examples.png` |

Each future notebook will write its own named current outputs to `output/`. Downstream stages will read explicit compatible files, rather than selecting historical copies or automatically launching expensive processing.

## 3. Main Results

No dataset-level analysis, eye-event detection, statistical comparison or model evaluation has been run in this repository. Consequently, no sample counts, event agreement scores, figures or scientific conclusions are reported.

## 4. Running and Reviewing the Study

Start with the [scientific plan](RESEARCH_METHODS_AND_EVIDENCE.md) and the repository's [data guide](../data/README.md). The [notebooks location](code/notebooks/) is currently an empty scaffold: no executable environment, installation command or Colab notebook is available yet.

The first authorised implementation step is to inspect a limited original-source subset and document its real schema, timing, coordinate units and recording identifiers. Only then should the numbered notebook sequence and its reusable methods be implemented and run.

## 5. Files and Saved Outputs

| Location | Purpose and present availability |
| --- | --- |
| [`RESEARCH_METHODS_AND_EVIDENCE.md`](RESEARCH_METHODS_AND_EVIDENCE.md) | Detailed design, evidence status and scientific limitations; available |
| [`code/notebooks/`](code/notebooks/) | Reserved for the three planned integrated notebooks; no notebooks yet |
| [`code/src/`](code/src/) | Reserved for S1-specific reusable functions; no modules yet |
| [`images/`](images/) | Reserved for curated explanatory figures; no scientific figures yet |
| [`output/`](output/) | Reserved for generated named result files; no results yet |
| [`../data/README.md`](../data/README.md) | Candidate sources, acquisition and provenance guidance |

The reserved directories contain only Git tracking placeholders where needed. No user dataset is included in Git.

## 6. Interpretation and Scope

S1 aims to make the relationship between measured gaze signals, quality control, algorithmically derived events and subsequent features transparent. Vendor event labels, where available, provide a comparison reference rather than an unquestioned gold standard. Any future numerical comparison must identify data coverage, units, parameter sensitivity and the level at which observations are independent.

S1 does not measure comprehension, attention disorders or other clinical/cognitive outcomes. Device-specific interpretation, event taxonomy, pupil availability and stimulus mapping remain dependent on the eventual inspected release. Supervised modelling and multimodal EEG analysis belong to separately designed later studies.
