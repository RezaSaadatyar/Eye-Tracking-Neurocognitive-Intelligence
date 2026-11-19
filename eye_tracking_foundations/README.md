# Eye-Tracking Foundations & Signal Analysis

A seven-notebook research and learning workflow for inspecting GazeBase raw eye-movement recordings, assessing gaze quality, deriving and analysing fixation and saccade events, building source-aware features and reporting bounded evidence.

**Status:** Seven source notebooks, study-specific Python modules and conceptual SVG workflows have been authored. Static checks and synthetic-data pipeline tests were performed in the development environment. **The original GazeBase dataset has not been run or independently reproduced in this delivery.** Scientific output values remain pending execution on the actual selected dataset.

**Scientific methods and evidence:** [RESEARCH_METHODS_AND_EVIDENCE.md](RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Dataset and Experimental Protocol

The initial candidate is GazeBase v2.0, [source repository](https://doi.org/10.6084/m9.figshare.12912257) and [original dataset paper](https://doi.org/10.1038/s41597-021-00959-y). Its CSV records document gaze position in degrees of visual angle and timestamps in milliseconds. The initial educational subset requires Round 1, Session 1 with FXS, HSS, RAN and TEX records for each selected participant; a maximum of three eligible participants is used for an initial bounded inspection. This is not intended as a random or representative cohort.

Inputs are source-identified gaze recordings, not directly observed cognitive states. Sample validity, EyeLink labels and pupil area are interpreted separately; vendor labels support an agreement comparison only when available. No Train/Validation/Test scheme is imposed on descriptive S1 analyses.

## 2. Experimental Programme

The source audit is configured for up to **ten complete participants** from Round 1, Session 1, with all four tasks (FXS, HSS, RAN and TEX). This is a requested upper bound, not an observed cohort size. Notebook 01 reports the number actually discovered; when fewer than ten complete participants exist, subsequent outputs are preliminary and must be regenerated once the intended source subset is available. Session 2 is not analysed in the current pipeline.


| Notebook | Main responsibility | Named current outputs for later stages |
| --- | --- | --- |
| [01 Source Audit and Gaze Quality](code/notebooks/01_source_audit_and_gaze_quality.ipynb) | Source records, schema, sample quality and temporal inspection | `01_gaze_samples.csv`, `01_quality_by_record.csv`, `01_source_manifest.csv`, plot |
| [02 Gap-Aware Preprocessing](code/notebooks/02_gap_aware_gaze_preprocessing.ipynb) | Valid uninterrupted intervals, optional median smoothing and velocity | `02_preprocessed_samples.csv`, preprocessing summary, plot |
| [03 I-VT/I-DT Detection](code/notebooks/03_ivt_idt_event_detection.ipynb) | Per-sample labels, event intervals and vendor reference coverage | `03_sample_labels.csv`, `03_events.csv`, `03_vendor_agreement.csv`, plot |
| [04 Features and Scanpaths](code/notebooks/04_gaze_features_and_scanpaths.ipynb) | Event/record-level features and fixation-centroid paths | `04_features_by_record.csv`, `04_feature_dictionary.csv`, plot |
| [05 Participant-Balanced Statistics](code/notebooks/05_participant_balanced_statistics.ipynb) | Descriptive summaries and exploratory participant-level bootstrap | `05_participant_features.csv`, `05_descriptive_statistics.csv`, plot |
| [06 Sensitivity and Vendor Reference](code/notebooks/06_detector_sensitivity_and_vendor_reference.ipynb) | Fixed I-VT threshold sensitivity using saved processed inputs | `06_sensitivity.csv`, `06_sensitivity_summary.csv`, plot |
| [07 Integrated Evidence Review](code/notebooks/07_integrated_evidence_and_reproducibility.ipynb) | Evidence consistency and lightweight synthesis of prior outputs | `07_evidence_summary.csv`, overview plot |

This is an approved seven-notebook expansion of the earlier three-stage S1 outline. The scientific responsibilities remain S1-only; supervised gaze prediction, deep learning and EEG fusion belong to later studies.

## 3. Main Results

**No GazeBase numerical study results are reported at present.** The software passed targeted static validation and a complete *synthetic* seven-notebook data-flow exercise. Synthetic examples are infrastructure checks, not source observations, detector accuracy or cohort estimates.

## 4. Running and Reviewing the Study

Run the notebooks in ascending order in VS Code with a Colab Python kernel. They begin with a Colab checkout followed by the original Human Movement-style `setup_colab()` interface and grouped imports. Prior-stage files are loaded from explicit paths and not recomputed. A new runtime can use the previous stage's downloadable ZIP in `/content/`; see the [dataset/ZIP guide](../data/README.md).

Notebook 01 expects extracted CSVs under `data/raw/gazebase/`. The dataset is not automatically downloaded. Notebook 03 uses pedagogical starting thresholds (I-VT 30 dva/s; I-DT dispersion 1 dva with 80-ms minimum duration) which are **not validated scientific selections**. Notebook 06 explores 20/30/40 dva/s as sensitivity settings. Results cannot support clinical, cognitive or person-generalisation claims.

## 5. Files and Saved Outputs

- `code/notebooks/`: seven numbered source notebooks.
- `code/src/`: source-adapter, quality, preprocessing, event, feature and descriptive-statistics modules.
- `images/`: seven editable five-band conceptual SVG workflow figures.
- `output/`: stage-owned machine-readable tables and experimental plots, produced only on execution.
- `../python_modules/core/`: shared notebook infrastructure; see the helper provenance instructions in the development delivery.
- `../data/README.md`: data access, source fields and current-stage ZIP transfers.

Each stage owns its numbered current outputs and replaces them on rerun. Files in `output/` are local by default. Do not infer that a ZIP from an earlier version is scientifically compatible merely because its filename matches; Notebook 07 checks record identities across relevant summaries.

## 6. Interpretation and Scope

Gaze positions are measured estimates; fixation and saccade events are algorithmically derived; vendor event annotations are instrument references rather than ground truth. Report actual timestamp behaviour, missingness, method sensitivity and the number of independent participants, rather than assuming the nominal sample rate or treating many gaze samples as many independent people.

The source sample, resulting empirical values, detector validity and uncertainty interpretation must be reviewed after real-data execution. No prospective, clinical or deployed neurocognitive inference is part of S1.
