# Eye-Tracking Foundations & Signal Analysis

A seven-notebook research and learning workflow for inspecting GazeBase raw eye-movement recordings, assessing gaze quality, deriving and analysing fixation and saccade events, building source-aware features and reporting bounded evidence.

**Status:** Notebook 01 has a committed run on the selected original GazeBase Round 1 / Session 1 subset (10 participants, 40 task records, 2,778,008 samples). Its outputs show a per-record valid-sample range of 67.35–100%; the corresponding generated CSVs are not committed. Notebooks 02–07 remain unexecuted on this cohort. The initial software testing used synthetic inputs and does not establish detector validity or independent replication.

**Scientific methods and evidence:** [RESEARCH_METHODS_AND_EVIDENCE.md](RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Dataset and Experimental Protocol

The initial candidate is GazeBase v2.0, [source repository](https://doi.org/10.6084/m9.figshare.12912257) and [original dataset paper](https://doi.org/10.1038/s41597-021-00959-y). Its CSV records document gaze position in degrees of visual angle and timestamps in milliseconds. The initial educational subset requires Round 1, Session 1 with FXS, HSS, RAN and TEX records for each selected participant; the current run included ten complete participants (subject source IDs 1001–1010). This is not intended as a random or representative cohort.

Inputs are source-identified gaze recordings, not directly observed cognitive states. Sample validity, EyeLink labels and pupil area are interpreted separately; vendor labels support an agreement comparison only when available. No Train/Validation/Test scheme is imposed on descriptive S1 analyses.

## 2. Experimental Programme

The source audit is configured for up to **ten complete participants** from Round 1, Session 1, with all four tasks (FXS, HSS, RAN and TEX). This selection was observed in the committed Notebook 01 output, not independently reproduced here. The first stage printed 2,778,008 input samples across 40 records. If the source subset changes, rerun Notebook 01 and regenerate dependent stage outputs. Session 2 is not analysed in the current pipeline.


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

**Recorded Notebook 01 evidence:** 10 distinct participants, 40 source recordings and 2,778,008 sample rows. The saved notebook reports valid-sample proportions ranging from 0.6735 to 1.0000 across the 40 records. These are descriptive source-audit findings, not event-detection results or evidence of generalisation. The stage-owned CSVs are not published; inspect the committed notebook execution cells for the recorded counts and quality table. Notebooks 02–07 have not produced real-cohort numerical results; synthetic pipeline checks remain infrastructure evidence only.

## 4. Running and Reviewing the Study

Run the notebooks in ascending order in VS Code with a Colab Python kernel. They begin with a Colab checkout followed by the original Human Movement-style `setup_colab()` interface and grouped imports. Prior-stage files are loaded from explicit paths and not recomputed. Notebook 02 streams complete records from the existing large `01_gaze_samples.csv` to limit peak memory rather than loading two full-cohort tables together. A new runtime can use the previous stage's downloadable ZIP in `/content/`; see the [dataset/ZIP guide](../data/README.md).

Notebook 01 expects extracted CSVs under `data/raw/gazebase/`. The dataset is not automatically downloaded. Notebook 03 uses pedagogical starting thresholds (I-VT 30 dva/s; I-DT dispersion 1 dva with 80-ms minimum duration) which are **not validated scientific selections**. Notebook 06 explores 20/30/40 dva/s as sensitivity settings. Results cannot support clinical, cognitive or person-generalisation claims.

## 5. Files and Saved Outputs

- `code/notebooks/`: seven numbered source notebooks.
- `code/src/`: source-adapter, quality, preprocessing, event, feature and descriptive-statistics modules.
- `images/`: seven editable five-band SVG figures. Figures 01 and 02 contain a short **measured raw gaze excerpt** from `S_1001_S1_FXS.csv`; Figure 01 also reports the committed source-audit counts and per-record validity range, whereas Figure 02's processing diagrams are explicitly conceptual and its derived outputs are still pending execution.
- `output/`: stage-owned machine-readable tables and experimental plots, produced only on execution.
- `../python_modules/core/`: shared notebook infrastructure; see the helper provenance instructions in the development delivery.
- `../data/README.md`: data access, source fields and current-stage ZIP transfers.

Each stage owns its numbered current outputs and replaces them on rerun. Files in `output/` are local by default. Do not infer that a ZIP from an earlier version is scientifically compatible merely because its filename matches; Notebook 07 checks record identities across relevant summaries.

## 6. Interpretation and Scope

Gaze positions are measured estimates; fixation and saccade events are algorithmically derived; vendor event annotations are instrument references rather than ground truth. Report actual timestamp behaviour, missingness, method sensitivity and the number of independent participants, rather than assuming the nominal sample rate or treating many gaze samples as many independent people.

Notebook 01 source counts and validity coverage have been recorded, but the quality table, detector validity and uncertainty interpretation must be reviewed using full compatible stage outputs. Remaining S1 stages require real-data execution. No prospective, clinical or deployed neurocognitive inference is part of S1.
