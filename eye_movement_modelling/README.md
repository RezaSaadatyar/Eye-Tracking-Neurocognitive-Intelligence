# Eye-Movement Modelling & Advanced AI | GazeBase v2.0

![Eye-Movement Modelling gaze-task modelling workflow, source provenance, model families and reserved evaluation](images/study_overview.svg)

*Figure 1. Main 100-person data preparation and eight candidate Validation models are now recorded through Notebook 04. The fixed Validation-based primary selection and the reserved main Test remain pending in Notebook 05. The earlier ten-person pilot is separate historical evidence; measured source traces and schematic elements are distinguished.*

**Aim:** Compare engineered record-level gaze features with compact temporal architectures for four-class GazeBase **recording-protocol identity** (FXS, HSS, RAN, TEX), not cognition, comprehension or clinical status. The historical ten-person pilot completed all five notebooks. The **100-person main cohort has completed Notebooks 01–04**, including all **eight candidate models on the same 64 Validation records from 16 independent participants**. The **20-person main Test** remains reserved for the Validation-selected evaluation in Notebook 05.

**Independence:** Reuse the already available **original GazeBase Round 1 / Session 1 CSV files**, but create all Eye-Movement Modelling splits, features, temporal windows, models and outputs independently in `eye_movement_modelling/output/`. No processed Eye-Tracking Foundations result is mixed into the new modelling cohort.

The ten-person pilot remains preserved in earlier Git revisions. **Current `01_*` through `04_*` artefacts all belong to the 100-person main cohort**. Current `05_*` files still belong to the pilot and will be replaced only when Notebook 05 runs on the main cohort. Read the [scientific methods and evidence](RESEARCH_METHODS_AND_EVIDENCE.md) before interpreting scores.

## 1. Dataset and Experimental Protocol

| Item | Implemented definition and boundary |
| --- | --- |
| Dataset | [GazeBase v2.0](https://doi.org/10.6084/m9.figshare.12912257), Round **1**, Session **1**, source CSVs from [`data/raw/gazebase/`](../data/README.md) |
| Recorded pilot scope | GazeBase Round 1 / Session 1: **10 participants, 40 source recordings, 40 record feature rows and 960 temporal windows**. Notebook 02 fitted four classical candidates on 24 Train records and evaluated 8 Validation records from 2 independent people; the pilot Test was then observed on 8 recordings from 2 independent people in Notebook 05 |
| **Executed main source cohort** | **100 complete people** (source IDs 1001–1108, excluding 1028, 1036, 1050, 1056, 1075, 1082, 1104 and 1105); **400 records**, **9,600 windows**. Main-cohort Notebooks 02–04 fitted all eight candidate models on the same 64 Train / 16 Validation / 20 reserved Test participant split |
| Prediction target | Recording protocol: **FXS**, **HSS**, **RAN**, **TEX**. Four known classes; **not cognitive diagnosis** |
| Predictors: classical | Recording validity, x/y dispersion, step displacement, velocity quantiles, positive pupil coverage/summary where available |
| Predictors: temporal | Fixed 256-sample windows; source-valid contiguous segments only; relative x/y and their first differences (four channels) |
| Window sampling | Up to 24 evenly distributed, **non-overlapping** windows per source record; all windows inherit the parent record Task |
| Group separation | All task records/windows for one person stay in exactly one Train, Validation or Test partition |
| Pilot (10 people) | Seed **17**, participant-disjoint Train/Validation/Test; Test is **exploratory pilot diagnostic**, not definitive unseen-person evidence |
| **Executed main split** | Seed 17, 64 Train / 16 Validation / 20 reserved Test people; corresponding 256 / 64 / 80 original recordings and 6,144 / 1,536 / 1,920 temporal windows |
| Train-only operations | Temporal channel normalisation and classical feature imputation/scaling fitted only on Training people |
| Selection and evaluation | Validation record-level Macro F1, then Balanced Accuracy, then fixed model priority. Frozen Test opened in Notebook 05 only |
| Source limitation | Different task stimuli and recording protocols may dominate task recognition; high task accuracy would not imply cognition |

**Do not change only the cohort-size setting after inspecting the main Test and present the new Test as untouched evidence.** Decide the final cohort before the main evaluation; rerun the full producer-to-consumer chain whenever it changes.

## 2. Experimental Programme

These five notebooks form one task-recognition study. **Notebooks 01–04 are complete for the 100-person main cohort; Notebook 05 remains pending**. The ten-person pilot's Test was exploratory and must not be confused with the still-reserved **20-person main Test**.

| Notebook | Main experiment and visible model-level evidence | Owned current output files |
| --- | --- | --- |
| [01 — Data and Experimental Protocol](code/notebooks/01_data_and_experimental_protocol.ipynb) | **Executed on 100 people**: 400 records, 9,600 windows, full task audit and independent 64/16/20 person split; nine current stage files packaged | `01_source_inventory.csv`, `01_split_assignments.csv`, `01_record_features.csv`, `01_sequence_windows.npz`, `01_sequence_normalisation.csv`, `01_protocol.csv`, `01_source_distribution.png`; source-audit files `01_source_acquisition_preflight.csv` and `01_source_acquisition_task_summary.csv` |
| [02 — Statistical and Classical ML](code/notebooks/02_statistical_and_classical_ml.ipynb) | **Executed on 100 people:** four fixed candidates fitted on 256 Train records, scored on 64 Validation recordings from 16 people; four fitted pipelines and 14 stage-owned outputs | `02_validation_metrics.csv`, four `02_*.joblib` pipelines, four prediction tables, four confusion plots and comparison plot |
| [03 — CNN and TCN](code/notebooks/03_cnn_and_tcn_models.ipynb) | **Executed on 100 people:** CNN **62/64** (Macro F1 **0.968735**) and TCN **61/64** (Macro F1 **0.953036**) on the 16-person Validation set. Selected epochs 7/6; parameter counts 11,492/12,772 | `03_cnn.pt`, `03_tcn.pt`, histories, 64-row predictions, evaluation plots and `03_neural_validation_metrics.csv` (10 files) |
| [04 — GRU and Transformer](code/notebooks/04_gru_and_transformer_models.ipynb) | **Executed on 100 people:** GRU **59/64** (Macro F1 **0.920940**, epoch 7) and Transformer **63/64** (Macro F1 **0.984360**, epoch 8) on 16 independent Validation participants; 3,780/33,764 parameters | `04_gru.pt`, `04_transformer.pt`, 64-row predictions, histories, figures and `04_neural_validation_metrics.csv` (10 files) |
| [05 — Robustness and Final Evaluation](code/notebooks/05_robustness_and_final_evaluation.ipynb) | **Next main-cohort stage:** validate all eight current candidate results, freeze the primary model using Validation only, then open the reserved 20-person main Test; tracked `05_*` still describes the historical ten-person pilot | `05_model_selection.csv`, `05_test_predictions.csv`, `05_test_metrics.csv`, `05_groupwise_test_metrics.csv`, `05_robustness_validation.csv`, comparison figures |

A separate editable five-band figure accompanies each notebook in [`images/`](images/). The [overview](images/study_overview.svg) shows all **eight completed main-cohort Validation candidates** and the pending main Test. The Notebook 04 figure now records the observed GRU/Transformer performance. Notebook 05's existing introductory diagram illustrates the historical pilot until the main results are available.

## 3. Main Results

**Observed 100-person main source preparation (Notebook 01):** The recorded GazeBase Round 1 / Session 1 source audit selected **100 complete participants** from IDs **1001–1108**, skipping eight unavailable/incomplete IDs. The execution produced **400 records** (100 per task), **400 feature rows** and **9,600 windows** of shape `(256, 4)`, with 24 windows per record. Train/Validation/reserved Test are **64/16/20 participants** (256/64/80 recordings). The audit reported **zero non-increasing timestamp transitions**; mean valid fractions were about **0.960 FXS, 0.964 HSS, 0.970 RAN and 0.947 TEX** (minimum individual value approximately **0.478**). The compressed temporal NPZ is approximately **19.7 MB**.

**Observed 100-person classical Validation (Notebook 02):** Four fixed models were fitted on **256 Train recordings from 64 people** and evaluated on the same **64 Validation recordings from 16 independent people**. Source identities, target labels, probability sums and metrics agree with the committed CSVs. The **20-person main Test was not opened**.

| Main Validation model | Accuracy | Macro F1 | Balanced accuracy | Correct records |
| --- | ---: | ---: | ---: | ---: |
| Class Prior | 0.2500 | 0.1000 | 0.2500 | 16/64 |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 64/64 |
| Random Forest | 0.9844 | 0.9844 | 0.9844 | 63/64 |
| Gradient Boosting | 0.9844 | 0.9844 | 0.9844 | 63/64 |

Both nonlinear models misclassified one TEX recording as RAN. High task discrimination may reflect source stimulus and recording-protocol differences; **Logistic Regression is not yet the main-study selected model**, since the temporal models' Validation results are pending.

**Observed 100-person CNN/TCN Validation (Notebook 03):** Both networks trained only on **6,144 Train windows from 64 people**, with checkpoints chosen using **64 Validation recordings from 16 disjoint people**, each represented by 24 windows. Current predictions, saved histories and metrics were independently checked for source/label alignment and numerical consistency.

| Main temporal model | Accuracy | Macro F1 | Balanced accuracy | Correct records | Selected epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.968750 | 0.968735 | 0.968750 | 62/64 | 7 |
| TCN | 0.953125 | 0.953036 | 0.953125 | 61/64 | 6 |

CNN misclassified one HSS as RAN and one RAN as FXS. TCN misclassified one HSS as RAN and two RAN as FXS. CNN had slightly higher Validation Macro F1; this is **not** final Test performance or evidence of superiority across datasets or acquisition protocols. **Main Test was not used**.

**Observed 100-person GRU/Transformer Validation (Notebook 04):** Both models used the frozen source windows: **6,144 Train windows from 64 people** and 64 original Validation recordings (24 windows each) from 16 independent people. Predictions, task labels, class-probability sums, saved metrics and selected epochs were independently checked against current source identities.

| Main temporal model | Accuracy | Macro F1 | Balanced accuracy | Correct records | Selected epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU | 0.921875 | 0.920940 | 0.921875 | 59/64 | 7 |
| Transformer | 0.984375 | 0.984360 | 0.984375 | 63/64 | 8 |

The GRU made five record-level errors; Transformer misclassified one HSS record as RAN. The Transformer checkpoint selected epoch 8 after tying epoch 6 on Macro F1 and achieving lower Validation window loss. **Neither model used the reserved main Test.** The main study's Validation-only ranking currently favours Logistic Regression (Macro F1 1.000), but the official primary selection belongs to Notebook 05.

**Historical ten-person pilot results (Notebooks 01–05):** These earlier results are **not** current 100-person scores. The tracked `05_*` outputs still belong to the pilot; older `02_*`–`04_*` pilot scores remain recoverable from Git history.

**Observed ten-person pilot:** Notebook 01 processed 40 original recordings into 40 feature rows and 960 temporal windows of shape `(960, 256, 4)`, with 6/2/2 Train/Validation/Test participants (24/8/8 recordings). Notebook 02 evaluated four classical candidates on the same **eight Validation recordings from two people**. The class prior scored **Accuracy 0.25, Macro F1 0.10, Balanced Accuracy 0.25**. Logistic Regression, Random Forest and Gradient Boosting each scored **1.00 on all three metrics** (8/8 correct). These are **exploratory participant-disjoint Validation results**, not independent population estimates, external validation or final Test results.

**Historical pilot Notebook 03 Validation:** CNN and TCN both classified 8/8 Validation recordings correctly (Accuracy, Macro F1 and Balanced Accuracy all **1.00**). CNN selected epoch 8 with **11,492 trainable parameters**; TCN selected epoch 5 with **12,772**. Only two independent Validation participants contributed; this earlier pilot evidence must not be conflated with current 100-person results.

Notebooks 01–04 established the pilot data contract and eight Validation candidates; Notebook 05 applied the predefined Validation ranking, froze the primary choice and recorded pilot Test metrics. Inspect source step-speed artefacts and stimulus/task confounding before interpreting apparently high scores. The 10-person experiment is a functional pilot, not a new population-level model validation.

**Historical pilot Notebook 04 Validation:** GRU selected epoch **1** (3,780 parameters) and correctly classified **2/8** Validation records, predicting TEX for all eight (Macro F1 0.10). Transformer selected epoch **2** (33,764 parameters) and correctly classified **4/8** (Macro F1 0.375), recognising HSS/RAN but missing FXS/TEX. Only two Validation participants contributed to the pilot; these are not the current main-cohort scores. The pilot's downstream Test is historical exploratory evidence.

**Recorded Notebook 05 pilot Test:** Logistic Regression was selected using **Validation Macro F1, Balanced Accuracy and fixed candidate priority**, before the notebook's Test-scoring section. Among five candidates with Validation Macro F1 **1.00**, its prespecified tie-priority made it the primary model. On the **eight Test records belonging to participants 001 and 005**, it achieved **Accuracy 0.875, Macro F1 0.867 and Balanced Accuracy 0.875** (7/8 correct). Its only error was `S_1005_S1_FXS` predicted as TEX. Participant 001 had 4/4 correct; participant 005 had 3/4 correct.

| Candidate | Validation Macro F1 | Pilot Test accuracy | Pilot Test Macro F1 |
| --- | ---: | ---: | ---: |
| Class Prior | 0.100 | 0.250 | 0.100 |
| **Logistic Regression (selected)** | **1.000** | **0.875** | **0.867** |
| Random Forest | 1.000 | 0.875 | 0.867 |
| Gradient Boosting | 1.000 | 0.875 | 0.867 |
| CNN | 1.000 | 0.750 | 0.733 |
| TCN | 1.000 | 0.750 | 0.742 |
| GRU | 0.100 | 0.250 | 0.100 |
| Transformer | 0.375 | 0.500 | 0.333 |

All model comparisons in the Test column are **secondary descriptions**. The selected model's **Validation-only pupil-feature masking** left Macro F1 unchanged (**1.00 to 1.00**, eight records); this is not proof of feature irrelevance or causal explainability. Only **two independent people** contributed to each split's Validation or Test, so scores cannot support stable inference or claims of generalisation. The current pilot Test has been examined and **must not be reused as untouched Test evidence** when adapting the experiment.

## 4. Running and Reviewing the Study

**Notebooks 01–04 are complete for the 100-person main-cohort stages.** The current features, normalised temporal inputs, four classical pipelines and four temporal checkpoints, histories and predictions are all committed under `eye_movement_modelling/output/`. New Colab sessions can read those outputs directly without downloading GazeBase or rerunning training.

1. Open [Notebook 05: Final Evaluation](code/notebooks/05_robustness_and_final_evaluation.ipynb) from the current `main` branch in a fresh Colab Runtime.
2. Read the **current** `01_*` through `04_*` output files already tracked in GitHub. A matching stage ZIP in `/content/` is optional; do not upload pilot ZIPs or mix cohorts.
3. Confirm **100 participants**, the fixed **64/16/20** person Train/Validation/reserved Test split and **64 identical Validation recordings** for all eight candidates. The notebook now recalculates every Validation metric before selection.
4. Run the predefined ranking (Validation Macro F1, Balanced Accuracy, then fixed candidate priority) and save the selected model **before** the separate Test-scoring section. The 20-person main Test should be evaluated only once and not used to change the selection.
5. Inspect the selected model's record-level Test confusion, per-person results and the descriptive Validation-only sensitivity analysis. Review the nine stage-05 files and commit the completed notebook and updated outputs.

A later 300-person experiment requires a separately frozen cohort, complete regeneration of its data and models, and a clearly labelled evaluation distinct from the observed 100-person Test. Do not mix ZIPs or checkpoints from different participant sets.

## 5. Files and Saved Outputs

| Location | Purpose |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Five authored Jupyter notebooks, each with structured Markdown and independent result cells |
| [`code/src/`](code/src/) | Study-specific data preparation, classical learners, temporal architectures, training and record-level metrics |
| [`images/study_overview.svg`](images/study_overview.svg) | Editable study-wide five-band modelling diagram |
| [`images/`](images/) | Five notebook-specific editable scientific SVGs |
| `output/` | Current Eye-Movement Modelling feature CSVs, window NPZ, checkpoints, metrics and figures; **can be tracked after review**, subject to the 50,000,000-byte file-size policy |
| [`data/raw/gazebase/`](../data/README.md) | Shared upstream **original** source CSVs (read only) |
| [`README.md`](README.md) | Study overview and navigation |
| [`RESEARCH_METHODS_AND_EVIDENCE.md`](RESEARCH_METHODS_AND_EVIDENCE.md) | Detailed training, split, metric and interpretation plan |
| [`tests/test_s2_contracts.py`](../tests/test_s2_contracts.py) | Synthetic contracts, not original-data reproduction |

A per-study output directory prevents overwriting Eye-Tracking Foundations's `eye_tracking_foundations/output/`. Study outputs are no longer ignored by Git. Raw and processed source datasets remain ignored. Gitignore cannot filter on size; the independent Git blob policy rejects unapproved files larger than 50,000,000 bytes. Model weights are not automatically added or published.

## 6. Interpretation and Scope

**The historical ten-person pilot is complete. The 100-person main study has completed Notebooks 01–04**, with all eight models evaluated on the same 16 independent Validation participants. **Final model selection and the 20-person main Test remain pending in Notebook 05.** Current `05_*` outputs still belong to the pilot. Strong within-GazeBase task recognition does not demonstrate cognitive inference, clinical performance or generalisation across sessions, stimuli or devices.

The following remain limitations even after a successful run: task identity is confounded with stimulus protocol; Test participant counts are very small in the pilot; the first complete source IDs are not a representative sample; gaze speeds and pupil fields may reflect measurement artefacts; and a model's Validation/held-out Test score does **not** establish neural, clinical or cognitive diagnosis.

**GazeBase attribution:** [Griffith, Lohr, Abdulin and Komogortsev, *Scientific Data* (2021)](https://doi.org/10.1038/s41597-021-00959-y); [dataset](https://doi.org/10.6084/m9.figshare.12912257), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is separate from the repository's MIT-licensed code.
