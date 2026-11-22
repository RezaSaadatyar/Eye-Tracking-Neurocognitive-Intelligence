# Eye-Movement Modelling & Advanced AI | GazeBase v2.0

![Eye-Movement Modelling gaze-task modelling workflow, source provenance, model families and reserved evaluation](images/study_overview.svg)

*Figure 1. Five-stage Eye-Movement Modelling workflow on the completed 100-person main cohort. The eight fixed candidate models have compatible Validation results, the predefined primary selection precedes Test, and the observed main Test comprises 80 original recordings from 20 independent people. The ten-person pilot is separate historical evidence; measured gaze and schematic elements are distinguished.*

**Aim:** Compare engineered record-level gaze features with compact temporal models for four-class GazeBase **recording-protocol identity** (FXS, HSS, RAN, TEX), not cognition, comprehension or clinical status. The historical ten-person pilot completed all five stages. **The 100-person main study has now completed Notebooks 01–05** using 400 source recordings, 9,600 temporal windows, eight compatible Validation models and a reserved 20-person Test. The **Validation-selected Logistic Regression classified all 80 main Test recordings correctly** (Accuracy and Macro F1 **1.000**). This bounded within-source result does not establish generalisation across sessions, stimuli, devices or populations.

**Independence:** Reuse the already available **original GazeBase Round 1 / Session 1 CSV files**, but create all Eye-Movement Modelling splits, features, temporal windows, models and outputs independently in `eye_movement_modelling/output/`. No processed Eye-Tracking Foundations result is mixed into the new modelling cohort.

The ten-person pilot remains preserved in earlier Git revisions. **All current `01_*` through `05_*` outputs belong to the 100-person main study**; earlier pilot results are labelled separately below and in the scientific report. The selected main Test has been examined and is not an independent confirmation set for future Test-informed model redesign.

## 1. Dataset and Experimental Protocol

| Item | Implemented definition and boundary |
| --- | --- |
| Dataset | [GazeBase v2.0](https://doi.org/10.6084/m9.figshare.12912257), Round **1**, Session **1**, source CSVs from [`data/raw/gazebase/`](../data/README.md) |
| Recorded pilot scope | GazeBase Round 1 / Session 1: **10 participants, 40 source recordings, 40 record feature rows and 960 temporal windows**. Notebook 02 fitted four classical candidates on 24 Train records and evaluated 8 Validation records from 2 independent people; the pilot Test was then observed on 8 recordings from 2 independent people in Notebook 05 |
| **Executed main source cohort** | **100 complete people** (source IDs 1001–1108, excluding 1028, 1036, 1050, 1056, 1075, 1082, 1104 and 1105); **400 records**, **9,600 windows**. All five notebooks completed using a 64 Train / 16 Validation / 20 Test participant split; the main Test includes 80 source recordings and is now evaluated |
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

These five notebooks form one task-recognition study. **Notebooks 01–05 have completed the 100-person main-cohort workflow.** Model selection used Validation only, followed by record- and person-level Test evaluation. The historical ten-person pilot remains separately documented.

| Notebook | Main experiment and visible model-level evidence | Owned current output files |
| --- | --- | --- |
| [01 — Data and Experimental Protocol](code/notebooks/01_data_and_experimental_protocol.ipynb) | **Executed on 100 people**: 400 records, 9,600 windows, full task audit and independent 64/16/20 person split; nine current stage files packaged | `01_source_inventory.csv`, `01_split_assignments.csv`, `01_record_features.csv`, `01_sequence_windows.npz`, `01_sequence_normalisation.csv`, `01_protocol.csv`, `01_source_distribution.png`; source-audit files `01_source_acquisition_preflight.csv` and `01_source_acquisition_task_summary.csv` |
| [02 — Statistical and Classical ML](code/notebooks/02_statistical_and_classical_ml.ipynb) | **Executed on 100 people:** four fixed candidates fitted on 256 Train records, scored on 64 Validation recordings from 16 people; four fitted pipelines and 14 stage-owned outputs | `02_validation_metrics.csv`, four `02_*.joblib` pipelines, four prediction tables, four confusion plots and comparison plot |
| [03 — CNN and TCN](code/notebooks/03_cnn_and_tcn_models.ipynb) | **Executed on 100 people:** CNN **62/64** (Macro F1 **0.968735**) and TCN **61/64** (Macro F1 **0.953036**) on the 16-person Validation set. Selected epochs 7/6; parameter counts 11,492/12,772 | `03_cnn.pt`, `03_tcn.pt`, histories, 64-row predictions, evaluation plots and `03_neural_validation_metrics.csv` (10 files) |
| [04 — GRU and Transformer](code/notebooks/04_gru_and_transformer_models.ipynb) | **Executed on 100 people:** GRU **59/64** (Macro F1 **0.920940**, epoch 7) and Transformer **63/64** (Macro F1 **0.984360**, epoch 8) on 16 independent Validation participants; 3,780/33,764 parameters | `04_gru.pt`, `04_transformer.pt`, 64-row predictions, histories, figures and `04_neural_validation_metrics.csv` (10 files) |
| [05 — Robustness and Final Evaluation](code/notebooks/05_robustness_and_final_evaluation.ipynb) | **Executed on 100 people:** selected Logistic Regression using 64 Validation records, then evaluated all eight frozen models on **80 Test records from 20 people**; primary achieved **80/80** and Macro F1 **1.000**. The selected model's pupil-feature masking left Validation Macro F1 **1.000 → 1.000** | `05_model_selection.csv`, 640-row Test predictions, eight-model Test metrics, 160 person-level rows, Validation robustness table and four plots (nine stage files) |

A separate editable five-band figure accompanies each notebook in [`images/`](images/). The [overview](images/study_overview.svg) now describes the completed main study. The Notebook 05 figure presents current 100-person selection and Test evidence; historical ten-person pilot figures remain recoverable from prior Git revisions.

## 3. Main Results

**Observed 100-person main source preparation (Notebook 01):** The recorded GazeBase Round 1 / Session 1 source audit selected **100 complete participants** from IDs **1001–1108**, skipping eight unavailable/incomplete IDs. The execution produced **400 records** (100 per task), **400 feature rows** and **9,600 windows** of shape `(256, 4)`, with 24 windows per record. Train/Validation/reserved Test are **64/16/20 participants** (256/64/80 recordings). The audit reported **zero non-increasing timestamp transitions**; mean valid fractions were about **0.960 FXS, 0.964 HSS, 0.970 RAN and 0.947 TEX** (minimum individual value approximately **0.478**). The compressed temporal NPZ is approximately **19.7 MB**.

**Observed 100-person classical Validation (Notebook 02):** Four fixed models were fitted on **256 Train records from 64 people** and compared on **64 Validation records from 16 disjoint people**. Original source IDs, labels, class-probability sums and recorded metrics match the committed CSVs. The Test was preserved through this selection stage and evaluated only later in Notebook 05.

| Main Validation model | Accuracy | Macro F1 | Balanced accuracy | Correct records |
| --- | ---: | ---: | ---: | ---: |
| Class Prior | 0.2500 | 0.1000 | 0.2500 | 16/64 |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 64/64 |
| Random Forest | 0.9844 | 0.9844 | 0.9844 | 63/64 |
| Gradient Boosting | 0.9844 | 0.9844 | 0.9844 | 63/64 |

Both nonlinear classical models misclassified one TEX record as RAN on Validation. These high scores may reflect task-specific stimuli or acquisition conditions; **Logistic Regression was subsequently selected as the primary model in Notebook 05** based on Validation Macro F1 **1.000**, not because of Test performance.

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

The GRU made five record-level Validation errors; Transformer classified one HSS record as RAN. The Transformer selected epoch 8 after tying epoch 6 in Macro F1 and achieving lower Validation window loss. This development comparison did not use Test and does not support claims about cognition or cross-session generalisation.

**Observed 100-person final selection and Test (Notebook 05):** All eight candidate models were ranked on the **same 64-record Validation set (16 people)** using the declared Macro F1, Balanced Accuracy and fixed-priority order. **Logistic Regression** was selected before the reserved Test was scored. Each model subsequently produced one prediction for each of **80 Test records from 20 independent participants**; the main Test was not used to revise the primary choice.

| Frozen candidate | Validation Macro F1 | Main Test accuracy | Main Test Macro F1 | Correct / 80 |
| --- | ---: | ---: | ---: | ---: |
| Class Prior | 0.1000 | 0.2500 | 0.1000 | 20 |
| **Logistic Regression (selected)** | **1.0000** | **1.0000** | **1.0000** | **80** |
| Random Forest | 0.9844 | 1.0000 | 1.0000 | 80 |
| Gradient Boosting | 0.9844 | 0.9875 | 0.9875 | 79 |
| CNN | 0.9687 | 0.9500 | 0.9502 | 76 |
| TCN | 0.9530 | 0.8375 | 0.8238 | 67 |
| GRU | 0.9209 | 0.8625 | 0.8649 | 69 |
| Transformer | 0.9844 | 0.9250 | 0.9250 | 74 |

The selected Logistic Regression classified **all four task recordings for each of the 20 Test people** correctly. This perfect observed Test score is not a guarantee of performance on an independent dataset, different stimulus protocol or future session. The model's Validation-only pupil-feature masking retained Macro F1 **1.000 → 1.000** on 64 records, which is a limited sensitivity observation rather than an explanation or proof that pupil fields are irrelevant.

**Pilot context:** A smaller ten-person feasibility run preceded this fixed 100-person study. Its eight-record Test estimates are historical, not additional evidence for the main Test. Detailed pilot methods and results are retained in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md); the earlier outputs are recoverable from Git history.

## 4. Running and Reviewing the Study

**All five notebooks are complete for the 100-person main-cohort workflow.** Current source features, temporal windows, classical pipelines, temporal checkpoints, Validation predictions and the nine final Test output files are committed under `eye_movement_modelling/output/`. There is no need to repeat fitting or Test evaluation to inspect the saved evidence.

1. Start with the [completed Notebook 05](code/notebooks/05_robustness_and_final_evaluation.ipynb): the primary decision appears before Test scoring, followed by confusion, person-level and descriptive sensitivity analyses.
2. Inspect the committed [output directory](output/) for the primary decision, aggregate metrics, 640 Test predictions, person-level summaries and upstream model artefacts.
3. To reproduce the study, run [Notebooks 01–05](code/notebooks/) in numerical order using the same original GazeBase source cohort and frozen protocol. All stages write their named results under `output/`.
4. Interpret the primary selection using Validation Macro F1, Balanced Accuracy and fixed priority; the Test-only model comparisons must not retrospectively change that choice.
5. For preprocessing, model settings, detailed results, sensitivity and limitations, see [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

A different cohort or acquisition protocol requires a separately declared study design and an evaluation that is not retrospectively selected using this observed Test.

## 5. Files and Saved Outputs

| Location | Purpose |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Five authored Jupyter notebooks, each with structured Markdown and independent result cells |
| [`code/src/`](code/src/) | Study-specific data preparation, classical learners, temporal architectures, training and record-level metrics |
| [`images/study_overview.svg`](images/study_overview.svg) | Editable study-wide five-band modelling diagram |
| [`images/`](images/) | Five notebook-specific editable scientific SVGs |
| [`output/`](output/) | Committed source registries, derived features and windows, eight model artefacts, Validation predictions and final Test evidence |
| [`data/raw/gazebase/`](../data/README.md) | Shared upstream **original** source CSVs (read only) |
| [`README.md`](README.md) | Study overview and navigation |
| [`RESEARCH_METHODS_AND_EVIDENCE.md`](RESEARCH_METHODS_AND_EVIDENCE.md) | Detailed implemented methods, results, evidence and limitations |
| [`tests/test_s2_contracts.py`](../tests/test_s2_contracts.py) | Synthetic contracts, not original-data reproduction |

Study-specific outputs are kept separate from Eye-Tracking Foundations. The current derived features, model checkpoints and evaluation files are tracked; the original GazeBase archive is not duplicated as a model-stage output. The repository's file-size policy rejects individual files exceeding 50,000,000 bytes.

## 6. Interpretation and Scope

**The historical ten-person pilot and the five-stage 100-person main GazeBase study are now complete.** On the main held-out Test (20 independent people; 80 task recordings), the Validation-selected Logistic Regression achieved **Accuracy 1.000, fixed-label Macro F1 1.000 and Balanced Accuracy 1.000**. Random Forest also scored 80/80 but did not replace the selected model. Current `05_*` outputs belong to the main cohort. These results demonstrate within-GazeBase task-protocol discrimination, **not** neurocognitive interpretation, clinical prediction, or generalisation across sessions, stimuli or devices.

The following remain limitations even after a successful run: task identity is confounded with stimulus protocol; Test participant counts are very small in the pilot; the first complete source IDs are not a representative sample; gaze speeds and pupil fields may reflect measurement artefacts; and a model's Validation/held-out Test score does **not** establish neural, clinical or cognitive diagnosis.

**GazeBase attribution:** [Griffith, Lohr, Abdulin and Komogortsev, *Scientific Data* (2021)](https://doi.org/10.1038/s41597-021-00959-y); [dataset](https://doi.org/10.6084/m9.figshare.12912257), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is separate from the repository's MIT-licensed code.
