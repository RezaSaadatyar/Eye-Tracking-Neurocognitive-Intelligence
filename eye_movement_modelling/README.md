# Eye-Movement Modelling & Advanced AI | GazeBase v2.0

![Eye-Movement Modelling gaze-task modelling workflow, source provenance, model families and reserved evaluation](images/study_overview.svg)

*Figure 1. Programme status following 100-person source preparation and completed classical Validation. Notebook 01 prepared the main cohort; Notebook 02 recorded four classical Validation models; Notebooks 03–05 remain pending on the main cohort. The ten-person pilot is historical evidence. Measured source traces are distinct from schematic workflow elements.*

**Aim:** Compare record-level engineered gaze features and four compact temporal architectures for the **four-class GazeBase recording-protocol target** (FXS, HSS, RAN, TEX). All five notebooks have a recorded historical ten-person pilot. The **100-person main cohort has completed Notebooks 01 and 02**, with 400 recordings, 9,600 temporal windows and four classical models evaluated on **64 Validation records from 16 participants**. Notebooks 03–05 remain pending on the main cohort; the 20-person main Test is reserved. This study does not predict comprehension or clinical/neurocognitive status.

**Independence:** Reuse the already available **original GazeBase Round 1 / Session 1 CSV files**, but create all Eye-Movement Modelling splits, features, temporal windows, models and outputs independently in `eye_movement_modelling/output/`. No processed Eye-Tracking Foundations result is mixed into the new modelling cohort.

The ten-person pilot remains preserved in earlier Git revisions. **Current `01_*` and `02_*` files belong to the 100-person main cohort**; current `03_*` through `05_*` files still belong to the pilot until those stages are rerun. Main model selection and Test evaluation remain pending. Read the [scientific methods and evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Dataset and Experimental Protocol

| Item | Implemented definition and boundary |
| --- | --- |
| Dataset | [GazeBase v2.0](https://doi.org/10.6084/m9.figshare.12912257), Round **1**, Session **1**, source CSVs from [`data/raw/gazebase/`](../data/README.md) |
| Recorded pilot scope | GazeBase Round 1 / Session 1: **10 participants, 40 source recordings, 40 record feature rows and 960 temporal windows**. Notebook 02 fitted four classical candidates on 24 Train records and evaluated 8 Validation records from 2 independent people; the pilot Test was then observed on 8 recordings from 2 independent people in Notebook 05 |
| **Executed main source cohort** | **100 complete people** (source IDs 1001–1108, excluding 1028, 1036, 1050, 1056, 1075, 1082, 1104 and 1105); **400 records**, **9,600 windows**. Notebook 02 fitted four classical models on 256 Train records and evaluated 64 Validation records; main temporal models remain pending |
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

These five notebooks form one task-recognition study. **Notebooks 01 and 02 are executed for the 100-person main cohort**, and Notebooks 03–05 remain pending. The historical ten-person pilot completed all five stages, including its exploratory Test, while the **20-person main Test is reserved**.

| Notebook | Main experiment and visible model-level evidence | Owned current output files |
| --- | --- | --- |
| [01 — Data and Experimental Protocol](code/notebooks/01_data_and_experimental_protocol.ipynb) | **Executed on 100 people**: 400 records, 9,600 windows, full task audit and independent 64/16/20 person split; nine current stage files packaged | `01_source_inventory.csv`, `01_split_assignments.csv`, `01_record_features.csv`, `01_sequence_windows.npz`, `01_sequence_normalisation.csv`, `01_protocol.csv`, `01_source_distribution.png`; source-audit files `01_source_acquisition_preflight.csv` and `01_source_acquisition_task_summary.csv` |
| [02 — Statistical and Classical ML](code/notebooks/02_statistical_and_classical_ml.ipynb) | **Executed on 100 people:** four fixed candidates fitted on 256 Train records, scored on 64 Validation recordings from 16 people; four fitted pipelines and 14 stage-owned outputs | `02_validation_metrics.csv`, four `02_*.joblib` pipelines, four prediction tables, four confusion plots and comparison plot |
| [03 — CNN and TCN](code/notebooks/03_cnn_and_tcn_models.ipynb) | CNN and TCN trained and evaluated on the same eight Validation recordings from two participants; both 8/8 correct, with epochs and parameter counts recorded | `03_cnn.pt`, `03_tcn.pt`, histories/predictions/plots, `03_neural_validation_metrics.csv` |
| [04 — GRU and Transformer](code/notebooks/04_gru_and_transformer_models.ipynb) | Recorded GRU (2/8 correct) and Transformer (4/8 correct) Validation, with checkpoints, histories and confusion matrices | `04_gru.pt`, `04_transformer.pt`, histories/predictions/plots, `04_neural_validation_metrics.csv` |
| [05 — Robustness and Final Evaluation](code/notebooks/05_robustness_and_final_evaluation.ipynb) | Executed Validation-selected Logistic Regression; 7/8 pilot Test correct, Macro F1 0.867; person-level errors and unchanged Validation F1 after pupil masking | `05_model_selection.csv`, `05_test_predictions.csv`, `05_test_metrics.csv`, `05_groupwise_test_metrics.csv`, `05_robustness_validation.csv`, comparison plots |

A separate editable five-band figure accompanies each notebook in [`images/`](images/). The [overview](images/study_overview.svg) distinguishes completed main-cohort preparation and classical Validation from pending neural stages. Notebook 01's figure shows source preparation; Notebook 02's diagram now reports the observed current Validation results while keeping schematic motifs clearly labelled.

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

**Historical ten-person pilot results (Notebooks 02–05):** These scores describe the earlier pilot, **not the current 100-person classical models**. Current stage-03 through stage-05 model outputs have not been regenerated for the main cohort.

**Observed ten-person pilot:** Notebook 01 processed 40 original recordings into 40 feature rows and 960 temporal windows of shape `(960, 256, 4)`, with 6/2/2 Train/Validation/Test participants (24/8/8 recordings). Notebook 02 evaluated four classical candidates on the same **eight Validation recordings from two people**. The class prior scored **Accuracy 0.25, Macro F1 0.10, Balanced Accuracy 0.25**. Logistic Regression, Random Forest and Gradient Boosting each scored **1.00 on all three metrics** (8/8 correct). These are **exploratory participant-disjoint Validation results**, not independent population estimates, external validation or final Test results.

**Recorded Notebook 03 Validation:** CNN and TCN both classified 8/8 Validation recordings correctly (Accuracy, Macro F1 and Balanced Accuracy all **1.00**). CNN selected epoch 8 with **11,492 trainable parameters**; TCN selected epoch 5 with **12,772**. This is observed notebook evidence from only two independent Validation participants, not an independent reproduction or estimate of population accuracy. Synthetic tests check source identity, grouped splits, train-only normalisation, record-level aggregation and file contracts; they do not establish generalisation.

Notebooks 01–04 established the pilot data contract and eight Validation candidates; Notebook 05 applied the predefined Validation ranking, froze the primary choice and recorded pilot Test metrics. Inspect source step-speed artefacts and stimulus/task confounding before interpreting apparently high scores. The 10-person experiment is a functional pilot, not a new population-level model validation.

**Recorded Notebook 04 Validation:** GRU selected epoch **1** (3,780 trainable parameters) and classified **2/8** records correctly, predicting TEX for all eight (Accuracy 0.25; Macro F1 0.10; Balanced Accuracy 0.25). Transformer selected epoch **2** (33,764 parameters) and correctly classified **4/8**, recognising both HSS and both RAN records but missing FXS and TEX (Accuracy 0.50; Macro F1 0.375; Balanced Accuracy 0.50). The two models used the **same two Validation people** as the other six candidates. These are exploratory pilot observations, not reliable population-level rankings. Notebook 05 subsequently applied the fixed Validation-based selection rule and evaluated the pilot Test.

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

**Notebooks 01 and 02 are complete for their 100-person main-cohort stages.** Their current outputs are committed under `eye_movement_modelling/output/`. The original GazeBase archive and raw CSVs do not need to be downloaded again for Notebook 03 because the current normalised stage-01 windows are already tracked.

1. Open [Notebook 03: CNN and TCN](code/notebooks/03_cnn_and_tcn_models.ipynb) from the current `main` branch in a fresh Colab Runtime.
2. The notebook reads committed `01_sequence_windows.npz`, `01_split_assignments.csv` and `01_protocol.csv`. Upload the matching `eye_movement_modelling_Notebook_01_Outputs.zip` to `/content/` only if deliberately replacing those current files; an explicitly uploaded ZIP takes precedence.
3. Verify **100 participants**, the frozen **64/16/20** person roles and **6,144/1,536/1,920** Train/Validation/reserved Test windows. The existing `03_*` outputs in GitHub still belong to the pilot and must not be used as current main results.
4. Run CNN and TCN sequentially, inspect their Validation metrics, histories, checkpoints and figures, then use the strengthened export cell to check both models against the exact Validation record identities and metric values before packaging `eye_movement_modelling_Notebook_03_Outputs.zip`.
5. Preserve the executed Notebook 03 and its reviewed outputs. Continue with Notebook 04 on the same frozen split; main model selection and the first main Test evaluation belong to Notebook 05.

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

**The ten-person pilot is completed, and Notebooks 01 and 02 have now executed on the 100-person main cohort.** Four current classical Validation results are recorded; current neural Validation and main Test remain pending. Files `03_*` through `05_*` still represent the pilot and must not be mixed with the current classical comparison. High within-GazeBase task-recognition performance does not establish cognitive inference, clinical utility, or generalisation across sessions, stimuli or equipment.

The following remain limitations even after a successful run: task identity is confounded with stimulus protocol; Test participant counts are very small in the pilot; the first complete source IDs are not a representative sample; gaze speeds and pupil fields may reflect measurement artefacts; and a model's Validation/held-out Test score does **not** establish neural, clinical or cognitive diagnosis.

**GazeBase attribution:** [Griffith, Lohr, Abdulin and Komogortsev, *Scientific Data* (2021)](https://doi.org/10.1038/s41597-021-00959-y); [dataset](https://doi.org/10.6084/m9.figshare.12912257), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is separate from the repository's MIT-licensed code.
