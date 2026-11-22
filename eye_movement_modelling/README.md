# Eye-Movement Modelling & Advanced AI | GazeBase v2.0

![Eye-Movement Modelling gaze-task modelling workflow, source provenance, model families and reserved evaluation](images/study_overview.svg)

*Figure 1. Scientific programme for independent source preparation and classical/temporal modelling. The ten-person GazeBase pilot now has recorded source preparation (Notebook 01), four classical Validation comparisons (Notebook 02), and CNN/TCN Validation results (Notebook 03); GRU and Transformer Validation is also recorded (Notebook 04); the pilot Test has also been executed (Notebook 05). The small measured source excerpt is taken from an original GazeBase CSV, while conceptual model/workflow elements are schematic.*

**Aim:** Compare record-level engineered gaze features and four compact temporal neural architectures for the **four-class GazeBase recording-protocol target** (FXS, HSS, RAN, TEX). **All five notebooks (01–05) have recorded ten-person pilot execution**, including Validation-based model selection, participant-disjoint pilot Test and descriptive robustness. The **100-person main-study code is now configured for execution, but has not been run**. A separately frozen run may select up to 322 Round-1 people, subject to source eligibility and Colab resources. Neither reading comprehension nor clinical/neurocognitive status is predicted.

**Independence:** Reuse the already available **original GazeBase Round 1 / Session 1 CSV files**, but create all Eye-Movement Modelling splits, features, temporal windows, models and outputs independently in `eye_movement_modelling/output/`. No processed Eye-Tracking Foundations result is mixed into the new modelling cohort.

The executed modelling run remains a **ten-person pilot**, although source acquisition verified original recordings for **20 participants (80 task records)**. The updated Notebook 01 defaults to **100 people**, with adjustable count and declared split proportions, but **no new-cohort outcomes exist**. The main Test subset must remain reserved until the new Validation-based selection is frozen. Read the [full methods and evidence plan](RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Dataset and Experimental Protocol

| Item | Implemented definition and boundary |
| --- | --- |
| Dataset | [GazeBase v2.0](https://doi.org/10.6084/m9.figshare.12912257), Round **1**, Session **1**, source CSVs from [`data/raw/gazebase/`](../data/README.md) |
| Recorded pilot scope | GazeBase Round 1 / Session 1: **10 participants, 40 source recordings, 40 record feature rows and 960 temporal windows**. Notebook 02 fitted four classical candidates on 24 Train records and evaluated 8 Validation records from 2 independent people; the pilot Test was then observed on 8 recordings from 2 independent people in Notebook 05 |
| Configured next scope (unexecuted) | First 100 Round-1/Session-1 people, four task files each; Train 64, Validation 16, Test 20 intended. Change one Notebook 01 parameter for a separate 300-person run (192/48/60), with source availability checked on each run |
| Prediction target | Recording protocol: **FXS**, **HSS**, **RAN**, **TEX**. Four known classes; **not cognitive diagnosis** |
| Predictors: classical | Recording validity, x/y dispersion, step displacement, velocity quantiles, positive pupil coverage/summary where available |
| Predictors: temporal | Fixed 256-sample windows; source-valid contiguous segments only; relative x/y and their first differences (four channels) |
| Window sampling | Up to 24 evenly distributed, **non-overlapping** windows per source record; all windows inherit the parent record Task |
| Group separation | All task records/windows for one person stay in exactly one Train, Validation or Test partition |
| Pilot (10 people) | Seed **17**, participant-disjoint Train/Validation/Test; Test is **exploratory pilot diagnostic**, not definitive unseen-person evidence |
| Main cohort split | Seed 17; highest 20% source IDs reserved for Test, 20% of the remaining development participants for Validation, remainder Train; **not yet executed on 100 people** |
| Train-only operations | Temporal channel normalisation and classical feature imputation/scaling fitted only on Training people |
| Selection and evaluation | Validation record-level Macro F1, then Balanced Accuracy, then fixed model priority. Frozen Test opened in Notebook 05 only |
| Source limitation | Different task stimuli and recording protocols may dominate task recognition; high task accuracy would not imply cognition |

**Do not change only the cohort-size setting after inspecting the main Test and present the new Test as untouched evidence.** Decide the final cohort before the main evaluation; rerun the full producer-to-consumer chain whenever it changes.

## 2. Experimental Programme

These five notebooks form one task-recognition experiment. **All five notebooks have recorded original-data pilot outputs.** The Test was opened only for the ten-person pilot; no 100-person main experiment has been conducted.

| Notebook | Main experiment and visible model-level evidence | Owned current output files |
| --- | --- | --- |
| [01 — Data and Experimental Protocol](code/notebooks/01_data_and_experimental_protocol.ipynb) | Integrated Section 02 source acquisition/audit for 20 participants and current ten-person pilot preparation; nine required files bundled in one ZIP | `01_source_inventory.csv`, `01_split_assignments.csv`, `01_record_features.csv`, `01_sequence_windows.npz`, `01_sequence_normalisation.csv`, `01_protocol.csv`, `01_source_distribution.png`; source-audit files `01_source_acquisition_preflight.csv` and `01_source_acquisition_task_summary.csv` |
| [02 — Statistical and Classical ML](code/notebooks/02_statistical_and_classical_ml.ipynb) | Separate sections for class prior, Logistic Regression, Random Forest and Gradient Boosting; each with Validation confusion matrix | `02_validation_metrics.csv`, four fitted `02_*.joblib` files, independent predictions/plots, comparison |
| [03 — CNN and TCN](code/notebooks/03_cnn_and_tcn_models.ipynb) | CNN and TCN trained and evaluated on the same eight Validation recordings from two participants; both 8/8 correct, with epochs and parameter counts recorded | `03_cnn.pt`, `03_tcn.pt`, histories/predictions/plots, `03_neural_validation_metrics.csv` |
| [04 — GRU and Transformer](code/notebooks/04_gru_and_transformer_models.ipynb) | Recorded GRU (2/8 correct) and Transformer (4/8 correct) Validation, with checkpoints, histories and confusion matrices | `04_gru.pt`, `04_transformer.pt`, histories/predictions/plots, `04_neural_validation_metrics.csv` |
| [05 — Robustness and Final Evaluation](code/notebooks/05_robustness_and_final_evaluation.ipynb) | Executed Validation-selected Logistic Regression; 7/8 pilot Test correct, Macro F1 0.867; person-level errors and unchanged Validation F1 after pupil masking | `05_model_selection.csv`, `05_test_predictions.csv`, `05_test_metrics.csv`, `05_groupwise_test_metrics.csv`, `05_robustness_validation.csv`, comparison plots |

A separate editable figure accompanies each notebook in [`images/`](images/), matching the approved five-band project style. The [overall diagram](images/study_overview.svg) distinguishes executed preparation and Validation results for all eight candidates from the now-recorded exploratory pilot Test; the proposed main experiment remains pending. The existing five-band figures show **historical ten-person pilot evidence**; the revised notebooks intentionally contain no copied 100-person result numbers. Each new run creates its own source and model figures.

## 3. Main Results

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

**Next experiment (ready to run, not executed):** Set the single
`MAX_PARTICIPANTS = 100` in Notebook 01, Section 02. For a separate
300-person study, change it to `300` **before any new main Test results**.
The source boundary is GazeBase v2.0, Round 1 / Session 1, FXS/HSS/RAN/TEX;
deterministic source IDs start at 1001, with four complete files required
per person. Published Round 1 contains 322 people, but every selected file
must be checked directly. The original 6,710,984,652-byte archive is
downloaded and MD5-verified in Colab only if source CSVs are missing;
only the chosen original CSVs are extracted. The temporary ZIP is removed
after extraction. **No Google Drive is required.**

With seed 17 and 20% Test/20% of the development pool for Validation,
the intended **100-person split is 64 Train, 16 Validation and 20 Test**.
For 300, it is **192 / 48 / 60**. The highest source IDs form Test and
are reserved until Notebook 05. Original task identity, not cognition,
remains the target.

Notebook 01 exports nine named derived files as
`/content/eye_movement_modelling_Notebook_01_Outputs.zip`, without
the original source archive or its raw CSVs. In a new Colab Runtime,
upload the matching stage ZIP to `/content/` before running a downstream
notebook. The new study-specific `ensure_s2_stage_inputs` helper treats
an **explicitly uploaded ZIP as authoritative over stale pilot files**
checked out from GitHub. `verify_main_cohort` then rejects incomplete
or mismatched participant registries. Without an uploaded ZIP, existing
named outputs are reused.

The ten-person pilot is preserved in earlier Git commits and existing
committed `output/` files. Newly authored notebook code cells were reset
to unexecuted for the proposed main study, avoiding false 100-person
results. Each producing notebook overwrites its own current `NN_*`
outputs on rerun. Download the ZIP from each stage before the Colab
Runtime ends; files above **50,000,000 bytes** are excluded from ordinary
Git commits (including large NPZs in a 300-person run). The ZIP itself is
a manual handoff, not a raw-data backup.


1. Open [Notebook 01](code/notebooks/01_data_and_experimental_protocol.ipynb) in a **fresh Colab Runtime**, verify the single cohort configuration equals 100, and execute all cells through stage-01 ZIP export.
2. **New main cohort:** leave `MAX_PARTICIPANTS = 100` for this run. Check all selected people, source recordings, participant role counts and data quality before continuing.
3. Run Notebook 01 fully and inspect the recorded IDs, split groups, feature coverage, temporal input shape, source diagram and generated output ZIP. Only then proceed to [Notebook 02](code/notebooks/02_statistical_and_classical_ml.ipynb), then [03](code/notebooks/03_cnn_and_tcn_models.ipynb), [04](code/notebooks/04_gru_and_transformer_models.ipynb) and [05](code/notebooks/05_robustness_and_final_evaluation.ipynb).
4. Subsequent stages restore **matching current-stage ZIPs** from `/content/` through `ensure_s2_stage_inputs`, or read existing outputs when none is supplied. The main-cohort check rejects stale pilot data; reading a report never retrains a model.
5. **Later 300-person experiment:** freeze a new cohort and Test plan first, regenerate dependent outputs and checkpoints, and declare any adaptation informed by the 100-person Test. Do not treat a newly enlarged experiment as an untouched continuation.
6. Reviewed derivative files under `eye_movement_modelling/output/` can now be committed after checking the independent **50,000,000-byte per-file limit**. Stage with `git add eye_movement_modelling/output/`, run `python scripts/check_git_file_sizes.py --staged`, and unstage oversized files using `git restore --staged <path>`. ZIP files generated under `/content/` are not staged automatically. Colab must have `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch` and `joblib` installed as listed in [`requirements.txt`](../requirements.txt).

**Runtime:** Small models, a maximum of eight epochs and a bound of 24 windows per record are initial resource choices, not measured speed claims. Notebook 05 reads saved checkpoints; it does not retrain them. If new data change the source selection, old model checkpoints must not be reused.

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

**Pilot completed; 100-person main implementation prepared, not executed.** Pilot results remain available in prior commits and current tracked outputs. Main-study notebooks contain no prefilled 100-person output cells. The observed Test contains only two people. High task-recognition performance is not evidence of cognitive understanding or generalisation to new participants, sessions, stimuli or hardware.

The following remain limitations even after a successful run: task identity is confounded with stimulus protocol; Test participant counts are very small in the pilot; the first complete source IDs are not a representative sample; gaze speeds and pupil fields may reflect measurement artefacts; and a model's Validation/held-out Test score does **not** establish neural, clinical or cognitive diagnosis.

**GazeBase attribution:** [Griffith, Lohr, Abdulin and Komogortsev, *Scientific Data* (2021)](https://doi.org/10.1038/s41597-021-00959-y); [dataset](https://doi.org/10.6084/m9.figshare.12912257), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is separate from the repository's MIT-licensed code.
