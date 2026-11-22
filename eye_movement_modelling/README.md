# Eye-Movement Modelling & Advanced AI | GazeBase v2.0

![Eye-Movement Modelling gaze-task modelling workflow, source provenance, model families and reserved evaluation](images/study_overview.svg)

*Figure 1. Scientific programme for independent source preparation and classical/temporal modelling. The ten-person GazeBase pilot now has recorded source preparation (Notebook 01), four classical Validation comparisons (Notebook 02), and CNN/TCN Validation results (Notebook 03); GRU and Transformer Validation is also recorded (Notebook 04); held-out Test results remain pending. The small measured source excerpt is taken from an original GazeBase CSV, while conceptual model/workflow elements are schematic.*

**Aim:** Compare record-level engineered gaze features and four compact temporal neural architectures for the **four-class GazeBase recording-protocol target** (FXS, HSS, RAN, TEX). **Notebooks 01–04 have recorded pilot execution**; the held-out pilot Test in Notebook 05 remains pending. Neither reading comprehension nor clinical/neurocognitive status is predicted.

**Independence:** Reuse the already available **original GazeBase Round 1 / Session 1 CSV files**, but create all Eye-Movement Modelling splits, features, temporal windows, models and outputs independently in `eye_movement_modelling/output/`. No processed Eye-Tracking Foundations result is mixed into the new modelling cohort.

The executed modelling run remains a **ten-person pilot**, although source acquisition verified original recordings for **20 participants (80 task records)**. A **100-person study is proposed but not yet executed**; its cohort, group-aware partitions and Test protocol must be fixed before the main experiment and all dependent notebooks rerun. Read the [full methods and evidence plan](RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Dataset and Experimental Protocol

| Item | Implemented definition and boundary |
| --- | --- |
| Dataset | [GazeBase v2.0](https://doi.org/10.6084/m9.figshare.12912257), Round **1**, Session **1**, source CSVs from [`data/raw/gazebase/`](../data/README.md) |
| Recorded pilot scope | GazeBase Round 1 / Session 1: **10 participants, 40 source recordings, 40 record feature rows and 960 temporal windows**. Notebook 02 fitted four classical candidates on 24 Train records and evaluated 8 Validation records from 2 independent people; no Test model scores have been observed |
| Prospective extended scope | A 100-person study is proposed, not implemented. The original source audit covers 20 people; the currently evaluated modelling cohort contains 10 |
| Prediction target | Recording protocol: **FXS**, **HSS**, **RAN**, **TEX**. Four known classes; **not cognitive diagnosis** |
| Predictors: classical | Recording validity, x/y dispersion, step displacement, velocity quantiles, positive pupil coverage/summary where available |
| Predictors: temporal | Fixed 256-sample windows; source-valid contiguous segments only; relative x/y and their first differences (four channels) |
| Window sampling | Up to 24 evenly distributed, **non-overlapping** windows per source record; all windows inherit the parent record Task |
| Group separation | All task records/windows for one person stay in exactly one Train, Validation or Test partition |
| Pilot (10 people) | Seed **17**, participant-disjoint Train/Validation/Test; Test is **exploratory pilot diagnostic**, not definitive unseen-person evidence |
| Code behaviour at ≥20 people | The current split helper reserves the highest source IDs for Test, but the main 100-person cohort and evaluation design have not been approved or executed |
| Train-only operations | Temporal channel normalisation and classical feature imputation/scaling fitted only on Training people |
| Selection and evaluation | Validation record-level Macro F1, then Balanced Accuracy, then fixed model priority. Frozen Test opened in Notebook 05 only |
| Source limitation | Different task stimuli and recording protocols may dominate task recognition; high task accuracy would not imply cognition |

**Do not change only the cohort-size setting after inspecting the main Test and present the new Test as untouched evidence.** Decide the final cohort before the main evaluation; rerun the full producer-to-consumer chain whenever it changes.

## 2. Experimental Programme

These five notebooks form one task-recognition experiment. **Notebooks 01–04 have recorded original-data outputs.** Notebook 05 is implemented but has not yet run on the held-out pilot Test.

| Notebook | Main experiment and visible model-level evidence | Owned current output files |
| --- | --- | --- |
| [01 — Data and Experimental Protocol](code/notebooks/01_data_and_experimental_protocol.ipynb) | Integrated Section 02 source acquisition/audit for 20 participants and current ten-person pilot preparation; nine required files bundled in one ZIP | `01_source_inventory.csv`, `01_split_assignments.csv`, `01_record_features.csv`, `01_sequence_windows.npz`, `01_sequence_normalisation.csv`, `01_protocol.csv`, `01_source_distribution.png`; source-audit files `01_source_acquisition_preflight.csv` and `01_source_acquisition_task_summary.csv` |
| [02 — Statistical and Classical ML](code/notebooks/02_statistical_and_classical_ml.ipynb) | Separate sections for class prior, Logistic Regression, Random Forest and Gradient Boosting; each with Validation confusion matrix | `02_validation_metrics.csv`, four fitted `02_*.joblib` files, independent predictions/plots, comparison |
| [03 — CNN and TCN](code/notebooks/03_cnn_and_tcn_models.ipynb) | CNN and TCN trained and evaluated on the same eight Validation recordings from two participants; both 8/8 correct, with epochs and parameter counts recorded | `03_cnn.pt`, `03_tcn.pt`, histories/predictions/plots, `03_neural_validation_metrics.csv` |
| [04 — GRU and Transformer](code/notebooks/04_gru_and_transformer_models.ipynb) | Recorded GRU (2/8 correct) and Transformer (4/8 correct) Validation, with checkpoints, histories and confusion matrices | `04_gru.pt`, `04_transformer.pt`, histories/predictions/plots, `04_neural_validation_metrics.csv` |
| [05 — Robustness and Final Evaluation](code/notebooks/05_robustness_and_final_evaluation.ipynb) | Validation-selected model, frozen Test evaluation, individual-person errors, Validation-only input perturbation and comparisons | `05_model_selection.csv`, `05_test_predictions.csv`, `05_test_metrics.csv`, `05_groupwise_test_metrics.csv`, `05_robustness_validation.csv`, comparison plots |

A separate editable figure accompanies each notebook in [`images/`](images/), matching the approved five-band project style. The [overall diagram](images/study_overview.svg) distinguishes executed preparation and Validation results for all eight candidates from the pending held-out Test. The notebooks display separate scientific results for every candidate, rather than combining all neural models into one cell.

## 3. Main Results

**Observed ten-person pilot:** Notebook 01 processed 40 original recordings into 40 feature rows and 960 temporal windows of shape `(960, 256, 4)`, with 6/2/2 Train/Validation/Test participants (24/8/8 recordings). Notebook 02 evaluated four classical candidates on the same **eight Validation recordings from two people**. The class prior scored **Accuracy 0.25, Macro F1 0.10, Balanced Accuracy 0.25**. Logistic Regression, Random Forest and Gradient Boosting each scored **1.00 on all three metrics** (8/8 correct). These are **exploratory participant-disjoint Validation results**, not independent population estimates, external validation or final Test results.

**Recorded Notebook 03 Validation:** CNN and TCN both classified 8/8 Validation recordings correctly (Accuracy, Macro F1 and Balanced Accuracy all **1.00**). CNN selected epoch 8 with **11,492 trainable parameters**; TCN selected epoch 5 with **12,772**. This is observed notebook evidence from only two independent Validation participants, not an independent reproduction or estimate of population accuracy. Synthetic tests check source identity, grouped splits, train-only normalisation, record-level aggregation and file contracts; they do not establish generalisation.

The recorded Notebook 01 and 02 outputs establish the pilot data contract and classical Validation evidence. Notebook 02 saves four fitted models, predictions, confusion matrices and a comparative table, but **not a final model choice**. Inspect high source step speeds and the small-sample/stimulus confounding before interpreting perfect Validation scores. GRU and Transformer are recorded as executed in Notebook 04; the frozen Test in Notebook 05 remains unevaluated.

**Recorded Notebook 04 Validation:** GRU selected epoch **1** (3,780 trainable parameters) and classified **2/8** records correctly, predicting TEX for all eight (Accuracy 0.25; Macro F1 0.10; Balanced Accuracy 0.25). Transformer selected epoch **2** (33,764 parameters) and correctly classified **4/8**, recognising both HSS and both RAN records but missing FXS and TEX (Accuracy 0.50; Macro F1 0.375; Balanced Accuracy 0.50). The two models used the **same two Validation people** as the other six candidates. These are exploratory pilot observations, not reliable population-level rankings. Final model selection remains a separate, pre-Test operation in Notebook 05.

## 4. Running and Reviewing the Study

**Notebook 01: Colab-only acquisition, preparation and one ZIP.**
The existing numbered **Section 02** now checks original GazeBase v2.0
Round 1 / Session 1 source IDs 1001–1020 (four recording tasks per person).
Files already present in the local checkout are reused. Missing files are
obtained from the published dataset and selectively extracted under the
Git-ignored `data/raw/gazebase/` directory. The source archive is
approximately 6.7 GB and may need downloading again in a new Colab
Runtime. No Google Drive connection or automatic raw-data upload to
GitHub is needed.

The source audit produces `01_source_acquisition_preflight.csv` and
`01_source_acquisition_task_summary.csv`. Notebook 01 then applies its
existing **ten-person development-pilot** configuration
(`MAX_PARTICIPANTS = 10`), saving the source inventory, independent
participant split, recording features, temporal windows, Train-only
normalisation, protocol and source-distribution figure. Checking original
files from 20 participants is **not** a 20-person model evaluation.

**Run Notebook 01 completely, from Section 00 through Section 07.**
Its final cell verifies that all nine current stage-01 files exist
before using the original `download_outputs_from_colab` helper to
produce a single ZIP:
`/content/eye_movement_modelling_Notebook_01_Outputs.zip`.
This ZIP contains all derived data needed by the current downstream
notebook inputs, including both source-audit CSVs. It does not contain the
original gaze CSVs or 6.7-GB archive.

Notebook 02 consumes `01_record_features.csv`,
`01_split_assignments.csv` and `01_protocol.csv`. Notebooks 03 and 04
consume `01_sequence_windows.npz` and the same matching split/protocol.
Notebook 05 additionally needs model artefacts, Validation predictions
and metrics saved by Notebooks 02–04.

If the same Colab Runtime remains active, the next notebook reads the
existing files directly. In a **new** Runtime, upload the corresponding
downloaded ZIP to `/content/`; its `ensure_stage_inputs` helper
restores the named data. Download the stage-02, stage-03 and stage-04
ZIPs in turn before leaving their Runtimes. Never mix stage files
generated from different participants, splits or model versions.

The latest Notebook 01 output records a complete original-archive download
using the official Figshare endpoint, a successful source size/MD5 check,
and quality audits for **20 people and 80 recordings**. These are
notebook-recorded observations, not an independent reproduction of the
source transfer. The modelling and classical Validation results remain
from the **ten-person pilot**. A subsequent
100-person experiment needs its own fixed evaluation protocol and
regenerated producer/consumer outputs.


1. Open [Notebook 01](code/notebooks/01_data_and_experimental_protocol.ipynb) from GitHub in a fresh Colab Runtime; execute setup/imports followed by all scientific sections, including source acquisition in Section 02, through the final ZIP export.
2. **Current pilot:** leave `MAX_PARTICIPANTS = 10` for the recorded modelling method. Section 02 can acquire and audit 20 source participants without changing the scientific cohort. Download the single current Notebook 01 ZIP before leaving Colab.
3. Run Notebook 01 fully and inspect the recorded IDs, split groups, feature coverage, temporal input shape, source diagram and generated output ZIP. Only then proceed to [Notebook 02](code/notebooks/02_statistical_and_classical_ml.ipynb), then [03](code/notebooks/03_cnn_and_tcn_models.ipynb), [04](code/notebooks/04_gru_and_transformer_models.ipynb) and [05](code/notebooks/05_robustness_and_final_evaluation.ipynb).
4. Each stage reads **explicit named current outputs** and can restore a matching, manually transferred ZIP through the existing Colab `ensure_stage_inputs` helper. Training is never a hidden side effect of opening a report.
5. **Later main experiment:** plan the proposed 100-person cohort and participant-disjoint evaluation before changing the scientific participant count. Recreate compatible outputs in each stage; do not use Test results to revise model selection or the cohort size.
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

**Partially executed pilot.** All eight candidates have recorded Validation results; final Validation-based model selection and the participant-disjoint held-out Test in Notebook 05 remain outstanding. The Test has only two people in this pilot. High task-recognition performance is not evidence of cognitive understanding or generalisation to new participants, sessions, stimuli or hardware.

The following remain limitations even after a successful run: task identity is confounded with stimulus protocol; Test participant counts are very small in the pilot; the first complete source IDs are not a representative sample; gaze speeds and pupil fields may reflect measurement artefacts; and a model's Validation/held-out Test score does **not** establish neural, clinical or cognitive diagnosis.

**GazeBase attribution:** [Griffith, Lohr, Abdulin and Komogortsev, *Scientific Data* (2021)](https://doi.org/10.1038/s41597-021-00959-y); [dataset](https://doi.org/10.6084/m9.figshare.12912257), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is separate from the repository's MIT-licensed code.
