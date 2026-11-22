# EEG–Eye-Tracking Multimodal Research

A reproducible six-notebook workflow for evaluating whether EEG and eye movements provide complementary information about **ZuCo 2.0 reading-task conditions**. **Notebooks 01–06 have recorded development executions for the initial six-person cohort**, including 206 matched EEG/gaze feature windows, seven classical and three temporal Validation candidates. Notebook 06 reconciled all ten models, recorded the **Validation-selected gaze logistic regression** and completed Validation-only sensitivity checks. **The reserved Test partition remains unscored.** The objective is to compare unimodal and fusion models on the *same* paired temporal windows from participant-disjoint Train, Validation and Test groups. The target is **normal reading (NR) versus task-specific reading (TSR)**, not comprehension, intelligence, a clinical outcome or cognitive ability.

**Expanded-cohort Stage 01 complete:** The newly executed [source audit](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) covers **18 participants**, one NR and one TSR Run-1 recording each (**36 original runs**). All **36/36 packages** are recorded as present (**3,373.4 MB**); native metadata report **128 EEG channels** and nominal **500-Hz EEG/EyeLink acquisition**. The source manifest, eligibility table, `01_split_assignments.csv` and `01_protocol.json` now agree on **11 Train / 3 Validation / 4 reserved Test people** and retain **all six pilot roles**. The expanded cohort is **source-audited, not yet signal-quality-screened or modelled**; current Stages 02–06 results remain from the original six-person pilot.

![Overview of EEG and eye-tracking data processing and evaluation](images/study_overview.svg)

*Figure 1. Current 18-person source audit and participant split beside historical six-person alignment, feature and model evidence. The workflow figure distinguishes executed versus pending expanded-cohort stages; it does not present 18-person model results.*

## 1. Dataset and Experimental Protocol

The source is **ZuCo 2.0**, NEMAR dataset [`nm000279`, release `v1.0.0`](https://www.nemar.org/dataset/nm000279). Notebook 01 confirmed **18 paired-file-eligible participants**, **36 Run-1 source packages** and a participant-disjoint **11/3/4 Train/Validation/Test allocation**, with all previous pilot roles preserved. The EEG and eye sidecars report **128 EEG channels**, nominal **500-Hz** acquisition and eye `StartTime` offsets of **−0.046 to +0.033 s**. These are source-level checks. The **206 paired 12-second windows** recorded in the existing Notebook-02 files belong to the **earlier six-person pilot**, not the expanded cohort.

Source acquisition reads only selected BIDS participants and NR/TSR Run-1 records through the public NEMAR source gateway. To select a different currently supported cohort size, change **`MAX_PARTICIPANTS`** in Notebook 01; the participant-role helper retains the original six-person assignments. `RUNS_PER_TASK=1` is an approved scientific setting, **not** a routine population-size control. The full original archive is not required in advance, and raw EEG/eye physiology is not committed to GitHub.

The initial design uses **one run per task per eligible person**, 12-second non-overlapping paired windows, six two-second feature bins per window, and a seeded **participant-disjoint** Train/Validation/Test split. Run and task labels are verified from BIDS metadata and are never inferred from EEG amplitudes or gaze patterns. Source eligibility rules and scientific boundaries are described in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

**Important limitations:** identical nominal EEG and gaze sample rates do not prove independent-clock synchronisation. The current mapping uses BIDS `StartTime` and sample frequencies, with drift unverified unless independent markers become available. NR versus TSR differs in instructions and possibly text/stimulus content; performance cannot automatically be attributed to cognitive processing or multimodal complementarity.

## 2. Experimental Programme

**Execution status:** **Stage 01 of the expanded 18-person study is executed and reconciled**, including source eligibility, downloading, acquisition metadata and the frozen pilot-preserving split. The historical **six-person pilot completed all six development-stage notebooks** without final Test scoring (reference: commit `8f2e206`). The current **32 named Stage-02–06 outputs** still describe the pilot; they must be **rerun in order** before reporting any 18-person windows, features, models or validation results.

The **seven current `01_*` outputs** now document the 18-person source cohort. The older **four `02_*`, six `03_*`, eleven `04_*`, eight `05_*` and three `06_*` outputs** concern the six-person pilot (**39 stage-owned files across two separate execution contexts**). A separately committed Notebook-01 ZIP duplicates deliverables and is not an additional research result. The source and downstream output groups must not be analysed as one completed experiment. The pilot `gaze_logistic` selection (Validation Macro F1 **0.8862**) does **not** establish 18-person performance.

| Stage | Notebook | Scientific role | Intended named outputs |
| --- | --- | --- | --- |
| 01 | [ZuCo Source Audit and Protocol](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) | **Expanded source stage complete:** 18 participants, 36/36 source packages, 11/3/4 pilot-preserving participant roles, no Test scoring | Seven reconciled current `01_*` files: manifest, assignments, eligibility, download/sidecar evidence, protocol and source overview |
| 02 | [EEG–Gaze Alignment and Quality](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) | **Expanded run next:** inspect all 36 original runs for timing, events and eye-coordinate window eligibility. Historical pilot: 206 eligible windows from 12 runs | Four pilot `02_*` outputs currently committed; new `02_person_task_quality.csv` expected on expanded execution |
| 03 | [Preprocessing, Features and Neurophysiology](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) | **Executed:** 206 of 206 source windows retain aligned EEG and gaze features; no feature exclusions or non-finite window-mean values; the raw EEG diagnostic is descriptive only | Six committed `03_*` outputs: paired feature CSV, matched six-bin NPZ, quality/exclusion tables and two figures |
| 04 | [Unimodal Baselines and Classical Fusion](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) | **Executed:** seven Validation candidates; gaze logistic Macro F1 **0.8862** (one person) | Eleven recorded `04_*` files: metrics, predictions, fitted pipelines and plots |
| 05 | [Temporal and Advanced Fusion](code/notebooks/05_temporal_and_advanced_fusion.ipynb) | **Executed:** EEG GRU Macro F1 **0.6890** led three temporal Validation models; gated fusion did not improve performance (one person) | Eight recorded `05_*` files: metrics, predictions, checkpoints, scaler, history and plot |
| 06 | [Validation Selection and Robustness](code/notebooks/06_robustness_and_final_evaluation.ipynb) | **Executed in development mode:** reconciled ten models, selected `gaze_logistic` (Validation Macro F1 **0.8862**), assessed fixed-model EEG/gaze masking; **Test unscored** | Three recorded `06_*` outputs: `06_model_selection.csv`, `06_validation_sensitivity.csv`, `06_validation_evidence.png` |

Each stage has an editable five-band scientific SVG in [`images/`](images/). The [Stage-01 figure](images/01_zuco_source_audit_and_protocol.svg) now reports the **verified 18-person source audit and frozen roles**; the [study overview](images/study_overview.svg) explicitly separates this from the historical six-person downstream workflow. Figures for Stages 02–06 will be reconciled after the corresponding expanded-cohort notebooks are executed.

## 3. Main Results

*Only Notebook-01 source/aquisition numbers and split identities below concern the **new 18-person cohort**. Stages 02–06 still contain **historical six-person pilot** results; no 18-person window count, feature extraction, model fit or Validation/Test score is claimed.*

**Executed expanded source audit (Notebook 01):** All **18 indexed participants** had complete original BIDS filename sets for one NR and one TSR Run-1 recording. The **36/36 packages** totalled **3,373.4 MB (3.14 GiB)**. Source sidecars report **128 EEG channels**, nominal **500 Hz** for EEG and EyeLink, and eye-start offsets from **−0.046 to +0.033 seconds**. The frozen partition has **11 Train, 3 Validation and 4 reserved Test people**; six previously assessed pilot roles remain unchanged. The CSV/JSON provenance and identities agree. Source presence does not establish final usable EEG–gaze windows, independent clock-drift verification or model performance.

The relevant comparison is whether a fusion model improves over the strongest **EEG-only or gaze-only** comparator on *identical*, participant-disjoint examples. A negative or mixed fusion result is scientifically informative. Primary model selection is based on Validation only and must **not** be revised using future Test scores.

**Recorded paired-window audit (Notebook 02):** **206/206** candidate 12-second windows met the finite x/y criterion (`eye_valid_fraction ≥ 0.60`), comprising **101 NR / 105 TSR** windows. **Train has 95 windows from three people; Validation 31 from one; Test 80 from two.** The short `sub-YAK / NR` Run contributed **one** window; the largest contributing runs were capped at 20 windows each. All `drift_verified` flags are `False`. This confirms a BIDS-metadata-supported paired-window registry, **not** independently measured clock synchrony, eye-tracker calibration, or model performance.

| Partition | NR windows | TSR windows | Total windows | Independent people |
| --- | ---: | ---: | ---: | ---: |
| Train | 41 | 54 | 95 | 3 |
| Validation | 20 | 11 | 31 | 1 |
| Test (reserved) | 40 | 40 | 80 | 2 |
| **Overall** | **101** | **105** | **206** | **6** |

**Recorded feature extraction (Notebook 03):** The complete **206-window** Stage-02 registry was processed, producing **206 rows with four EEG and four gaze mean features** and two matching temporal arrays, each **206 × 6 × 4**. **Zero feature-extraction failures**, **zero non-finite window-mean values**, and all 12 participant/task groups were retained. The original split remains **95 Train / 31 Validation / 80 reserved Test windows**, comprising **three / one / two independent people**. The recorded gaze-valid fraction is constant at **1.000** in these 206 examples; it does not independently validate gaze calibration or artefact quality.

The exploratory fixation-locked raw EEG plot has a large uncorrected baseline offset (approximately **−10,100 µV**). It is **not a validated fixation-related potential or filtered EEG feature plot** and must not support physiological conclusions.

**Recorded classical Validation evaluation (Notebook 04):** Seven fixed candidates were fitted on **95 Train windows from three participants** and compared on the **same 31 Validation windows from one participant** (20 NR / 11 TSR). The `04_validation_predictions.csv` file contains **217 prediction rows**: 31 per model, with matching source IDs, labels and valid probability pairs. Scores recomputed from the saved predictions agree with `04_validation_metrics.csv`.

| Classical candidate | Validation Macro F1 | Balanced Accuracy |
| --- | ---: | ---: |
| Gaze logistic regression | **0.8862** | **0.8636** |
| Fixed equal-weight late fusion | 0.7786 | 0.7682 |
| Gaze random forest | 0.6601 | 0.6682 |
| Early-fusion logistic regression | 0.5620 | 0.5909 |
| EEG random forest | 0.2955 | 0.2955 |
| Class prior | 0.2619 | 0.5000 |
| EEG logistic regression | 0.1622 | 0.2727 |

Gaze logistic regression classified **20/20 NR** and **8/11 TSR** Validation windows correctly (Accuracy **0.9032**). Both classical fusion approaches scored below this strongest unimodal comparator in this participant. This is an **exploratory single-participant Validation observation**, not evidence that gaze-only modelling generally outperforms fusion.

**Recorded temporal Validation evaluation (Notebook 05):** Three fixed GRU candidates were trained on **95 Train windows from three people** and scored on the **same 31 Validation windows from one person**. The saved `05_validation_predictions.csv` has **93 rows** and `05_training_history.csv` has **43 epoch records**. Recomputed Validation metrics match the saved three-candidate table.

| Temporal candidate | Validation Macro F1 | Balanced Accuracy | Observed class predictions |
| --- | ---: | ---: | --- |
| EEG GRU | **0.6890** | **0.6932** | 15/20 NR and 7/11 TSR correct |
| Gaze GRU | 0.3922 | 0.5000 | All 31 predicted NR |
| Gated fusion GRU | 0.2619 | 0.5000 | All 31 predicted TSR |

The EEG GRU led the temporal candidates; **gated fusion did not outperform either unimodal GRU**. Comparing all ten saved candidates, gaze logistic regression from Notebook 04 still has the highest observed Validation Macro F1 (**0.8862**), ahead of EEG GRU (**0.6890**). All scores refer to **one independent Validation participant**, and no Test performance or cross-person uncertainty estimate follows from them. The **80 Test windows remain unscored**. Notebook 06 has since **reconciled these results and saved the prespecified Validation selection**; no final Test evaluation was undertaken.


**Recorded Validation selection and robustness (Notebook 06):** All **310 prediction rows** across ten candidates were reconciled on the same **31 Validation windows from one independent person**. The selected candidate was **`gaze_logistic`** (Macro F1 **0.8862**, Balanced Accuracy **0.8636**, Accuracy **0.9032**). The `06_model_selection.csv` records `run_mode=development`, `validation_n_people=1` and `test_used_in_selection=False`.

A **fixed early-fusion logistic estimator** was subjected to descriptive imputation-based feature masking, using its existing Train-fitted imputer without model refitting:

| Input condition | Validation Macro F1 | Balanced Accuracy |
| --- | ---: | ---: |
| EEG and gaze unchanged | 0.5620 | 0.5909 |
| EEG masked | **0.8862** | **0.8636** |
| Gaze masked | 0.4833 | 0.5182 |

These perturbations describe the behaviour of **one unchanged early-fusion model on one Validation person**. They are **not** causal evidence about the EEG or gaze modalities, and the perturbed outputs did not determine the primary-model choice.

![Recorded ten-model Validation ranking and fixed early-fusion masking sensitivity](output/06_validation_evidence.png)

*Figure 2. Executed Notebook-06 development evidence: ranking of ten candidates and descriptive early-fusion input masking on the same single Validation participant. No Test score is depicted.*

**Final-evaluation status:** The **six-person pilot** completed Validation-only comparison and selection but **never scored Test**. The newly expanded study has only completed source audit and frozen role assignment. No expanded 18-person Test inference or Validation estimate is available. The optional final evaluation is a separate decision after the expanded development protocol and results have been reviewed.

## 4. Running and Reviewing the Study

**Next execution: Notebook 02.** The **18-person Stage-01 audit is complete** and its seven current named outputs are committed. Open the latest [Notebook 02](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) in a **fresh Colab session**; its Setup and Imports now follow the approved, short repository checkout/`setup_colab("eeg_eye_tracking_fusion")` template. Run from the first cell, allowing it to obtain the current Stage-01 files. Before processing, it checks that the manifest, participant roles and `selected_people=18` agree.

Notebook 02 then inspects the **36 original NR/TSR Run-1 records** for temporal overlap, source events, finite gaze-coordinate coverage and **12-second paired-window** eligibility (maximum 20 per run; minimum 0.60 finite eye fraction). It is expected to overwrite its **own** `02_*` outputs and add `02_person_task_quality.csv`; it must not rewrite Stage-01 files or silently exclude a participant. If any person/task has no usable window, stop and review the recorded quality evidence before proceeding. Download, review and commit the executed Notebook-02 artefacts before running Notebook 03.

Run expanded Notebooks **03, 04, 05 and 06 in order** only after their upstream dependencies have been checked and published. Each stage overwrites its current named outputs, retains participant-disjoint roles and fits any data-derived transformations on Train only. **Keep `RUN_FINAL_TEST = False`** throughout development. Historical six-person metrics are comparators, not outputs of the 18-person rerun.

The raw data path is not persistent between Colab sessions. Stage 02/03 source retrieval is intentionally limited to the selected runs rather than the full 18-GB release. GPU use is optional for the compact temporal models; free Colab resource availability is not guaranteed. The original source's attribution and use conditions remain applicable.

## 5. Files and Saved Outputs

**Additional stage-owned outputs on the next run:** `02_person_task_quality.csv` describes each original run's usable window and finite-coordinate support; `03_feature_quality.csv` describes Train feature missingness and constant predictors; `04_validation_person_metrics.csv`, `05_validation_person_metrics.csv`, and `06_validation_person_metrics.csv` report class support and fixed-label metrics per independent Validation participant. They are **not yet present** and are not required to interpret the recorded six-person pilot. The frozen model-selection rule continues to use pooled-window Validation Macro F1; person-level summaries are complementary rather than a retroactive selection change.

```text
eeg_eye_tracking_fusion/
├── README.md
├── RESEARCH_METHODS_AND_EVIDENCE.md
├── requirements.txt
├── code/
│   ├── notebooks/              # Six ordered Colab notebooks
│   └── src/                    # Source, alignment, features and model helpers
├── images/                     # Overview and six role-based SVG figures
├── output/                     # Stage-01–06 development evidence files
└── data/                       # Source cache is external to Git by design
```

Notebook outputs in [`output/`](output/) have distinct `01_*` through `06_*` prefixes. **Seven new Stage-01 files** contain the verified 18-person source audit and frozen roles; **32 older Stage-02–06 files** are historical six-person pilot results (**39 stage-owned outputs**). A Notebook-01 ZIP has also been committed, but duplicates those stage artefacts and is not a separate finding. Preserve this cross-stage provenance boundary until each dependent expanded stage is rerun and reviewed. No final held-out Test outputs are present.

## 6. Interpretation and Scope

This project studies original reading-task labels from paired EEG and eye movements. A classifier may exploit task instructions, source texts, session conditions, eye movements and other nuisance differences; high task discrimination **does not directly demonstrate language comprehension, neurocognitive state, clinical validity or causal EEG contributions**.

Temporal EEG features derive from a fixed 16-channel subset without anatomical localisation claims. The published exploratory raw fixation-locked trace has a large uncorrected offset and does not substitute for a referenced/filtered, artefact-controlled ERP/FRP analysis. Participant-disjoint validation controls person overlap, **not cross-session or stimulus confounding**. Independent timing drift and external generalisation remain unverified unless explicitly assessed.

**Data and attribution:** [ZuCo 2.0 (Hollenstein et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.18/), [NEMAR release nm000279](https://doi.org/10.82901/nemar.nm000279), licence reported in the BIDS metadata as CC BY 4.0. Code and methodological outputs in this repository are separate from the original source recordings.
