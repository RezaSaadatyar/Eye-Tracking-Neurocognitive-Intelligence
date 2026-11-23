# EEG–Eye-Tracking Multimodal Research

A reproducible, six-notebook workflow investigating whether EEG and eye movements provide complementary information for distinguishing **Normal Reading (NR) from Task-Specific Reading (TSR)** in ZuCo 2.0. The **18-person expanded study has completed and published Stages 01–03**, including source auditing, paired-window eligibility and EEG/gaze feature extraction. All **630 paired 12-second windows** are retained, with **371 Train, 99 Validation and 160 reserved Test** windows from **11/3/4 independent people**. Stages **04–06 have not yet been executed on the expanded cohort**. A historical six-person pilot completed the seven classical and three temporal model comparisons; those results are **not estimates for this cohort**. The reserved Test remains unscored.

**Current expanded evidence:** Stage 01 audited **18 paired-file-eligible participants** and **36 NR/TSR Run-1 packages** (3,373.4 MB), with all six pilot roles preserved. Stage 02 generated and published **630/630 eligible non-overlapping 12-second windows**, using BIDS-relative EEG/eye timing and a finite-coordinate threshold of 0.60. Stage 03 processed **all 630 windows**, yielding **eight finite window-mean EEG/gaze features**, no exclusions, **two temporal arrays of shape (630, 6, 4)** and the new Train-derived feature-quality report. `gaze_valid_fraction` is constant at 1.0 in the present cohort. The first three stages are **recorded and reconciled**; fitting and Validation evaluation are the next steps.

![Overview of EEG and eye-tracking data processing and evaluation](images/study_overview.svg)

*Figure 1. Executed 18-person source audit, 630-window alignment and 630-row feature extraction, followed by proposed classical/temporal modelling and Validation-only selection. No expanded-model performance is shown.*

## 1. Dataset and Experimental Protocol

The source is **ZuCo 2.0**, NEMAR dataset [`nm000279`, release `v1.0.0`](https://www.nemar.org/dataset/nm000279). The current source cohort contains **18 paired-file-eligible participants** with **36 original NR/TSR Run-1 recordings**, split at participant level into **11 Train, 3 Validation and 4 reserved Test people**, retaining all six previously assessed pilot assignments. EEG/eye sidecars report nominal **500-Hz sampling**, **128 EEG channels**, and relative eye `StartTime` offsets between **−0.046 and +0.033 s**. Stage 02 used this source registry to produce **630 eligible paired windows**. Independent clock drift and calibration validity remain unverified.

Source acquisition reads only selected BIDS participants and NR/TSR Run-1 records through the public NEMAR source gateway. To select a different currently supported cohort size, change **`MAX_PARTICIPANTS`** in Notebook 01; the participant-role helper retains the original six-person assignments. `RUNS_PER_TASK=1` is an approved scientific setting, **not** a routine population-size control. The full original archive is not required in advance, and raw EEG/eye physiology is not committed to GitHub.

The initial design uses **one run per task per eligible person**, 12-second non-overlapping paired windows, six two-second feature bins per window, and a seeded **participant-disjoint** Train/Validation/Test split. Run and task labels are verified from BIDS metadata and are never inferred from EEG amplitudes or gaze patterns. Source eligibility rules and scientific boundaries are described in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

**Important limitations:** identical nominal EEG and gaze sample rates do not prove independent-clock synchronisation. The current mapping uses BIDS `StartTime` and sample frequencies, with drift unverified unless independent markers become available. NR versus TSR differs in instructions and possibly text/stimulus content; performance cannot automatically be attributed to cognitive processing or multimodal complementarity.

## 2. Experimental Programme

**Execution status:** Expanded **Notebooks 01–03 have executed and are committed**, with the 18-person role registry, 630-window paired alignment and complete 630-row EEG/gaze feature set. The Stage-03 CSV has **zero non-finite window-mean predictors**, and `03_feature_quality.csv` identifies one Train-constant predictor, `gaze_valid_fraction`. The historical **six-person pilot** completed all six development notebooks without final Test scoring (reference commit `8f2e206`). The current expanded Stage 04 is **prepared but has not run**; no expanded Validation scores have been produced.

**Current stage-owned files:** **Seven `01_*` source files**, **five `02_*` alignment files**, and **seven `03_*` feature/quality files** are now committed (**19 compatible expanded outputs**). The older six-person `04_*`–`06_*` outputs are retained in repository history but **not** the current output directory. A historical gaze-logistic Validation Macro F1 of **0.8862** came from **31 windows of one pilot participant**, not the new 99-window, three-person Validation cohort. No Test scores exist for the current experiment.

| Stage | Notebook | Scientific role | Intended named outputs |
| --- | --- | --- | --- |
| 01 | [ZuCo Source Audit and Protocol](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) | **Expanded source stage complete:** 18 participants, 36/36 source packages, 11/3/4 pilot-preserving participant roles, no Test scoring | Seven reconciled current `01_*` files: manifest, assignments, eligibility, download/sidecar evidence, protocol and source overview |
| 02 | [EEG–Gaze Alignment and Quality](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) | **Executed, published:** 36 source runs, 630/630 finite-coordinate-eligible windows; 371/99/160 window split, drift unverified | Five committed `02_*` artefacts, including per-person/task QA |
| 03 | [Preprocessing, Features and Neurophysiology](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) | **Executed, published:** 630/630 matched feature rows, eight finite mean features, two 630×6×4 arrays, zero exclusions, one Train-constant gaze feature | Seven committed `03_*` artefacts: feature CSV, NPZ, exclusion/quality tables and two figures |
| 04 | [Unimodal Baselines and Classical Fusion](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) | **Ready to execute:** seven prespecified candidates on 371 Train windows; 99 Validation windows from three people; Test untouched | Six fitted estimators, fixed late-fusion rule, pooled and per-person Validation tables, and two descriptive figures |
| 05 | [Temporal and Advanced Fusion](code/notebooks/05_temporal_and_advanced_fusion.ipynb) | **Pending expanded run:** three GRU candidates on matched six-bin arrays and the same frozen Train/Validation split | Temporal model checkpoints, scaler, history, Validation predictions and descriptive person-level metrics |
| 06 | [Validation Selection and Robustness](code/notebooks/06_robustness_and_final_evaluation.ipynb) | **Pending expanded run:** compare ten candidates using Validation-only selection and descriptive robustness; no Test inference | Validation-only model selection and sensitivity evidence; Test branch remains disabled |

Each established notebook has an editable five-band scientific SVG in [`images/`](images/). The [Stage-01 source figure](images/01_zuco_source_audit_and_protocol.svg), [Stage-02 alignment figure](images/02_eeg_gaze_alignment_and_quality.svg), [Stage-03 feature figure](images/03_preprocessing_features_and_neurophysiology.svg) and [study overview](images/study_overview.svg) now reflect the executed **18-person expanded stages**. Stage-04–06 numerical figures must be updated only after those notebooks are rerun; historical pilot figures are not current expanded results.

## 3. Main Results

*Stages 01–03 below refer to the **executed 18-person study**. The subsequent classical, temporal and model-selection scores are explicitly **historical six-person pilot evidence**, not outputs from the current expanded cohort. Current expanded Test is reserved and has not been scored.*

**Executed expanded source audit (Notebook 01):** All **18 indexed participants** had complete original BIDS filename sets for one NR and one TSR Run-1 recording. The **36/36 packages** totalled **3,373.4 MB (3.14 GiB)**. Source sidecars report **128 EEG channels**, nominal **500 Hz** for EEG and EyeLink, and eye-start offsets from **−0.046 to +0.033 seconds**. The frozen partition has **11 Train, 3 Validation and 4 reserved Test people**; six previously assessed pilot roles remain unchanged. The CSV/JSON provenance and identities agree. Source presence does not establish final usable EEG–gaze windows, independent clock-drift verification or model performance.

The relevant comparison is whether a fusion model improves over the strongest **EEG-only or gaze-only** comparator on *identical*, participant-disjoint examples. A negative or mixed fusion result is scientifically informative. Primary model selection is based on Validation only and must **not** be revised using future Test scores.

**Recorded expanded alignment and window audit (Notebook 02 execution, files pending):** The saved Colab notebook processed **36 original EEG–EyeLink run pairs from 18 people**, generated **630 distinct 12-second windows**, and recorded **630/630 passing** the `eye_valid_fraction ≥ 0.60` finite x/y rule. The mean and minimum finite-coordinate fractions were **1.000 across all 36 person/task groups**. `sub-YAK / NR` contributed **one window** because its EEG source lasts approximately **15.196 seconds**; many other runs reached the predefined **20-window cap**. The grouping and original tasks are:

| Partition | NR windows | TSR windows | Total windows | Independent people |
| --- | ---: | ---: | ---: | ---: |
| Train | 201 | 170 | **371** | 11 |
| Validation | 60 | 39 | **99** | 3 |
| Test (reserved) | 80 | 80 | **160** | 4 |
| **Overall** | **341** | **289** | **630** | **18** |

These figures are now supported by **five committed 18-person Stage-02 files**, including the 630-window registry, per-run alignment evidence, per-person/task QA and the saved figure. The finite x/y criterion is not independent eye-calibration or drift verification. All 36 original person/task groups produced at least one eligible window.

![Updated scientific illustration of expanded paired-window availability](images/02_eeg_gaze_alignment_and_quality.svg)

*Figure 2. Executed Stage-02 evidence: 18 participants, 630 eligible windows and BIDS-relative alignment limits. No independent inter-device drift validation is claimed.*

**Executed Stage-03 feature extraction (Notebook 03):** All **630** paired Stage-02 windows were processed, preserving **371 Train, 99 Validation and 160 reserved Test** rows from **11/3/4 people**. The saved `03_features.csv` contains **four EEG and four gaze window-mean fields**, all finite; the original source metadata and all 630 unique window IDs match Stage 02. Two temporal arrays have shape **(630, 6, 4)** per modality, as printed by the executed notebook. All **36 participant/task groups** remain represented, and no extraction failure was reported. The eight-row Train-derived quality table identifies **`gaze_valid_fraction` as constant (1.000)**, whereas the other seven mean predictors vary on Train. These are feature-availability findings, not model results or calibration validation.

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

[Historical six-person Validation ranking and masking figure](https://github.com/RezaSaadatyar/Eye-Tracking-Neurocognitive-Intelligence/blob/8f2e2064272955782400c5fb5ce15dca76640d13/eeg_eye_tracking_fusion/output/06_validation_evidence.png)

*Figure 2. Executed Notebook-06 development evidence: ranking of ten candidates and descriptive early-fusion input masking on the same single Validation participant. No Test score is depicted.*

**Final-evaluation status:** The expanded study has completed feature extraction only; **no new classical or temporal Validation scores have yet been computed**. The six-person pilot selection is historical. The four expanded Test participants and their 160 paired windows remain reserved and must not influence model fitting, selection or interpretation during development.

## 4. Running and Reviewing the Study

**Next execution: Notebook 04.** The seven Stage-01, five Stage-02 and seven Stage-03 outputs have been committed and reconciled on the **same 18 participants and 630 windows**. Open the current [Notebook 04](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) in a fresh Colab Runtime. Its repository setup uses the **approved short Colab template** and shared `setup_colab("eeg_eye_tracking_fusion")` helper; no custom checkout manager is needed. Its input guard checks that the 630 Stage-03 feature IDs, task labels and participant roles exactly match the Stage-02 registry.

Notebook 04 will fit the predefined **seven classical/simple-fusion candidates** using **371 Train windows (11 people)**, score **99 Validation windows (three people)** and record descriptive per-person metrics. Its expected new `04_validation_person_metrics.csv` contains **21 model/person rows**. No expanded score can be inferred from the old six-person pilot. Preserve the current model recipes, seed and primary pooled-window Macro F1 selection criterion; do **not** score the 160 Test windows.

After successful execution, review and commit Notebook 04 and its own `04_*` outputs before running Notebook 05. The subsequent temporal and final development stages must read the same frozen 18-person cohort. Notebook 06 remains in **`RUN_FINAL_TEST=False`** mode until a separate final-Test decision.

The raw data path is not persistent between Colab sessions. Stage 02/03 source retrieval is intentionally limited to the selected runs rather than the full 18-GB release. GPU use is optional for the compact temporal models; free Colab resource availability is not guaranteed. The original source's attribution and use conditions remain applicable.

## 5. Files and Saved Outputs

**Expanded-output status:** `02_person_task_quality.csv` and `03_feature_quality.csv` are now committed alongside the other current Stage-02 and Stage-03 artefacts. The next run will create `04_validation_person_metrics.csv` from the three independent Validation participants. Stage-05 and Stage-06 person-level Validation summaries remain intended downstream outputs; no expanded model-score files have yet been committed.

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

Notebook outputs in [`output/`](output/) are stage-owned `01_*`–`06_*` files. **Seven `01_*` files, five `02_*` files and seven `03_*` files** now form a compatible expanded 18-person evidence set (**19 named files**). **No current `04_*`–`06_*` outputs** are published for the expanded cohort; those stages' older pilot files are available only in Git history. Do not combine historical six-person model scores with the new 18-person feature table. The reserved Test remains unscored.

## 6. Interpretation and Scope

This project studies original reading-task labels from paired EEG and eye movements. A classifier may exploit task instructions, source texts, session conditions, eye movements and other nuisance differences; high task discrimination **does not directly demonstrate language comprehension, neurocognitive state, clinical validity or causal EEG contributions**.

Temporal EEG features derive from a fixed 16-channel subset without anatomical localisation claims. The published exploratory raw fixation-locked trace has a large uncorrected offset and does not substitute for a referenced/filtered, artefact-controlled ERP/FRP analysis. Participant-disjoint validation controls person overlap, **not cross-session or stimulus confounding**. Independent timing drift and external generalisation remain unverified unless explicitly assessed.

**Data and attribution:** [ZuCo 2.0 (Hollenstein et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.18/), [NEMAR release nm000279](https://doi.org/10.82901/nemar.nm000279), licence reported in the BIDS metadata as CC BY 4.0. Code and methodological outputs in this repository are separate from the original source recordings.
