# EEG–Eye-Tracking Multimodal Research

A reproducible six-notebook workflow for evaluating whether EEG and eye movements provide complementary information about **ZuCo 2.0 reading-task conditions**. **Notebooks 01–06 have recorded development executions for the initial six-person cohort**, including 206 matched EEG/gaze feature windows, seven classical and three temporal Validation candidates. Notebook 06 reconciled all ten models, recorded the **Validation-selected gaze logistic regression** and completed Validation-only sensitivity checks. **The reserved Test partition remains unscored.** The objective is to compare unimodal and fusion models on the *same* paired temporal windows from participant-disjoint Train, Validation and Test groups. The target is **normal reading (NR) versus task-specific reading (TSR)**, not comprehension, intelligence, a clinical outcome or cognitive ability.

**Expanded-cohort progress:** Stage 01 is fully documented for **18 participants and 36 NR/TSR Run-1 recordings**, with all source packages present and frozen **11 Train / 3 Validation / 4 reserved Test** participant roles preserving the six-person pilot assignments. **Notebook 02 has now been executed in Colab** on this expanded cohort: its stored outputs report **630 eligible 12-second paired windows** from all **18 participants**. However, the **five new `02_*` files have not been committed to GitHub**. The four currently tracked `02_*` artefacts still represent the six-person pilot; do not combine them with the current 18-person Stage-01 files.

![Overview of EEG and eye-tracking data processing and evaluation](images/study_overview.svg)

*Figure 1. The verified 18-person source audit and the newly recorded 630-window Notebook-02 execution are distinguished from six-person historical feature extraction and model evaluations. The expanded Stage-02 CSV/plot publication remains pending; no expanded predictive results are shown.*

## 1. Dataset and Experimental Protocol

The source is **ZuCo 2.0**, NEMAR dataset [`nm000279`, release `v1.0.0`](https://www.nemar.org/dataset/nm000279). Stage 01 verified **18 paired-file-eligible people**, **36 Run-1 source packages** and the participant-disjoint **11 Train / 3 Validation / 4 reserved Test** assignment, with the original six pilot roles preserved. EEG and eye sidecars report **128 channels**, nominal **500-Hz** sampling and eye `StartTime` offsets **−0.046 to +0.033 s**. The new Notebook-02 execution recorded **630 paired 12-second windows**, but the current committed Stage-02 CSVs still contain **206 historical six-person pilot windows**.

Source acquisition reads only selected BIDS participants and NR/TSR Run-1 records through the public NEMAR source gateway. To select a different currently supported cohort size, change **`MAX_PARTICIPANTS`** in Notebook 01; the participant-role helper retains the original six-person assignments. `RUNS_PER_TASK=1` is an approved scientific setting, **not** a routine population-size control. The full original archive is not required in advance, and raw EEG/eye physiology is not committed to GitHub.

The initial design uses **one run per task per eligible person**, 12-second non-overlapping paired windows, six two-second feature bins per window, and a seeded **participant-disjoint** Train/Validation/Test split. Run and task labels are verified from BIDS metadata and are never inferred from EEG amplitudes or gaze patterns. Source eligibility rules and scientific boundaries are described in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

**Important limitations:** identical nominal EEG and gaze sample rates do not prove independent-clock synchronisation. The current mapping uses BIDS `StartTime` and sample frequencies, with drift unverified unless independent markers become available. NR versus TSR differs in instructions and possibly text/stimulus content; performance cannot automatically be attributed to cognitive processing or multimodal complementarity.

## 2. Experimental Programme

**Execution status:** Expanded **Stage 01 is executed, published and reconciled**. Expanded **Notebook 02 has executed in Colab** with 630/630 finite-coordinate-eligible windows (11/3/4 participants), but **its new five Stage-02 outputs are not yet committed**. The historical six-person pilot completed Stages 01–06 in development mode without Test scoring (reference commit `8f2e206`). Expanded Stages 03–06 are not yet run; their 206-window features and ten-model Validation results remain pilot evidence.

**Published evidence boundary:** The **seven `01_*` outputs** now describe 18 people. The **four tracked `02_*` files** and **28 tracked `03_*`–`06_*` files** describe the six-person pilot, so the GitHub output directory still contains **39 named stage artefacts across incompatible cohorts**. In contrast, the **executed Notebook-02 outputs** report 630 windows and five newly packaged files, including the yet-untracked `02_person_task_quality.csv`. These must be published before cross-stage validation or Stage 03; model scores of the historical pilot do not estimate expanded-cohort performance.

| Stage | Notebook | Scientific role | Intended named outputs |
| --- | --- | --- | --- |
| 01 | [ZuCo Source Audit and Protocol](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) | **Expanded source stage complete:** 18 participants, 36/36 source packages, 11/3/4 pilot-preserving participant roles, no Test scoring | Seven reconciled current `01_*` files: manifest, assignments, eligibility, download/sidecar evidence, protocol and source overview |
| 02 | [EEG–Gaze Alignment and Quality](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) | **Expanded notebook executed:** 36 source runs, **630/630 eligible windows** from 18 people (371 Train / 99 Validation / 160 reserved Test); independent clock drift not verified | **Five new `02_*` files packaged in Colab but not committed**; four tracked `02_*` files still contain the 206-window pilot |
| 03 | [Preprocessing, Features and Neurophysiology](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) | **Executed:** 206 of 206 source windows retain aligned EEG and gaze features; no feature exclusions or non-finite window-mean values; the raw EEG diagnostic is descriptive only | Six committed `03_*` outputs: paired feature CSV, matched six-bin NPZ, quality/exclusion tables and two figures |
| 04 | [Unimodal Baselines and Classical Fusion](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) | **Executed:** seven Validation candidates; gaze logistic Macro F1 **0.8862** (one person) | Eleven recorded `04_*` files: metrics, predictions, fitted pipelines and plots |
| 05 | [Temporal and Advanced Fusion](code/notebooks/05_temporal_and_advanced_fusion.ipynb) | **Executed:** EEG GRU Macro F1 **0.6890** led three temporal Validation models; gated fusion did not improve performance (one person) | Eight recorded `05_*` files: metrics, predictions, checkpoints, scaler, history and plot |
| 06 | [Validation Selection and Robustness](code/notebooks/06_robustness_and_final_evaluation.ipynb) | **Executed in development mode:** reconciled ten models, selected `gaze_logistic` (Validation Macro F1 **0.8862**), assessed fixed-model EEG/gaze masking; **Test unscored** | Three recorded `06_*` outputs: `06_model_selection.csv`, `06_validation_sensitivity.csv`, `06_validation_evidence.png` |

Each stage has an editable five-band scientific SVG in [`images/`](images/). [Stage 01](images/01_zuco_source_audit_and_protocol.svg) depicts its verified 18-person source/role evidence. The updated [Stage-02 figure](images/02_eeg_gaze_alignment_and_quality.svg) depicts the **executed notebook's 630-window result** and the pending output publication. The [study overview](images/study_overview.svg) distinguishes expanded stages 01–02 from historical pilot stages 03–06; other stage figures will be reconciled after their expanded runs.

## 3. Main Results

*The expanded Stage-01 source/role files are committed, while expanded Stage-02 numerical findings are currently supported by the **executed notebook only**. Tracked Stage-02 CSVs and all Stage-03–06 artefacts remain from the **six-person pilot**. The following descriptions therefore identify the execution scope explicitly; no expanded model score or Test evaluation is claimed.*

**Executed expanded source audit (Notebook 01):** All **18 indexed participants** had complete original BIDS filename sets for one NR and one TSR Run-1 recording. The **36/36 packages** totalled **3,373.4 MB (3.14 GiB)**. Source sidecars report **128 EEG channels**, nominal **500 Hz** for EEG and EyeLink, and eye-start offsets from **−0.046 to +0.033 seconds**. The frozen partition has **11 Train, 3 Validation and 4 reserved Test people**; six previously assessed pilot roles remain unchanged. The CSV/JSON provenance and identities agree. Source presence does not establish final usable EEG–gaze windows, independent clock-drift verification or model performance.

The relevant comparison is whether a fusion model improves over the strongest **EEG-only or gaze-only** comparator on *identical*, participant-disjoint examples. A negative or mixed fusion result is scientifically informative. Primary model selection is based on Validation only and must **not** be revised using future Test scores.

**Recorded expanded alignment and window audit (Notebook 02 execution, files pending):** The saved Colab notebook processed **36 original EEG–EyeLink run pairs from 18 people**, generated **630 distinct 12-second windows**, and recorded **630/630 passing** the `eye_valid_fraction ≥ 0.60` finite x/y rule. The mean and minimum finite-coordinate fractions were **1.000 across all 36 person/task groups**. `sub-YAK / NR` contributed **one window** because its EEG source lasts approximately **15.196 seconds**; many other runs reached the predefined **20-window cap**. The grouping and original tasks are:

| Partition | NR windows | TSR windows | Total windows | Independent people |
| --- | ---: | ---: | ---: | ---: |
| Train | 201 | 170 | **371** | 11 |
| Validation | 60 | 39 | **99** | 3 |
| Test (reserved) | 80 | 80 | **160** | 4 |
| **Overall** | **341** | **289** | **630** | **18** |

These counts are from the **executed Notebook-02 outputs**, **not** the presently tracked Stage-02 CSVs. The new ZIP contains **five files**: the 630-window registry, 36-run timing/quality report, six-group summary, 36-group person/task QA table and plot. **Upload them before treating Stage 02 as a published dependency for Notebook 03.** All run-level `drift_verified` entries are false by design; finite gaze coordinates do **not** independently establish calibration, blink/pupil quality, physiological EEG validity or clock synchronisation.

![Updated scientific illustration of expanded paired-window availability](images/02_eeg_gaze_alignment_and_quality.svg)

*Figure 2. Stage-02 notebook execution evidence: 18 source participants, 630 paired windows and BIDS-relative timing limitations. The SVG is a scientific schematic; the generated CSV/PNG files are still awaiting publication.*

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

**Final-evaluation status:** The six-person pilot completed Validation-only comparisons with Test disabled. Expanded Stages 01–02 have source and paired-window **execution evidence**, but the expanded Stage-02 files have not yet been published and no expanded features, model fits, Validation scores or Test inference exist. The final Test decision remains separate from development.

## 4. Running and Reviewing the Study

**Immediate action: publish the executed Notebook-02 outputs.** The saved notebook reports `Files packaged: 5` at `/content/eeg_eye_tracking_fusion_Notebook_02_Outputs.zip`. Download that ZIP if needed and **replace the four old `02_*` files** in [`output/`](output/) with the new versions; also **add `02_person_task_quality.csv`**. Do not substitute older pilot files or rebuild the CSVs from Markdown. Keep the executed Notebook-02 code/output history intact.

After pushing these five artefacts, reconcile their **630 unique source window IDs**, **36 original person/task groups**, **18 independent people**, **371/99/160** split window counts and pilot-preserving participant roles against `01_*`. Check that all original tasks contribute at least one eligible window and that drift remains recorded as unverified. Only then run [Notebook 03](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) from the current 18-person Stage-01/02 inputs and review/commit its **own** updated outputs.

The later Stages **04–06** must be rerun in order on the same expanded cohort. They retain Train-only transformations and Validation-only model selection. **Keep `RUN_FINAL_TEST=False`** throughout development. The older 206-window outputs remain historical pilot evidence and must not be combined with new 630-window records.

The raw data path is not persistent between Colab sessions. Stage 02/03 source retrieval is intentionally limited to the selected runs rather than the full 18-GB release. GPU use is optional for the compact temporal models; free Colab resource availability is not guaranteed. The original source's attribution and use conditions remain applicable.

## 5. Files and Saved Outputs

**Expanded-run artefact status:** `02_person_task_quality.csv` was **created and packaged by the executed Notebook 02** but is **not yet in GitHub**; the expanded `02_window_registry.csv`, `02_alignment_quality.csv`, `02_window_summary.csv` and `02_alignment_quality.png` likewise still need uploading. Later `03_feature_quality.csv`, `04_validation_person_metrics.csv`, `05_validation_person_metrics.csv` and `06_validation_person_metrics.csv` remain planned outputs of the unexecuted expanded downstream stages.

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

Notebook outputs in [`output/`](output/) have stage-specific `01_*`–`06_*` names. Currently, the **seven committed `01_*` files** describe 18 people, while the **four committed `02_*` files and 28 `03_*`–`06_*` files** still describe the six-person pilot. A fresh **five-file Stage-02 ZIP was generated in Colab but has not been committed**. The 39 current named output artefacts therefore **must not be treated as a complete 18-person study**. No final held-out Test outputs have been produced.

## 6. Interpretation and Scope

This project studies original reading-task labels from paired EEG and eye movements. A classifier may exploit task instructions, source texts, session conditions, eye movements and other nuisance differences; high task discrimination **does not directly demonstrate language comprehension, neurocognitive state, clinical validity or causal EEG contributions**.

Temporal EEG features derive from a fixed 16-channel subset without anatomical localisation claims. The published exploratory raw fixation-locked trace has a large uncorrected offset and does not substitute for a referenced/filtered, artefact-controlled ERP/FRP analysis. Participant-disjoint validation controls person overlap, **not cross-session or stimulus confounding**. Independent timing drift and external generalisation remain unverified unless explicitly assessed.

**Data and attribution:** [ZuCo 2.0 (Hollenstein et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.18/), [NEMAR release nm000279](https://doi.org/10.82901/nemar.nm000279), licence reported in the BIDS metadata as CC BY 4.0. Code and methodological outputs in this repository are separate from the original source recordings.
