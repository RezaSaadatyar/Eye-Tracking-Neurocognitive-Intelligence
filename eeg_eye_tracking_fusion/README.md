# EEG–Eye-Tracking Multimodal Research

A reproducible six-notebook workflow for evaluating whether EEG and eye movements provide complementary information about **ZuCo 2.0 reading-task conditions**. **Notebooks 01–06 have recorded development executions for the initial six-person cohort**, including 206 matched EEG/gaze feature windows, seven classical and three temporal Validation candidates. Notebook 06 reconciled all ten models, recorded the **Validation-selected gaze logistic regression** and completed Validation-only sensitivity checks. **The reserved Test partition remains unscored.** The objective is to compare unimodal and fusion models on the *same* paired temporal windows from participant-disjoint Train, Validation and Test groups. The target is **normal reading (NR) versus task-specific reading (TSR)**, not comprehension, intelligence, a clinical outcome or cognitive ability.

**Active development update (implemented, not yet executed):** The existing six notebooks now support an expandable ZuCo 2.0 participant cohort. In Notebook 01, edit only `MAX_PARTICIPANTS` (default **18**; supported pilot-preserving selection **6–18** for the currently audited release). The original six Train/Validation/Test role assignments stay fixed. Additional eligible people are assigned deterministically using the full source eligibility pool, so changing the requested number does not reshuffle already selected people. Expanded-cohort model outputs are **pending execution**; every numerical score and figure below remains from the **original six-person pilot** until stages 01–06 are rerun and verified.

![Overview of EEG and eye-tracking data processing and evaluation](images/study_overview.svg)

*Figure 1. Overview of the six executed development stages, including source audit, paired-window processing, ten-model Validation comparison, recorded model selection and the still-reserved Test. The schematic is a workflow summary, not a display of independent generalisation.*

## 1. Dataset and Experimental Protocol

The source is **ZuCo 2.0**, NEMAR dataset [`nm000279`, release `v1.0.0`](https://www.nemar.org/dataset/nm000279), with original BIDS EEG/eye recordings from simultaneous reading tasks. The release documents 18 participants, 128-channel EEG and EyeLink data at nominal 500 Hz. Notebook 01 selected **six participants and twelve original NR/TSR runs**. Notebook 02 identified **206 BIDS-relative paired windows**, each 12 seconds, with all meeting the specified finite gaze x/y criterion; independent clock drift remains unverified.

Source acquisition reads only requested BIDS participants and NR/TSR runs using the public NEMAR file gateway. The same code supports an initial six-person development subset and an expanded source cohort by changing `MAX_PARTICIPANTS` (and, if needed, `RUNS_PER_TASK`) in Notebook 01. The complete original archive is not required in advance. Raw signals remain outside the Git repository, in Colab's temporary `/content/zuco2_source_cache` directory.

The initial design uses **one run per task per eligible person**, 12-second non-overlapping paired windows, six two-second feature bins per window, and a seeded **participant-disjoint** Train/Validation/Test split. Run and task labels are verified from BIDS metadata and are never inferred from EEG amplitudes or gaze patterns. Source eligibility rules and scientific boundaries are described in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

**Important limitations:** identical nominal EEG and gaze sample rates do not prove independent-clock synchronisation. The current mapping uses BIDS `StartTime` and sample frequencies, with drift unverified unless independent markers become available. NR versus TSR differs in instructions and possibly text/stimulus content; performance cannot automatically be attributed to cognitive processing or multimodal complementarity.

## 2. Experimental Programme

**Execution status:** The **six-person pilot** completed all six development-stage notebooks, with **39 saved pilot artefacts** and no Test scores. The expanded-cohort implementation is committed but **has not executed**. Its code reads the current participant counts from the Stage-01 source manifest and never assumes a fixed 95/31/80 window allocation. Model families, six-bin features, seed 17, fixed-label Macro F1 and held-out Test policy remain unchanged.

All six notebooks have been **executed in development mode** on the six-person cohort, and **39 stage-owned outputs** are committed. Notebook 06 recorded Validation model selection and feature-masking sensitivity with **`RUN_FINAL_TEST=False`**; this does not constitute a final held-out Test evaluation. They run sequentially in Google Colab. Each stage owns named output files that subsequent stages read; rerunning a stage regenerates its current outputs rather than collecting historical fallback results.

| Stage | Notebook | Scientific role | Intended named outputs |
| --- | --- | --- | --- |
| 01 | [ZuCo Source Audit and Protocol](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) | Inspect BIDS metadata, choose paired NR/TSR runs, download selected raw files and freeze person-level roles | `01_source_manifest.csv`, `01_split_assignments.csv`, `01_protocol.json`, audit tables and source plot |
| 02 | [EEG–Gaze Alignment and Quality](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) | **Executed:** 12 source runs, **206 eligible 12-second windows** and a recorded BIDS-relative timing/finite-x/y audit; independent clock drift not verified | Four committed `02_*` outputs: registry, 12-run quality table, split/task summary and diagnostic plot |
| 03 | [Preprocessing, Features and Neurophysiology](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) | **Executed:** 206 of 206 source windows retain aligned EEG and gaze features; no feature exclusions or non-finite window-mean values; the raw EEG diagnostic is descriptive only | Six committed `03_*` outputs: paired feature CSV, matched six-bin NPZ, quality/exclusion tables and two figures |
| 04 | [Unimodal Baselines and Classical Fusion](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) | **Executed:** seven Validation candidates; gaze logistic Macro F1 **0.8862** (one person) | Eleven recorded `04_*` files: metrics, predictions, fitted pipelines and plots |
| 05 | [Temporal and Advanced Fusion](code/notebooks/05_temporal_and_advanced_fusion.ipynb) | **Executed:** EEG GRU Macro F1 **0.6890** led three temporal Validation models; gated fusion did not improve performance (one person) | Eight recorded `05_*` files: metrics, predictions, checkpoints, scaler, history and plot |
| 06 | [Validation Selection and Robustness](code/notebooks/06_robustness_and_final_evaluation.ipynb) | **Executed in development mode:** reconciled ten models, selected `gaze_logistic` (Validation Macro F1 **0.8862**), assessed fixed-model EEG/gaze masking; **Test unscored** | Three recorded `06_*` outputs: `06_model_selection.csv`, `06_validation_sensitivity.csv`, `06_validation_evidence.png` |

Each notebook also has an editable five-band scientific SVG in [`images/`](images/) using the same role-based typography as the other projects in this repository. Notebook 01–03 figures now report **observed acquisition, eligible-window and feature-extraction counts**. Notebook 04 and 05 figures show their respective **recorded Validation results**. Notebook 06's figure now shows the **recorded Validation-selected model, sensitivity scores and absence of Test scoring**.

## 3. Main Results

*The recorded figures and numerical tables in this section are historical results from the executed **six-person pilot**. They do not report performance for the implemented 18-person expansion.*

**Recorded source audit (Notebook 01):** NEMAR listed **18 participants**. Six were selected with NR and TSR Run 1, yielding **12 original source runs** and **1,135.6 MB** of downloaded EEG, EyeLink and BIDS files. The saved split contains **3 Train, 1 Validation and 2 Test people**. Source metadata report **128 EEG channels**, nominal **500-Hz EEG and eye sampling** and eye-start offsets from **−0.046 to +0.010 seconds**. These are acquisition observations, not verified synchronisation or model performance.

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

**Final-evaluation status:** All six notebooks have executed their **development stages**, but no final Test inference or independent generalisation estimate has been produced. The **80 Test windows from two participants remain reserved**. A reviewed and frozen evaluation protocol is required before enabling the optional final Test branch.

## 4. Running and Reviewing the Study

**Expanded-cohort run sequence:** Open the latest Notebook 01 in a **fresh Colab Runtime**, configure `MAX_PARTICIPANTS = 18` in its Section 02 (or another eligible count from 6 to 18) and run it. Its `01_protocol.json` records the selected count, complete source pool, and locked original pilot roles. The current source audit reported all 18 ZuCo 2.0 participants have complete paired NR/TSR Run-1 file sets; usable window and signal quality must still be checked in Stages 02–03. With 18 source-eligible participants, the provisional split policy plans **11 Train / 3 Validation / 4 Test people**, including the original six in unchanged roles. A different number or source pool can yield different counts.

Execute Notebooks **02, 03, 04, 05 and 06 in order**, using the newly saved outputs of the preceding stage. Each stage replaces **only its own current named outputs**; do not combine a new Stage-01 cohort with older Stage-02–06 predictions. Download/commit reviewed outputs and the executed notebook after each stage, then open the next notebook in a **fresh Runtime** so its isolated `-fusion-expanded` checkout receives the latest committed version. The checkout refuses stale remote revisions rather than resetting or deleting untracked local results. Notebook 06 still defaults to **`RUN_FINAL_TEST = False`**, which must remain unchanged during this expansion run. A separate, scientifically reviewed final Test decision follows the expanded Validation analysis.

1. Review the [executed Notebook 01](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) and its seven committed [`01_*` files](output/). The fixed source manifest and split are available without rerunning acquisition.
2. Review the [completed Notebook 02](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) and four saved `02_*` artefacts. It established the 206-window registry and documented the BIDS clock-drift limitation; do not rerun it only to consume its recorded outputs.
3. Review the [executed Notebook 03](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) and six saved `03_*` outputs. All 206 paired windows were retained with finite feature means; rerunning it is **not required** merely to use the recorded predictors in later stages. The raw fixation trace is exploratory and has an uncorrected offset.
4. Review the [executed Notebook 04](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) and the eleven saved `04_*` files. It fitted seven classical/simple-fusion candidates on the original **95 Train windows** and scored the **31 Validation windows** only; the saved prediction and metric tables have been reconciled.
5. Review [executed Notebook 05](code/notebooks/05_temporal_and_advanced_fusion.ipynb) and its eight saved `05_*` files: three checkpoints, a Train-fitted scaler, Validation metrics and predictions, training history and learning curves. It uses the same original participant roles; its recorded results were reconciled without retraining.
6. Review the [executed Notebook 06](code/notebooks/06_robustness_and_final_evaluation.ipynb) and the three saved `06_*` development outputs. It reconciled the ten Validation candidates, recorded the selected model and generated fixed-model sensitivity evidence with `RUN_FINAL_TEST = False`. All saved notebook execution counts and outputs are retained; no Test scores were generated.
7. Before a separate final Test run, review whether the current **six-person pilot cohort**, single-person Validation split, source-task confounding and unverified drift support the intended scientific claim. Any cohort expansion or protocol change requires fresh dependent outputs and a clearly separated development analysis, rather than relabelling the existing scores as final results.

The new study checkout is `/content/Eye-Tracking-Neurocognitive-Intelligence-fusion-expanded`; it is separate from earlier Colab checkouts, and setup refuses a stale revision instead of deleting changes. If a selected participant has no usable NR or TSR window, Notebook 02 stops with a diagnostic rather than silently changing the cohort or split. Resolve that scientific eligibility decision **before** further modelling.

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

Notebook outputs in [`output/`](output/) use distinct `01_*` through `06_*` prefixes. Seven Stage-01, four Stage-02, six Stage-03, eleven Stage-04, eight Stage-05 and **three Stage-06 outputs** are committed (**39 stage-owned files**). Notebook 06 has executed in **development mode only**; no Test scoring or final Test files have been produced. No separate output-directory README is required. The source cache is `/content/zuco2_source_cache` during Colab execution, not `data/raw` in the tracked repository.

## 6. Interpretation and Scope

This project studies original reading-task labels from paired EEG and eye movements. A classifier may exploit task instructions, source texts, session conditions, eye movements and other nuisance differences; high task discrimination **does not directly demonstrate language comprehension, neurocognitive state, clinical validity or causal EEG contributions**.

Temporal EEG features derive from a fixed 16-channel subset without anatomical localisation claims. The published exploratory raw fixation-locked trace has a large uncorrected offset and does not substitute for a referenced/filtered, artefact-controlled ERP/FRP analysis. Participant-disjoint validation controls person overlap, **not cross-session or stimulus confounding**. Independent timing drift and external generalisation remain unverified unless explicitly assessed.

**Data and attribution:** [ZuCo 2.0 (Hollenstein et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.18/), [NEMAR release nm000279](https://doi.org/10.82901/nemar.nm000279), licence reported in the BIDS metadata as CC BY 4.0. Code and methodological outputs in this repository are separate from the original source recordings.
