# EEG–Eye-Tracking Multimodal Research

A reproducible six-notebook workflow for evaluating whether EEG and eye movements provide complementary information about **ZuCo 2.0 reading-task conditions**. **Notebooks 01–06 have recorded development executions for the initial six-person cohort**, including 206 matched EEG/gaze feature windows, seven classical and three temporal Validation candidates. Notebook 06 reconciled all ten models, recorded the **Validation-selected gaze logistic regression** and completed Validation-only sensitivity checks. **The reserved Test partition remains unscored.** The objective is to compare unimodal and fusion models on the *same* paired temporal windows from participant-disjoint Train, Validation and Test groups. The target is **normal reading (NR) versus task-specific reading (TSR)**, not comprehension, intelligence, a clinical outcome or cognitive ability.

**Expanded-cohort update (source audit recorded; role correction required):** A new Notebook-01 execution audited **18 people and 36 paired NR/TSR Run-1 recordings**, all **36/36 source packages present**, totalling **3,373.4 MB**. However, the uploaded execution used an old `split_people` implementation and moved **five previously assessed pilot participants** between Train, Validation and Test. Its **11/3/4** split counts are correct, but the participant identities **violate the approved pilot-role lock**. The corrected `expanded_cohort_roles` code has been restored to Notebook 01 and **must be executed again** before alignment, feature extraction or model training. In the corrected code, `MAX_PARTICIPANTS = 18` is the single routine cohort-size control (currently supported 6–18). Numerical outputs of Notebooks 02–06 remain from the **six-person pilot**; no 18-person model results exist.

![Overview of EEG and eye-tracking data processing and evaluation](images/study_overview.svg)

*Figure 1. Overview of the six executed development stages, including source audit, paired-window processing, ten-model Validation comparison, recorded model selection and the still-reserved Test. The schematic is a workflow summary, not a display of independent generalisation.*

## 1. Dataset and Experimental Protocol

The source is **ZuCo 2.0**, NEMAR dataset [`nm000279`, release `v1.0.0`](https://www.nemar.org/dataset/nm000279), with original BIDS EEG/eye recordings during reading tasks. The **new source audit** selected all **18 filename-eligible participants** and obtained **36 original NR/TSR Run-1 packages**; sidecars report **128-channel EEG** and nominal **500-Hz EEG and EyeLink** acquisition. Its participant split remains **unapproved pending a corrected rerun**. The older six-person Notebook-02 execution identified **206 BIDS-relative paired windows**, each 12 seconds, meeting the finite gaze x/y criterion. These 206 windows are **not** a result from the new 18-person source set; independent clock drift remains unverified.

Source acquisition reads only requested BIDS participants and NR/TSR runs using the public NEMAR file gateway. The same code supports an initial six-person development subset and an expanded source cohort by changing `MAX_PARTICIPANTS` (and, if needed, `RUNS_PER_TASK`) in Notebook 01. The complete original archive is not required in advance. Raw signals remain outside the Git repository, in Colab's temporary `/content/zuco2_source_cache` directory.

The initial design uses **one run per task per eligible person**, 12-second non-overlapping paired windows, six two-second feature bins per window, and a seeded **participant-disjoint** Train/Validation/Test split. Run and task labels are verified from BIDS metadata and are never inferred from EEG amplitudes or gaze patterns. Source eligibility rules and scientific boundaries are described in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

**Important limitations:** identical nominal EEG and gaze sample rates do not prove independent-clock synchronisation. The current mapping uses BIDS `StartTime` and sample frequencies, with drift unverified unless independent markers become available. NR versus TSR differs in instructions and possibly text/stimulus content; performance cannot automatically be attributed to cognitive processing or multimodal complementarity.

## 2. Experimental Programme

**Execution status:** The **six-person pilot** completed all six development-stage notebooks without Test scoring. Its historical baseline is retained in Git history at commit `8f2e206`. An 18-person Notebook-01 source audit has subsequently executed, replacing the seven current `01_*` artefacts; its recorded split used stale code and **needs correction**. The other **32** current `02_*`–`06_*` files still describe the **six-person pilot**. No 18-person aligned windows, extracted features, trained models or Validation results have been produced.

All six notebooks have **historical six-person development executions**. The 18-person extension has so far completed **source retrieval and metadata inspection only**; its split must be corrected before it becomes a valid downstream input. There are **39 named output files** currently committed, but they span **two incompatible execution contexts**: seven recent 18-person Stage-01 outputs and 32 older six-person Stage-02–06 outputs. They must not be analysed as one completed experiment. The frozen six-person Validation selection was `gaze_logistic` with `RUN_FINAL_TEST=False`; this result is not an 18-person estimate.

| Stage | Notebook | Scientific role | Intended named outputs |
| --- | --- | --- | --- |
| 01 | [ZuCo Source Audit and Protocol](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) | **18-person source audit recorded:** 36/36 packages present; **split rejected** because five pilot roles changed under a stale checkout. Corrected code restored; rerun required | Seven current `01_*` files from the nonconforming execution, to be regenerated |
| 02 | [EEG–Gaze Alignment and Quality](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) | **Executed:** 12 source runs, **206 eligible 12-second windows** and a recorded BIDS-relative timing/finite-x/y audit; independent clock drift not verified | Four committed `02_*` outputs: registry, 12-run quality table, split/task summary and diagnostic plot |
| 03 | [Preprocessing, Features and Neurophysiology](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) | **Executed:** 206 of 206 source windows retain aligned EEG and gaze features; no feature exclusions or non-finite window-mean values; the raw EEG diagnostic is descriptive only | Six committed `03_*` outputs: paired feature CSV, matched six-bin NPZ, quality/exclusion tables and two figures |
| 04 | [Unimodal Baselines and Classical Fusion](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) | **Executed:** seven Validation candidates; gaze logistic Macro F1 **0.8862** (one person) | Eleven recorded `04_*` files: metrics, predictions, fitted pipelines and plots |
| 05 | [Temporal and Advanced Fusion](code/notebooks/05_temporal_and_advanced_fusion.ipynb) | **Executed:** EEG GRU Macro F1 **0.6890** led three temporal Validation models; gated fusion did not improve performance (one person) | Eight recorded `05_*` files: metrics, predictions, checkpoints, scaler, history and plot |
| 06 | [Validation Selection and Robustness](code/notebooks/06_robustness_and_final_evaluation.ipynb) | **Executed in development mode:** reconciled ten models, selected `gaze_logistic` (Validation Macro F1 **0.8862**), assessed fixed-model EEG/gaze masking; **Test unscored** | Three recorded `06_*` outputs: `06_model_selection.csv`, `06_validation_sensitivity.csv`, `06_validation_evidence.png` |

Each notebook also has an editable five-band scientific SVG in [`images/`](images/) using the same role-based typography as the other projects in this repository. Notebook 01's SVG reports the **18-person source audit** and flags the unresolved split. Stage-02/03 figures still describe the **six-person pilot**. Notebook 04 and 05 figures show their respective **recorded Validation results**. Notebook 06's figure now shows the **recorded Validation-selected model, sensitivity scores and absence of Test scoring**.

## 3. Main Results

*Only the Stage-01 source-audit results below concern the newly inspected **18-person cohort**; their participant roles are provisional. All Stage-02–06 numbers, plots and predictive results remain from the earlier **six-person pilot** and must not be read as results for the expanded cohort.*

**Recorded expanded source audit (Notebook 01):** All **18 participants** had complete BIDS filename sets for one NR and one TSR Run-1 recording: **36/36 packages** totalling **3,373.4 MB (about 3.14 GiB)**. EEG/eye sidecars consistently report **128 EEG channels** and nominal **500 Hz** for both modalities. Eye-stream `StartTime` offsets range from **−0.046 to +0.033 s**. The uploaded source manifest contains **11 Train, 3 Validation and 4 Test people**, but **five original pilot roles changed**. In particular, `sub-YDR`, previously examined in Validation, was reassigned to Test. This is not an acceptable previously unseen Test group. Corrected split code is now restored, but the seven Stage-01 files still require regeneration. No 18-person window-level quality, EEG–gaze synchronisation, model fitting or predictive performance has been established.

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

**Required next execution:** Open the **current** Notebook 01 in a **fresh Colab Runtime**. Its checkout path must be `/content/Eye-Tracking-Neurocognitive-Intelligence-fusion-expanded`, and its imports must include `expanded_cohort_roles` rather than the older `split_people`. Keep `MAX_PARTICIPANTS = 18` and execute it from the first cell. Check that `01_protocol.json` contains `selected_people`, `eligible_source_pool`, `role_policy` and `pilot_role_lock`, and that `01_split_assignments.csv` retains original pilot roles: `sub-YAC`/`sub-YAG` in Test; `sub-YAK`/`sub-YDG`/`sub-YFR` in Train; `sub-YDR` in Validation. The intended total remains **11 Train / 3 Validation / 4 Test people**, but **the uploaded current split does not satisfy this identity-level requirement**.

Execute Notebooks **02, 03, 04, 05 and 06 in order**, using the newly saved outputs of the preceding stage. Each stage replaces **only its own current named outputs**; do not combine a new Stage-01 cohort with older Stage-02–06 predictions. Download/commit reviewed outputs and the executed notebook after each stage, then open the next notebook in a **fresh Runtime** so its isolated `-fusion-expanded` checkout receives the latest committed version. The checkout refuses stale remote revisions rather than resetting or deleting untracked local results. Notebook 06 still defaults to **`RUN_FINAL_TEST = False`**, which must remain unchanged during this expansion run. A separate, scientifically reviewed final Test decision follows the expanded Validation analysis.

1. **Rerun** [Notebook 01](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) with the restored pilot-preserving code, then commit the seven replaced [`01_*` files](output/) and the newly executed notebook. Do not start Notebook 02 from the current nonconforming roles.
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

Notebook outputs in [`output/`](output/) use distinct `01_*` through `06_*` prefixes. **Seven** recent `01_*` files contain the new **18-person** source audit with **nonconforming provisional roles**; **four** `02_*`, **six** `03_*`, **eleven** `04_*`, **eight** `05_*` and **three** `06_*` files are still **six-person pilot** outputs (**39 files altogether**). Do not treat this mixed directory as one valid complete cohort. The final Test branch remains unexecuted. No separate output-directory README is required. The source cache is `/content/zuco2_source_cache` during Colab execution, not `data/raw` in the tracked repository.

## 6. Interpretation and Scope

This project studies original reading-task labels from paired EEG and eye movements. A classifier may exploit task instructions, source texts, session conditions, eye movements and other nuisance differences; high task discrimination **does not directly demonstrate language comprehension, neurocognitive state, clinical validity or causal EEG contributions**.

Temporal EEG features derive from a fixed 16-channel subset without anatomical localisation claims. The published exploratory raw fixation-locked trace has a large uncorrected offset and does not substitute for a referenced/filtered, artefact-controlled ERP/FRP analysis. Participant-disjoint validation controls person overlap, **not cross-session or stimulus confounding**. Independent timing drift and external generalisation remain unverified unless explicitly assessed.

**Data and attribution:** [ZuCo 2.0 (Hollenstein et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.18/), [NEMAR release nm000279](https://doi.org/10.82901/nemar.nm000279), licence reported in the BIDS metadata as CC BY 4.0. Code and methodological outputs in this repository are separate from the original source recordings.
