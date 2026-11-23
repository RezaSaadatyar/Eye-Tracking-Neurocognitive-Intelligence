# EEG–Eye-Tracking Multimodal Research: Research Methods and Evidence

**Current status:** The expanded **18-person ZuCo 2.0 Stages 01–03 have executed and been published**, covering **36 NR/TSR Run-1 recordings**, **630 eligible 12-second EEG–gaze windows**, and **630 source-matched feature rows**. The frozen group split remains **11 Train / 3 Validation / 4 reserved Test participants**, or **371/99/160 windows**, with all six previously assessed pilot roles retained. Stage-03 output includes **eight finite window-mean EEG/gaze predictors**, **two 630×6×4 temporal representations**, **zero extraction failures**, and a Train-derived quality finding that `gaze_valid_fraction` is constant (1.000). The seven current Stage-03 files are committed. Notebook 04 has been **prepared** for seven classical/simple-fusion models and per-person Validation reporting, but **has not yet executed on the expanded cohort**. Historical model results from the six-person pilot (best Validation Macro F1 0.8862 on 31 windows of one person) are retained as earlier work, not attributed to the new 18-person study. No Test inference has been produced.

## 1. Research Objective and Scope

**Research question:** For paired EEG and gaze windows obtained from ZuCo 2.0, can multimodal predictors distinguish documented reading-task conditions more effectively than gaze-only and EEG-only models under unseen-participant evaluation?

The primary supervised target is **NR versus TSR** (source task protocol). It is not direct comprehension prediction, cognition measurement or a clinical endpoint. We explicitly test whether fusion adds value over each individual modality on the *same eligible windows*, avoiding an automatic assumption that an additional modality improves classification. Advanced fusion is exploratory given the small number of independent people.

The study extends existing eye-tracking infrastructure with EEG reading, source-specific co-registration and grouped inference. No features, labels or results from GazeBase are transferred into this analysis.

## 2. Data Sources, Cohort and Target Definitions

### Sources and access

Source: [ZuCo 2.0](https://www.nemar.org/dataset/nm000279), BIDS version 1.9.0, NEMAR release `nm000279` **v1.0.0**. The source's BIDS description attributes the dataset to Hollenstein, Tröndle, Zhang and Langer (2020), with reported **CC-BY-4.0** terms. The release lists 18 people and nominally 128-channel EEG (EGI) and EyeLink 1000 left-eye data at 500 Hz. Some *per-run* EEG sidecars use different manufacturer/reference strings from general task metadata; preprocessing must follow the run actually read, and the acquisition reference cannot be inferred from the source-level summary.

The repository does not bundle original EEG/eye source signals. The source helper retrieves named files from the public NEMAR `data.nemar.org/nm000279/v1.0.0/` gateway, using the [BIDS file index](https://github.com/nemarDatasets/nm000279). Each run requires BrainVision `.eeg`, `.vhdr`, `.vmrk`, EEG JSON, channels/events tables and EyeLink `physio.tsv.gz` with its JSON sidecar. The **current expanded source manifest** contains **all 18 listed participants**, with **one NR and one TSR Run-1 recording per person**. The **earlier six-person pilot** included `sub-YAC`, `sub-YAG`, `sub-YAK`, `sub-YDG`, `sub-YDR` and `sub-YFR`; its roles remain locked in the expanded protocol. Raw source data must be reacquired or accessed through a valid local cache after Colab Runtime changes.

### Cohort and analytical units

The original person is the **independent partition and uncertainty unit**. Each person may contribute paired NR/TSR runs, each run supplies non-overlapping **12-second temporal windows**, and each window supplies **six two-second bins**. Prediction rows are windows, not independent participants; windows from the same person never cross Train/Validation/Test roles.

The **historical pilot** used seeded participant-disjoint partitioning to assign **3 Train / 1 Validation / 2 reserved Test people**; its saved Stage-02 run generated **95 Train / 31 Validation / 80 reserved Test** windows. The **current expanded Stage-01 protocol** uses `expanded_cohort_roles` with `seed=17`, allocating **11 Train / 3 Validation / 4 reserved Test people** and explicitly preserving the original pilot assignments. `01_source_manifest.csv`, `01_split_assignments.csv`, `01_source_eligibility.csv` and `01_protocol.json` reconcile on **18 unique person identities and 36 original NR/TSR Run-1 records**. The expanded cohort has **not yet been screened for adequate overlap or eligible 12-second EEG–gaze windows**.

### Inputs and targets

**Task** is read from the BIDS filename/metadata (`nr` / `tsr`), mapped to class indices 0/1. Person ID, run ID, source file name and split role are **provenance/grouping fields**, never predictors. Candidate predictors include gaze coordinates, pupil and vendor fixation events, and EEG bandpower/RMS features within the same valid time window. Source text content is not used as a predictor in the implemented benchmark. Text, task/run duration and other stimulus differences could still confound this classification; a text-only baseline would require an independently specified compatible mapping.

## 3. Data Preparation and Representations

**Recorded expanded Stage-01 source evidence:** The newly uploaded files list **18 of 18 source participants** with complete NR and TSR Run-1 filename sets, selecting **36 distinct original task runs** with **36/36 source packages recorded present** (eight BIDS files per run). Total recorded retrieval was **3,373.4 MB (approximately 3.14 GiB)**. Every source sidecar reports **128 EEG channels**, nominal **500-Hz** EEG and eye sampling and gaze `StartTime` offsets in **[−0.046, +0.033] seconds**. The smallest observed package remains `sub-YAK / NR` (~**3.95 MB**). In the earlier *six-person* Stage-02 audit, its EEG lasted about **15.2 s** and yielded only **one eligible window**; the 18-person Stage-02 analysis has not been performed.

**Recorded source/role provenance (resolved):** The latest Notebook-01 execution used `expanded_cohort_roles` and saved the required protocol fields: `selected_people=18`, `eligible_source_pool=18`, `role_policy=pilot_roles_locked_full_pool_seeded_expansion`, the explicit six-person `pilot_role_lock`, and split counts **11/3/4**. Original pilot Test participants `sub-YAC` and `sub-YAG` remain Test; `sub-YDR` remains Validation; and `sub-YAK`, `sub-YDG`, `sub-YFR` remain Train. The new reserved Test includes `sub-YMD` and `sub-YRP`, selected before any expanded model fitting. The saved manifest, roles and eligibility agree. A stale split from an earlier execution is **superseded**, not a remaining scientific defect in the current Stage-01 evidence.

Stage 02 uses lazy BrainVision EEG loading and the compressed EyeLink table for each selected run. EEG sample times and EyeLink `StartTime + sample_index / gaze_sampling_frequency` define the **BIDS-relative common-timebase assumption**. For each run, the window builder checks plausible original event onset bounds and derives non-overlapping **12-second windows** from their common time interval, capped at **20 windows per Run 1**. The finite-eye criterion is **at least 60% rows with finite gaze x/y coordinates**; it is *not* a calibrated eye-tracking quality metric. No shared-raw-trigger drift fit or independent clock-correction estimate is implemented; every Run-level report sets `drift_verified=False`.

**Executed and published Stage-02 evidence:** All **36 original NR/TSR Run-1 pairs** yielded non-empty BIDS-relative EEG–EyeLink overlap and at least one eligible **12-second window** per participant/task. The five committed Stage-02 files contain **630/630 eligible windows** (341 NR, 289 TSR): **371 Train** (201 NR, 170 TSR; 11 people), **99 Validation** (60 NR, 39 TSR; three people), and **160 reserved Test** (80 NR, 80 TSR; four people). All 630 window IDs are unique; person roles, Run 1 and tasks match Stage 01. The **36 person/task quality groups** have mean and minimum finite gaze x/y coverage **1.000** under the narrow eligibility criterion. `sub-YAK / NR` contributes **one window** from a ~15.196-s EEG recording. Several runs reach the fixed **20-window cap**, so pooled-window inference weights participants/tasks unequally.

**Alignment limitation:** `drift_verified=False` is recorded in all 36 run-level reports. The source synchronisation assumption relies on BIDS `StartTime` and nominal sample frequencies; finite x/y coordinates are **not** independent evidence of valid gaze calibration, blink/pupil quality, EEG artefact rejection or clock-drift correction. All five current Stage-02 outputs have now been published and reconciled with Stage 01.

**Executed and published Stage-03 feature evidence:** All **630** Stage-02 eligible paired windows were processed and retained, with **371 Train, 99 Validation and 160 reserved Test** feature rows from **11/3/4 participants**. The `03_features.csv` preserves the exact 630 source window IDs, participants, task labels, original Run IDs, split roles and time bounds from Stage 02. Four EEG window-mean fields and four gaze window-mean fields have **zero non-finite values**; the extraction-exclusion CSV contains no data rows, and the 36 person/task support groups sum to 630. Executed Notebook 03 printed **two aligned temporal array shapes of (630, 6, 4)**, one per modality; the current compressed NPZ has been committed. The eight-row `03_feature_quality.csv` reports **371 Train windows per feature**, with `gaze_valid_fraction` **constant on Train** (standard deviation 0.000), whereas the other seven predictors vary. This is a feature-quality and provenance check, **not** a model result or independent evidence of eye calibration.

Stage 03 reads 16 actual EEG channels (`E1` onwards when available, for compact analysis). Per-window channel centring acts as a reference and a 1–40 Hz Butterworth filter is applied to the loaded EEG window. For each two-second bin, Welch spectra yield channel-mean theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) log bandpower and EEG RMS (µV). Eye features are finite-coordinate fraction, median pixel speed, median pupil size and fraction of time overlapping vendor fixation events. These eight features are separate modality groups and each paired window is represented both as a mean-feature vector and as six time bins per modality. All units and missingness are preserved; Train-only scaling/imputation belongs to modelling stages. When temporal models load the saved numeric sequences, they verify each modality's per-window six-bin means against the corresponding feature CSV row before fitting. This checks original-row alignment without enabling unsafe Pickle deserialisation; the source data and feature values are not modified.

A **Train-only**, fixation-locked raw single-channel EEG average is saved as an exploratory visual diagnostic. The recorded plot shows a large **uncorrected voltage offset of approximately −10,100 µV**; its source trace is not the mean-referenced or 1–40 Hz filtered EEG used for the spectral model features. Because overlapping reading fixations, possible ocular artefacts and absent deconvolution/baseline control limit interpretation, it is **not** a validated ERP/FRP effect, cortical localisation or explanation of reading cognition.

## 4. Experimental Design and Evaluation

**Expanded source, alignment and feature scope:** The current **18-person Stages 01–03** are published and reconciled: 36 source runs, 630 eligible EEG–eye windows, 630 paired feature rows and two 630×6×4 temporal representations. Their participant roles remain immutable, with all six pilot-role assignments preserved. The seven classical/simple-fusion candidates in **Notebook 04 are ready but not yet fitted** on the expanded Train cohort. An additional per-person Validation summary is included to show heterogeneity across the three independent Validation participants; primary selection remains based on the predefined pooled-window Macro F1. Clock drift, calibration and stimulus confounding remain scientific limitations.

**Scientific rationale:** Previously inspected pilot Validation examples must not be reclassified as unseen Test data. Seed-based re-splitting after changing the participant limit could expose the selected model to people subsequently labelled Test. Source-run and Task coverage, finite EEG/gaze temporal pairing and participant-level class counts are therefore checked before modelling. The new Stage-02 run-task quality table and Stage-03 Train-feature quality table document exclusions and constant or missing measurements, without claiming independently verified clock drift. If a selected run loses both-class coverage, the pipeline stops rather than silently changing its scientific population.

The fixed participant grouping is the protection against direct same-person overlap. All **ten recorded candidates** used the original Train/Validation roles: seven class-prior, classical unimodal or early/late-fusion baselines and three compact temporal GRU models. The saved Validation prediction tables were checked against the same 31 original window identities and task labels, with probabilities ordered NR then TSR.

**Executed primary selection (before Test):** maximise **fixed-label Validation Macro F1**, then Balanced Accuracy, then fixed candidate priority. Stage 06 applied this prespecified ranking to the ten saved Validation candidates and recorded **`selected_model=gaze_logistic`**, **`validation_macro_f1=0.8861689106487147`**, **`validation_n_people=1`**, **`test_used_in_selection=False`** and **`run_mode=development`** in `06_model_selection.csv`. No Test result contributed to this decision.

**Executed development mode:** Notebook 06 ran with **`RUN_FINAL_TEST=False`** on the initial six-person engineering subset. It saved selection and Validation-only sensitivity evidence, without computing Test metrics, per-person Test results or Test confusion matrices. Its conditional Test branch is retained for a **separately reviewed and authorised** protocol. A cohort expansion or changed candidate set would require a new documented development analysis; modifying the protocol after examining Test outcomes cannot be reported as untouched independent Test evidence.

## 5. Methods, Models and Configurations

**Implementation and environment:** All six notebooks now use the approved **short Colab checkout template** and shared `setup_colab("eeg_eye_tracking_fusion")` helper; the expanded Notebook 03 execution retained all 630 windows. The model protocol remains unchanged: **12-second source windows**, **six two-second bins**, fixed seed **17**, seven classical/simple-fusion models and three compact GRU models, with participant-disjoint Train/Validation/Test roles. Notebook 04 has been prepared to fit on **371 Train windows (11 people)** and score **99 Validation windows (three people)**. A descriptive `04_validation_person_metrics.csv` now supplements its pooled Validation scores, without altering estimator configurations or model selection. Per-person metrics are not a new hyperparameter-selection criterion. Grouped cross-validation and independent Test scoring have not been added.

| Candidate | Input | Configuration and role |
| --- | --- | --- |
| Class Prior | No physiological predictor | Training class frequency baseline |
| Gaze Logistic / Gaze Forest | Gaze-only mean features | Train-only median imputation; regularised logistic or 150-tree random forest |
| EEG Logistic / EEG Forest | EEG-only spectral/RMS means | Same fixed preprocessing and model families |
| Early Fusion Logistic | Joined EEG+gaze means | Same Train-fitted median/scaler/logistic pipeline |
| Late Equal Fusion | Gaze and EEG logistic probabilities | Fixed `0.5/0.5` mean; no Test-tuned weight |
| Gaze GRU / EEG GRU | Six two-second modality bins | Hidden size 16, dropout 0.2, AdamW, maximal 15 epochs, early stopping on Validation |
| Gated Fusion GRU | Paired EEG and gaze bins | Sigmoid blend of modality embeddings plus compact GRU; gating is not causal attribution |

The temporal models apply **Train-only** feature-wise medians, means and standard deviations; the saved `05_scaling.npz` supports consistent inference. Each GRU used **six temporal bins**, **hidden width 16**, **dropout 0.2**, **AdamW (learning rate 0.001, weight decay 0.001)**, batch size **64**, up to **15 epochs** and early-stopping patience **four**. Checkpoints were selected by Validation Macro F1, Balanced Accuracy and lower Validation cross-entropy. Common seed **17** is recorded. Stage 05 saved **1,090 parameters** for each unimodal GRU and **2,354** for gated fusion. These compact architectures remain exploratory given the low independent participant count.

## 6. Results and Comparative Analysis

**Current expanded source-to-feature evidence (Notebooks 01–03):** Eighteen independent participants supplied **36 original NR/TSR Run-1 source recordings**. Notebook 02 generated **630 BIDS-relative, finite-coordinate-eligible 12-second windows**, and Notebook 03 extracted **630 paired window-mean feature rows**, with **zero exclusions and no non-finite means**. The two saved modality-specific temporal arrays were reported as **(630, 6, 4)**. The frozen split is **371 Train windows (11 people)**, **99 Validation (three people)** and **160 reserved Test (four people)**. All unique source IDs, roles, reading-task labels and window bounds reconcile between the current `02_window_registry.csv` and `03_features.csv`. The sole Train-constant summary feature is `gaze_valid_fraction` (1.000), which provides no within-cohort discrimination. **No expanded-cohort model scoring has been performed.**

**Historical six-person pilot: classical and simple-fusion Validation comparison (Notebook 04).** The following table is **archived development evidence**, not an executed result from the current 18-person study. In the earlier pilot, seven fixed candidates were evaluated on **31 Validation windows (20 NR, 11 TSR) from one person**. Recalculation of the historical **217 prediction rows** agreed with the recorded metrics. The expanded Notebook 04 must be rerun before any model ranking for the **99-window, three-person** Validation cohort is available.

| Candidate | Validation Macro F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| Gaze logistic regression | **0.8862** | **0.8636** | **0.9032** |
| Equal-weight late fusion | 0.7786 | 0.7682 | 0.8065 |
| Gaze random forest | 0.6601 | 0.6682 | 0.6774 |
| Early-fusion logistic regression | 0.5620 | 0.5909 | 0.7097 |
| EEG random forest | 0.2955 | 0.2955 | 0.3548 |
| Class prior | 0.2619 | 0.5000 | 0.3548 |
| EEG logistic regression | 0.1622 | 0.2727 | 0.1935 |

The highest-scoring classical candidate, gaze logistic regression, correctly predicted **20/20 NR** and **8/11 TSR** windows. Compared with this strongest unimodal comparator, late fusion scored **0.1076 lower** and early fusion **0.3242 lower** in Validation Macro F1. These comparisons are descriptive: repeated windows from one held-out person do not establish a population-level model ranking or independent generalisation. Source text/task differences, unequal run contributions, gaze artefacts and ocular activity in EEG may contribute to task discrimination.

**Historical six-person pilot: temporal Validation comparison (Notebook 05).** The following GRU metrics and **93 predictions** concern the earlier **31-window, single-participant** Validation set, not the expanded study. No temporal model has yet been trained or evaluated on the 18-person cohort.

| Temporal candidate | Best epoch | Parameters | Validation Macro F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEG GRU | 9 | 1,090 | **0.6890** | **0.6932** | **0.7097** |
| Gaze GRU | 15 | 1,090 | 0.3922 | 0.5000 | 0.6452 |
| Gated EEG–gaze GRU | 15 | 2,354 | 0.2619 | 0.5000 | 0.3548 |

The EEG GRU achieved **15/20 NR** and **7/11 TSR** correct predictions, whereas the gaze GRU classified **all 31** Validation windows as NR and the gated fusion GRU classified **all 31** as TSR. The 43 saved model-epoch records comprise **15 gaze GRU, 13 EEG GRU and 15 gated fusion GRU epochs**. These saved trajectories document training and checkpoint selection, not population-level generalisation.

[Historical pilot Stage-05 learning curves](https://github.com/RezaSaadatyar/Eye-Tracking-Neurocognitive-Intelligence/blob/8f2e2064272955782400c5fb5ce15dca76640d13/eeg_eye_tracking_fusion/output/05_learning_curves.png)

*Figure: Recorded Stage-05 training and Validation trajectories. Validation scores are calculated on 31 repeated windows from one participant, rather than 31 independent participants.*

**Historical pilot comparison across ten candidates:** In the earlier six-person development run, gaze logistic regression had Validation Macro F1 **0.8862**, ahead of EEG GRU (**0.6890**) on the same 31 windows. Early, late and gated-fusion variants did not surpass the strongest unimodal comparator in that single-person Validation set. **These historical rankings do not determine the expanded-cohort winner** and cannot be assumed to generalise across the three current Validation participants.

**Historical pilot evidence and limits:** The previous model-score reconciliation covered **310 saved Validation prediction rows** (217 classical and 93 temporal) and compared fixed-label metrics and probabilities from the saved pilot experiment. Those records remain available in repository Git history, not in the present expanded `output/` directory. They were not independent model retraining. The current expanded model comparison is pending, and no reserved Test labels have been used to choose its model.


**Historical six-person Stage-06 selection and confirmation.** The earlier pilot compared ten candidates and selected gaze logistic regression using the predefined Validation-only rule (Macro F1 **0.8862**, Balanced Accuracy **0.8636**, Accuracy **0.9032** on one person). That choice is **not** a selection for the expanded cohort. Historical model-input checks, fixed NR/TSR probabilities and saved checkpoint integrity checks did not establish external validity; final Test scoring remained disabled.

[Historical pilot Stage-06 Validation ranking and sensitivity](https://github.com/RezaSaadatyar/Eye-Tracking-Neurocognitive-Intelligence/blob/8f2e2064272955782400c5fb5ce15dca76640d13/eeg_eye_tracking_fusion/output/06_validation_evidence.png)

*Figure: Recorded Stage-06 development evidence: all ten candidate Macro F1 values and three fixed-model Validation masking conditions. No held-out Test data are scored or plotted.*


## 7. Robustness, Uncertainty and Additional Analyses

**Executed Stage-06 Validation sensitivity:** The fixed Stage-04 early-fusion logistic model was scored after replacing EEG features or gaze features with missing values handled by its already fitted Train-median imputer. No model was refitted and the Validation-selected primary candidate was not changed.

| Fixed early-fusion input | Validation Macro F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| Both modalities unchanged | 0.5620 | 0.5909 | 0.7097 |
| EEG group masked | **0.8862** | **0.8636** | **0.9032** |
| Gaze group masked | 0.4833 | 0.5182 | 0.4839 |

These observations concern **31 repeated Validation windows from one person** and indicate sensitivity of a **fixed estimator** to imputation-based feature removal. The equality between the EEG-masked Macro F1 and the independently trained gaze-logistic model's recorded Macro F1 **does not prove identical predictions or causal modality importance**. Neither a robust EEG harm effect nor gaze-specific physiology is established.

**Uncertainty and withheld Test:** `RUN_FINAL_TEST=False` meant no person-level Test metrics, bootstrap intervals, Test predictions or Test figures were produced. If a separately authorised future final Test has at least three independent people, the current conditional code provides **500 participant-cluster bootstrap resamples**; the existing two-person reserved Test does not support a credible population-level confidence interval. Repeated windows must not be mistaken for independent subjects.

Task/stimulus confounding, unequal quality, eye artefacts in EEG, reading-event overlap, timebase assumptions, limited electrode coverage and small independent sample size are material limitations. No cross-session or external validation is implemented in the initial six-notebook programme.

## 8. Current Outputs and Reproducibility

**Current published evidence boundary:** The active `output/` directory now contains **19 compatible named files**: **seven Stage-01**, **five Stage-02** and **seven Stage-03** outputs for the **18-person, 630-window** expanded study. All have been committed and reconciled through Stage 03. **No current `04_*`–`06_*` model outputs** have yet been produced for this cohort. The earlier six-person pilot's model checkpoints, plots and metrics are **historical Git revisions**, not current expanded scores. The reserved Test group remains unchanged and unscored.

**Stage-owned deliverables:** The complete current Stage-02 alignment set includes `02_person_task_quality.csv`, and the current Stage-03 set includes `03_feature_quality.csv`, `03_features.csv`, `03_temporal_sequences.npz`, quality/exclusion tables and two descriptive plots. Both stages are now **executed and published**. Notebook 04 is prepared to create seven-model pooled Validation evidence and **21 per-model/per-person Validation metric rows** (seven candidates × three participants); these outputs are **not yet executed**. Later Stages 05–06 must read only the compatible current expanded outputs.

| Artefact | Path | Producing stage and intended use |
| --- | --- | --- |
| **Executed 18-person source audit and frozen roles** | `output/01_source_manifest.csv`, `01_split_assignments.csv`, `01_source_eligibility.csv`, `01_download_inventory.csv`, `01_acquisition_metadata.csv`, `01_protocol.json`, `01_source_overview.png` | **Seven reconciled Stage-01 files:** 18 selected people, 36 NR/TSR source recordings, 36/36 downloaded packages, 3,373.4 MB, 128-channel and 500-Hz acquisition metadata and 11/3/4 pilot-preserving participant roles |
| **Expanded Stage-02 executed alignment and window audit** | `output/02_window_registry.csv`, `02_alignment_quality.csv`, `02_window_summary.csv`, `02_person_task_quality.csv`, `02_alignment_quality.png` | **Five current files:** 630 unique windows, 36 source runs, 18 people, six split/task summaries, 36 per-person/task QA rows and unverified drift |
| **Expanded Stage-03 executed EEG/gaze representations** | `output/03_features.csv`, `03_temporal_sequences.npz`, `03_exclusion_reasons.csv`, `03_modality_quality.csv`, `03_feature_quality.csv`, `03_temporal_example.png`, `03_fixation_locked_diagnostic.png` | **Seven current files:** 630 aligned feature rows, eight finite mean predictors, 630×6×4 modality arrays, 36 person/task groups, zero exclusions, one Train-constant feature and two descriptive figures (raw EEG is not a validated FRP) |
| **Stage-04 classical comparisons (next expanded run)** | `output/04_validation_metrics.csv`, `04_validation_predictions.csv`, `04_validation_person_metrics.csv`, `04_*.joblib`, `04_late_fusion.json`, `04_validation_comparison.png`, `04_validation_confusion.png` | **Expected, not yet executed for 18 people:** seven pooled Validation candidates, 693 per-model/window prediction rows and 21 per-model/person rows, evaluated on the same 99 Validation windows |
| **Stage-05 temporal comparisons (future expanded run)** | `output/05_validation_metrics.csv`, `05_validation_predictions.csv`, `05_training_history.csv`, `05_scaling.npz`, `05_gaze_gru.pt`, `05_eeg_gru.pt`, `05_gated_fusion_gru.pt`, `05_learning_curves.png` | **Pending expanded execution:** three compact GRU models; earlier six-person results exist only in history |
| **Stage-06 Validation selection (future expanded run)** | `output/06_model_selection.csv`, `06_validation_sensitivity.csv`, `06_validation_evidence.png` | **Pending expanded execution:** ten-candidate Validation ranking and fixed-model sensitivity, with the final Test branch disabled |

All six notebooks and stage-level scientific SVGs exist. The active expanded study has **seven Stage-01, five Stage-02 and seven Stage-03 files (19 compatible current outputs)**, all published and reconciled for 18 people and 630 paired windows. **No current Stage-04–06 outputs are present.** The historical six-person model evidence is preserved in Git revisions, not reproduced by the current expanded workflow. Additional runs require the original ZuCo source and the corresponding current saved upstream artefacts. No Test-scored output is published.

## 9. Discussion, Limitations and Future Work

**Immediate next execution checkpoint:** Run the current **Notebook 04** from a fresh Colab session, using the committed **630-row Stage-03 feature CSV** and **630-window Stage-02 registry**. Its unchanged short Setup and source-ID checks guard the original participant allocations; it will fit seven classical/simple-fusion candidates exclusively on **371 Train windows from 11 people**, evaluate **99 Validation windows from three people**, and save the pooled and descriptive person-level metrics without Test scoring. Inspect and commit the revised `04_*` results before executing Notebook 05, then Notebook 06 in **`RUN_FINAL_TEST=False`** development mode. The participant-disjoint design does not remove text/task or event-timing confounding, so interpret new metrics conservatively.

The recorded study provides a transparent, grouped Validation comparison of **ten EEG-only, gaze-only and fusion candidates**, a completed **development-mode Validation selection** and a fixed early-fusion sensitivity analysis, with explicit provenance, timebase uncertainty and participant-disjoint partitioning. On one held-out Validation participant, classical gaze logistic regression had the highest saved Macro F1 (**0.8862**). EEG GRU was the strongest temporal candidate (**0.6890**), while the gated temporal model made exclusively TSR predictions and scored **0.2619** Macro F1. The two separate fusion approaches were not consistently beneficial in this small sample. These outcomes support only **descriptive, within-cohort Validation observations**; they do not establish general model superiority, a population-level fusion disadvantage, physiological explanations or clinical relevance. Stage 06 has now **executed the prespecified Validation-only selection** and a fixed-model perturbation analysis; the reserved Test evaluation remains **unexecuted**.

ZuCo NR and TSR differ in task instructions and may also differ in source-text characteristics; a classifier may discriminate task protocol rather than cognition. The new 18-person source, alignment and feature outputs have been reconciled, but **independent EEG–EyeLink clock-drift measurement, eye calibration, ocular artefact rejection and validated FRP/ERP analysis are not established**. The constant `gaze_valid_fraction` predictor does not distinguish source tasks. Repeated windows within participants and unequal original run durations limit naive window-level uncertainty interpretations. Future work may investigate stimulus-matched controls, independent timing markers, broader EEG analysis and external validation under a separately specified design.

**Attribution:** Hollenstein, N., Tröndle, M., Zhang, C., & Langer, N. (2020). *ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation*. LREC 2020; [publication](https://aclanthology.org/2020.lrec-1.18/), [NEMAR dataset DOI](https://doi.org/10.82901/nemar.nm000279). BIDS metadata identify the dataset licence as CC BY 4.0.
