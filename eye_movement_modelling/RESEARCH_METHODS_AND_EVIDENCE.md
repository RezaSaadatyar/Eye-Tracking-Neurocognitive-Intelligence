# Eye-Movement Modelling & Advanced AI | Research Methods and Evidence

**Project:** Eye-Movement Modelling & Advanced AI in Eye-Tracking & Neurocognitive Intelligence. **Target:** original four-class GazeBase recording-protocol identity. **Evidence:** The historical ten-person pilot completed all five notebooks with exploratory Test. The **100-person main study has now completed Notebooks 01–05**, with eight Validation candidates compared on 64 records from 16 people and final Test evaluation on **80 records from 20 independent people**. The pre-Test Validation-selected primary Logistic Regression achieved main Test Macro F1 **1.000** (80/80). Current `01_*`–`05_*` outputs belong to the main study; pilot evidence remains identifiable in older Git revisions. See [README](README.md) and [study workflow](images/study_overview.svg).

## 1. Research Objective and Scope

The primary question is whether supervised models using engineered gaze features or temporal sequences can distinguish the original GazeBase recording protocol for samples from unseen participants: FXS, HSS, RAN or TEX. This is **task identity**, not a cognitive or clinical endpoint. The historical ten-person pilot established the workflow. The **100-person main cohort now has all eight candidate Validation results and a reserved Test evaluation**: 64/16/20 person Train/Validation/Test groups, 64 Validation and 80 Test task recordings. Logistic Regression was selected on Validation before Test, and achieved **80/80 correct on the main Test**.

Five research questions guide the implementation: do simple priors/linear models offer meaningful baselines; do nonlinear classical models improve record-level Validation; do local/dilated temporal convolutions add information; do recurrent/attention models warrant their extra complexity; and how stable are selected predictions to withheld participants and input-channel removal?

This study does not train a fixation detector, infer reading comprehension, measure attention disorders, perform EEG fusion, predict disease or demonstrate external-dataset generalisation.

## 2. Data Sources, Cohort and Target Definitions

### 2.1 Source and availability

Use [GazeBase v2.0](https://doi.org/10.6084/m9.figshare.12912257) **Round 1, Session 1** with four tasks: FXS (fixations), HSS (horizontal saccades), RAN (random saccades) and TEX (reading). Ten source participants were previously used in the pilot. The new Notebook 01 execution verified and processed **100 complete participants and 400 original task recordings**, selected from the same GazeBase Round 1 / Session 1 source release. The Git checkout contains derived stage-01 outputs; the complete original archive is not distributed with them.

Original CSVs are read from [`data/raw/gazebase/`](../data/README.md). Reuse the maintained `gaze_source` source adapter from Eye-Tracking Foundations as **read-only infrastructure**, without using any derived Eye-Tracking Foundations output CSV. Source data are subject to CC BY 4.0; repository code has a separate MIT licence.

### 2.2 Cohort sizing and target mapping

Notebook 01 executed with `MAX_PARTICIPANTS = 100` and seed 17. The original archive supplied the first **100 eligible complete people** in source-ID order, ranging **1001–1108**. Eight incomplete or unavailable source IDs were skipped: **1028, 1036, 1050, 1056, 1075, 1082, 1104 and 1105**. Their replacement by later eligible IDs maintained a cohort of **100 people and four tasks per person**. Both the per-record audit CSV and `01_protocol.csv` preserve the exact identities; no source records were selected using model performance.

The archive passed the published size and MD5 checks in the recorded Colab source acquisition. The resulting 400 original recordings were audited and each contained valid samples, without implying uniform signal quality. Observed valid fraction ranged approximately **0.478 to 1.000**, with no non-increasing timestamp transitions reported. **Classical, CNN and TCN Validation are now recorded on this cohort**; GRU, Transformer and the reserved Test are not. A 300-person design would be a separate cohort requiring source eligibility checks and a new frozen Test definition. Task labels map to indices 0–3 in the fixed FXS/HSS/RAN/TEX order.

Each record has one Task target. Each time window inherits that record target, so thousands of windows from one person **are not independent observations**. No person/Task/record ID, sample duration, vendor parser label, source file path or post-outcome covariate enters a model input.

### 2.3 Explicit independence boundary

Use a seeded (`seed=17`) **person-disjoint** Train/Validation/Test registry. The main split keeps the **highest 20% source IDs for Test** and selects **20% of the remaining development people for Validation**, allocating the rest to Train. For 100 people this yields 64/16/20; for 300, 192/48/60. These are **observed assignments for the 100-person execution** (and only planned assignments for any future 300-person run). All four task records remain together for each person. A different cohort size requires a new frozen Test assignment before main Test exposure.

A larger experiment reassigns and reconstructs the entire model cohort and may alter Train/Validation distribution, model weights and Test composition; it is **not merely a larger copy of the Pilot score**. Inspecting a preliminary main Test and then adapting the cohort using its results would invalidate an untouched-Test claim.

## 3. Data Preparation and Representations

### Colab-only source acquisition and named downstream files

Notebook 01 downloaded and checksum-verified the original **6.71-GB GazeBase v2.0 archive** in Colab, indexing actual Round 1 / Session 1 source membership rather than assuming consecutive subject IDs. It selectively extracted the **400 chosen FXS/HSS/RAN/TEX CSVs**, preserving previously obtained local source files where available. Google Drive is not used. After successful extraction and source audit, the temporary archive was removed from the Runtime; this does not mean the original archive has been backed up.

The source preflight generated `01_source_acquisition_preflight.csv` and the task-level summary `01_source_acquisition_task_summary.csv`. Across the 100 people, the source audit reports the following results:

| Source task | Records | Original sample rows | Mean valid-sample fraction |
| --- | ---: | ---: | ---: |
| FXS | 100 | 1,507,864 | 0.9601 |
| HSS | 100 | 10,132,594 | 0.9635 |
| RAN | 100 | 10,111,342 | 0.9695 |
| TEX | 100 | 6,024,366 | 0.9470 |

The processed cohort yielded **400 feature rows** and **9,600 temporal windows** of 256 samples and four motion channels. All selected recordings contributed the configured maximum of 24 windows. The resulting split contains **64 Train people (256 records; 6,144 windows)**, **16 Validation people (64 records; 1,536 windows)** and **20 reserved Test people (80 records; 1,920 windows)**. Record IDs match exactly across the source audit, inventory, split and feature tables. The current `01_protocol.csv` records the 100 chosen IDs, source range, seed, split fractions and representation settings.

The final notebook cell generated one stage-01 ZIP containing exactly **nine derived outputs**, including the temporal NPZ (approximately **19.7 MB**) and `01_source_distribution.png`. The ZIP excludes all original raw CSVs and the source archive. Because these derived files are committed in GitHub, Notebook 02 may read them directly from a fresh checkout; alternatively, an explicitly uploaded current stage ZIP under `/content/` overrides any existing stage-01 files using `ensure_s2_stage_inputs`. Downstream `verify_main_cohort` checks the independent participant set against the protocol's exact IDs.

**Current stage boundary:** Notebooks 01–04 have now executed on the **same frozen 100-person cohort**. Current `01_*` source data, `02_*` classical models, `03_*` CNN/TCN and `04_*` GRU/Transformer checkpoints and Validation predictions are compatible with the saved 64/16/20 participant split. Tracked `05_*` artefacts remain from the ten-person pilot until Notebook 05 is executed for the main cohort; they must not be treated as main Test results. `01_protocol.csv` governs the cohort.

### 3.1 Record feature construction

For each original source CSV, read timestamps in milliseconds; `x/y` in degrees of visual angle; `val=0` source-valid gaze; positive `dP` as **uncalibrated pupil measurement** where supplied. Compute: valid fraction, horizontal and vertical position spread, median/95th percentile angular displacement and speed for neighbouring valid steps no more than 5 ms apart, and pupil coverage/median. Keep `record_id`, `participant_id`, `task`, original row count and split as **audit identifiers, not predictors**.

Preprocessing must not import the completed Eye-Tracking Foundations smoothed sample table or its feature outputs. Extreme raw step speeds remain possible and should be inspected in execution before interpretation; pilot model results must not be characterised as robust to artefacts automatically.

### 3.2 Temporal sequences

Select complete source-valid time-contiguous windows of **256 original samples**, with at most **24 evenly distributed, non-overlapping windows** per source record. A window does not bridge invalid gaze or any adjacent timestamp gap greater than 5 ms. Each window contains four channels: median-centred x/y (dva) and their first differences (dva per source step); this is **not** a learnt gaze-velocity threshold, calibrated pupil diameter or event detector.

Fit per-channel mean and standard deviation on **Training windows only**; apply those values unchanged to Validation/Test. The temporal archive stores the original record keys and window labels for grouping, but model forward functions receive only the four-channel array.

### 3.3 Missingness and exclusions

Train-only median imputation is part of each classical scikit-learn Pipeline, with scaling for Logistic Regression. If any original source recording has no contiguous 256-sample valid window, raise an informative error rather than silently discard the record or mix incompatible sample sets. The selected subset remains source-ID-ordered and therefore not random/representative.

## 4. Experimental Design and Evaluation

### 4.1 Train, Validation and Test roles

Train participants fit model parameters, classical imputation/scaling and temporal normalisation. Validation participants inform neural checkpoint selection and the final **pre-Test model ranking**. The main Test group comprises **20 independent people (80 source task recordings)** and was examined only after the fixed Validation-based primary decision was saved in Notebook 05. The historical pilot Test was separately examined in its earlier run. This chronology is supported by the notebook's code order and stored decision, not formal preregistration; any later Test-informed tuning requires a distinct qualification.

Evaluation is **record-based**: a classical model produces one prediction per source recording; a neural model produces probabilities for each window, which are then averaged by original recording before classification. Every model is compared on the same recording set in the relevant split.

### 4.2 Primary metrics and selection

Report fixed-label **Macro F1**, **Balanced Accuracy**, Accuracy, confusion matrices and, where available, performance by independent person. Validation selection sorts by Macro F1, then Balanced Accuracy, then the prespecified order: class prior, Logistic Regression, Random Forest, Gradient Boosting, CNN, TCN, GRU and Transformer. The code identifies the primary model **before its Test-scoring cell**. This order and selection flag are documented in the saved notebook; external chronological verification or formal preregistration has not been established.

All-class Test comparison is secondary and exploratory; a model with the best observed Test score does not retrospectively replace the Validation-selected primary candidate. Report sample size and participant support near every Test metric.

### 4.3 Pilot versus main evaluation

At n=10, two independent Test people cannot support a stable population accuracy estimate; the earlier results remain **pilot diagnostics**. The **100-person main cohort** provides eight candidate Validation comparisons on 16 independent people and a final Test on **20 different people**, each with four task recordings. The main Test is now observed. The first eligible source IDs are not a representative population sample; shared session/stimulus/protocol conditions and lack of external or cross-session data limit generalisation.

## 5. Methods, Models and Configurations

| Notebook | Implemented approach | Fixed initial settings |
| --- | --- | --- |
| 01 | Complete-source discovery, person-disjoint partitions, feature extraction and train-only normalisation | Main: `MAX_PARTICIPANTS=100` (historical pilot: 10); source R1/S1; `seed=17`; 256 samples × four channels; up to 24 windows/record |
| 02 | Class-prior DummyClassifier | Train-class prior with no gaze-discriminative model |
| 02 | Logistic Regression | Median imputation, StandardScaler, balanced classes, max 800 iterations |
| 02 | Random Forest | 120 trees, depth ≤8, min leaf 2, balanced classes |
| 02 | Histogram Gradient Boosting | 65 boosting iterations, seven max leaves, min leaf 2, learning rate 0.05 |
| 03 | 1D-CNN | Two Conv1d layers, hidden 32/64, temporal pooling, four classes |
| 03 | TCN | Three residual dilated blocks (1, 2, 4), non-causal offline input |
| 04 | GRU | One recurrent layer, hidden 32, last hidden state classification |
| 04 | Compact Transformer Encoder | 4× temporal pooling, hidden 32, four attention heads, two encoder blocks |
| 03–04 | Neural training and epoch selection | AdamW lr 0.001, weight decay 0.01, batch 64, up to eight epochs, early stopping patience 3, seed 17 |
| 05 | Final comparison and descriptive robustness | Frozen models; Validation-based choice; one Test evaluation; feature-channel intervention on Validation |

Models are intentionally compact; this does not imply equal effective tuning budget or training cost among architectures. Each temporal candidate saved its state dictionary, record-aggregated Validation predictions, history and confusion display. Parameter counts and selected epochs are recorded for CNN, TCN, GRU and Transformer. Notebook 05 subsequently performed the **100-person pre-Test model selection and frozen Test evaluation** of all eight candidates.

## 6. Results and Comparative Analysis

**Executed main-cohort Notebook 01:** 100 complete GazeBase Round 1 / Session 1 people, 400 source recordings, 400 classical feature rows and 9,600 train-normalised temporal windows of shape `(9,600, 256, 4)`. Train/Validation/Test comprise **64/16/20 independent people** and **256/64/80 original task recordings**. All eight candidates in Notebooks 02–04 used this fixed source cohort; Notebook 05 subsequently evaluated the main Test after the Validation-based primary selection.

**Executed 100-person Notebook 02 classical Validation:** Four fixed models were fitted on **256 Train recordings from 64 participants** and evaluated on **64 Validation recordings from 16 disjoint participants**. Recorded scores were independently recalculated from the current prediction CSVs:

| Main Validation model | Accuracy | Fixed-label Macro F1 | Balanced accuracy | Correct / 64 |
| --- | ---: | ---: | ---: | ---: |
| Class Prior | 0.250000 | 0.100000 | 0.250000 | 16 |
| Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 64 |
| Random Forest | 0.984375 | 0.984360 | 0.984375 | 63 |
| Gradient Boosting | 0.984375 | 0.984360 | 0.984375 | 63 |

The class-prior model predicted FXS for every Validation record. Random Forest and Gradient Boosting each classified one TEX record as RAN. The four 64-row classical prediction tables match the stored split, original labels and metrics. Main Test evaluation was performed later in Notebook 05; these development scores alone did not determine the final Test outcome.

**Executed 100-person Notebook 03 CNN and TCN Validation:** Both models trained on **6,144 Train windows from 64 participants** and selected checkpoints using record-aggregated predictions on **64 Validation recordings from 16 distinct people**. Each Validation recording contributed 24 temporal windows. The saved tables, histories, source identities and values agree with independent metric recalculation.

| Main temporal model | Accuracy | Fixed-label Macro F1 | Balanced accuracy | Correct / 64 | Parameters | Selected epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1D-CNN | 0.968750 | 0.968735 | 0.968750 | 62 | 11,492 | 7 |
| TCN | 0.953125 | 0.953036 | 0.953125 | 61 | 12,772 | 6 |

CNN made two errors (HSS→RAN, RAN→FXS); TCN made three (HSS→RAN, two RAN→FXS). CNN's selected epoch 7 tied epoch 5 on record Macro F1 but had lower Validation window loss, while TCN selected epoch 6 over a tied epoch 8. These are **development-stage observations**, not independent Test estimates or evidence of generalisation across datasets, sessions or devices.

**Executed main-cohort Notebook 04 GRU/Transformer Validation:** Both temporal models used **6,144 Train windows (64 people)**. Their frozen checkpoints were selected from record-level Macro F1 on **64 Validation recordings (16 people)**, with window loss as the tie-breaker. Their 64-row prediction tables independently reproduce the saved metrics and source identities.

| Main temporal model | Accuracy | Fixed-label Macro F1 | Balanced accuracy | Correct / 64 | Parameters | Selected epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 0.921875 | 0.920940 | 0.921875 | 59 | 3,780 | 7 |
| Transformer | 0.984375 | 0.984360 | 0.984375 | 63 | 33,764 | 8 |

The GRU made five Validation errors; Transformer made one, HSS→RAN. Transformer selected epoch 8 because its record Macro F1 tied epoch 6 with lower Validation window loss. These observations precede the final Test. Logistic Regression remained the highest Validation candidate and was **formally selected as the main primary model in Notebook 05**, before the saved main Test results.

**Executed main-cohort Notebook 05 Validation selection and Test:** Eight candidates were ordered using **Validation record-level Macro F1**, then Balanced Accuracy, then fixed model priority. The selected **Logistic Regression** scored **Validation Macro F1 1.000 on 64 records from 16 people** and was saved as the primary model before Test scoring. The main Test contained **80 original task recordings from 20 unseen people** (20 per task), with no overlap against Train or Validation participants. Predictions and all reported aggregate scores were independently recalculated from the current CSVs.

| Fixed candidate | Validation Macro F1 | Main Test accuracy | Main Test Macro F1 | Balanced accuracy | Correct / 80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Class Prior | 0.100000 | 0.250000 | 0.100000 | 0.250000 | 20 |
| **Logistic Regression (primary)** | **1.000000** | **1.000000** | **1.000000** | **1.000000** | **80** |
| Random Forest | 0.984360 | 1.000000 | 1.000000 | 1.000000 | 80 |
| Gradient Boosting | 0.984360 | 0.987500 | 0.987492 | 0.987500 | 79 |
| CNN | 0.968735 | 0.950000 | 0.950169 | 0.950000 | 76 |
| TCN | 0.953036 | 0.837500 | 0.823828 | 0.837500 | 67 |
| GRU | 0.920940 | 0.862500 | 0.864905 | 0.862500 | 69 |
| Transformer | 0.984360 | 0.925000 | 0.924951 | 0.925000 | 74 |

The selected model classified **all four task recordings for each of the 20 Test people** correctly. Random Forest also attained 80/80 but was not selected using Test. The secondary models' errors were concentrated in specific original task confusions, particularly RAN/HSS/FXS among temporal networks; they are descriptive comparisons, not a basis for post-Test model changes. There are 20 independent Test participants, not 80 independent people, and no external/session-disjoint validation was carried out.

**Historical ten-person pilot Notebook 01 preparation:** GazeBase Round 1 / Session 1, 10 complete participants, 40 recordings, 40 record feature rows and 960 temporal windows `(960, 256, 4)`. Seed-17 Train/Validation/Test: 6/2/2 people and 24/8/8 source records.

**Historical pilot Notebook 02 Validation:** Four fixed classical candidates fitted on 24 Train records from 6 participants and evaluated on eight Validation records from 2 participants. Class Prior achieved Accuracy **0.25**, Macro F1 **0.10** and Balanced Accuracy **0.25**; Logistic Regression, Random Forest and Gradient Boosting scored **1.00** on all three metrics. These exploratory pilot observations are separate from the main-cohort classical results.

**Interpretation and limits:** The class-prior model predicted FXS for all 8 Validation records (2 correct). The three feature-based models have tied perfect pilot Validation scores, but there are only **two independent Validation people**. Acquisition protocol and stimulus differences may dominate discrimination. These scores do not establish unbiased selected-model performance, generalisation, causal gaze mechanisms or cognitive inference. Notebook 05 has now applied the predefined selection and opened the two-person pilot Test; this does not support an unbiased estimate of broad model superiority.

**Historical pilot Notebook 03 Validation:** CNN and TCN each reached **Accuracy = Macro F1 = Balanced Accuracy = 1.00** on eight Validation recordings from two people. CNN had **11,492 trainable parameters** and selected epoch **8**; TCN had **12,772** and selected epoch **5**. These are historical pilot results, distinct from the current main-cohort CNN/TCN scores; their limited Validation support does not establish generalisation.

**Historical ten-person pilot Notebook 04 Validation:** GRU (3,780 parameters, epoch 1) achieved 2/8 correct and Macro F1 0.10; Transformer (33,764 parameters, epoch 2) achieved 4/8 correct and Macro F1 0.375. Only two independent Validation participants contributed to these historical pilot results. This is distinct from the **current main-cohort Notebook 04** with 64 Validation recordings. The pilot Test was exploratory and cannot substitute for the reserved main Test.

**Recorded Notebook 05 pilot model selection and Test:** The saved Validation-based rule uses descending Macro F1, then Balanced Accuracy, then a fixed candidate priority. Logistic Regression, Random Forest, Gradient Boosting, CNN and TCN tied at **Validation Macro F1 1.00** on eight recordings from two people; the fixed priority selected **Logistic Regression**. The stage-owned `05_model_selection.csv` records `test_used_in_selection=False`. This is a flag reflecting the code's intended selection rule; it does not independently verify the timing of execution or constitute formal preregistration.

The subsequently recorded **pilot Test comprises eight original task recordings from two people (001 and 005)**. The primary Logistic Regression achieved **Accuracy 0.875 (7/8)**, fixed-label **Macro F1 0.867** and **Balanced Accuracy 0.875**. Its only incorrect label was `S_1005_S1_FXS`, classified as TEX. Participant 001 had **4/4** correct (Macro F1 1.00); participant 005 had **3/4** correct (Macro F1 0.667). The Test predictions of all eight models were saved for descriptive comparison, but Test scores did not change the recorded primary choice.

| Frozen candidate | Pilot Test accuracy | Pilot Test Macro F1 | Pilot Test balanced accuracy |
| --- | ---: | ---: | ---: |
| Class Prior | 0.250 | 0.100 | 0.250 |
| **Logistic Regression (primary)** | **0.875** | **0.867** | **0.875** |
| Random Forest | 0.875 | 0.867 | 0.875 |
| Gradient Boosting | 0.875 | 0.867 | 0.875 |
| CNN | 0.750 | 0.733 | 0.750 |
| TCN | 0.750 | 0.742 | 0.750 |
| GRU | 0.250 | 0.100 | 0.250 |
| Transformer | 0.500 | 0.333 | 0.500 |

All eight models used the same Test records. These are **within-GazeBase, two-person pilot observations**, not external validation, an independent replication or evidence of generalisation to new sessions, stimuli and devices.

## 7. Robustness, Uncertainty and Additional Analyses

The **100-person main Validation-only sensitivity analysis** masked `pupil_coverage` and `pupil_median_uncalibrated` for the fixed Logistic Regression pipeline without refitting. The observed Macro F1 was **1.000 on original Validation records and 1.000 after masking (64 records, 16 people)**. The historical ten-person pilot had also shown 1.00 → 1.00 on eight Validation records. Neither result demonstrates causal feature irrelevance, a faithful model explanation, or stability on new acquisition protocols. A Class Prior model ignores gaze predictors.

Scientific interpretation must consider stimulus/task-protocol confounding, sample eligibility (first 100 complete source people), source validity and anomalous step speeds, participant dependence among the four recordings per person, and the absence of session-disjoint or external testing. One Test split from **20 independent people** is useful but not a population-wide reliability or calibrated uncertainty guarantee. Do not treat 80 task records as 80 independent participants.

Potential future additions—different source rounds/sessions, stimulus-disjoint evaluation, a preregistered cognitive endpoint or an external dataset—would be new experiments and require their own leakage checks and independent source support.

## 8. Current Outputs and Reproducibility

Study outputs are written under `eye_movement_modelling/output/` with distinct stage prefixes. Current `01_*` contains the 100-person source registry, features and 9,600-window archive; `02_*` contains four classical models and their current Validation evidence; `03_*` and `04_*` contain the four fitted temporal candidates, histories and 64-record Validation predictions. **Current `05_*` now contains the main Validation-selected primary decision, 640 Test predictions, eight Test metric rows, 160 model-by-participant rows, Validation-only sensitivity and four figures (nine stage-owned files).** The earlier pilot is recoverable through Git history and must not be mixed with these current outputs.

Study outputs can now be staged after review. `.gitignore` cannot apply size rules; the separate Git policy blocks unapproved blobs larger than **50,000,000 bytes**. Raw source folders remain ignored. ZIPs generated in `/content/` are not automatically copied into Git. The original GazeBase CSVs already tracked in the public repository retain their original CC BY 4.0 attribution. Stages reuse files in one Colab Runtime or explicitly upload the matching stage ZIP in `/content/` for `ensure_s2_stage_inputs`; uploaded ZIP files supersede older tracked pilot stage outputs. The main-cohort gate stops an unintentional 10-person read.

The historical ten-person pilot remains recoverable from previous Git revisions. The complete **100-person main-cohort producer-to-consumer workflow** now has current outputs for all five notebooks. The final 640 Test prediction rows were cross-checked against the frozen 80 source IDs and original labels; each model's probability vectors, aggregate metrics and model-by-person support agree with saved Test tables. Model fitting was **not independently reproduced** by this review. The nine current Stage-05 files are published individually under `output/`; the separate Colab-generated ZIP was not independently inspected.

## 9. Discussion, Limitations and Future Work

The implemented workflow is designed to compare simple feature-based models and compact temporal neural networks with participant-disjoint partitions, separate per-model outputs and explicit Test boundaries. Its scientific interpretation will remain narrow: **classification of the original recording task**. Differences in stimulus layout and recording duration, even when duration itself is excluded as a predictor, may drive Task identity. These scores would not demonstrate neurocognitive insight.

The ten-person pilot remains a historical feasibility study. The **100-person main study is now complete for its fixed GazeBase recording-protocol recognition objective**. Logistic Regression was selected from 64 Validation recordings before Test and scored **80/80** on 20 participant-disjoint Test people; Random Forest also scored 80/80 as a secondary comparison, but Test performance did not change the primary selection. The conclusion is narrow to this source, round, session, task protocol and cohort. Cross-session or external evaluation is a separate experiment, not an unperformed step in this completed study.

The historical ten-person pilot and **main 100-person five-notebook workflow are complete**. Preserve the existing notebooks, models and recorded scores without Test-informed tuning. All current `01_*`–`05_*` files belong to the main study; the Test is no longer reserved. Further work could independently examine different sessions, task/stimulus controls, more representative people or a genuinely external source, each with a new declared protocol. Within the repository's programme, the next principal research direction is **EEG–eye-tracking multimodal study planning**, while camera-based gaze estimation remains an optional independent branch. No new study has yet been implemented here.

**Source attribution:** [GazeBase dataset](https://doi.org/10.6084/m9.figshare.12912257) and [Griffith, Lohr, Abdulin and Komogortsev (2021)](https://doi.org/10.1038/s41597-021-00959-y), [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
