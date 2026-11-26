# Eye-Movement Modelling & Advanced AI | Research Methods and Evidence

**Project:** Eye-Movement Modelling & Advanced AI, within Eye-Tracking & Neurocognitive Intelligence. **Target:** four-class GazeBase recording-protocol identity. **Scope:** 100 independent people (64 Train, 16 Validation, 20 Test), 400 source task recordings and eight fixed candidate models. The Validation-selected Logistic Regression achieved held-out Test Accuracy and Macro F1 **1.000 (80/80 across 20 people)**. Current `01_*`–`05_*` outputs document this one completed experiment. See [README](README.md) and [workflow](images/study_overview.svg).

## 1. Research Objective and Scope

The primary question is whether engineered gaze features and temporal sequences distinguish the original GazeBase recording protocols FXS, HSS, RAN and TEX for unseen participants. The target is **task identity**, not cognition or clinical status. The fixed 100-person cohort comprises **64 Train, 16 Validation and 20 Test people**. Eight candidates were compared on the same 64 Validation records; the selected primary achieved **80/80 correct** on the held-out Test.

Five research questions guide the implementation: do simple priors/linear models offer meaningful baselines; do nonlinear classical models improve record-level Validation; do local/dilated temporal convolutions add information; do recurrent/attention models warrant their extra complexity; and how stable are selected predictions to withheld participants and input-channel removal?

This study does not train a fixation detector, infer reading comprehension, measure attention disorders, perform EEG fusion, predict disease or demonstrate external-dataset generalisation.

## 2. Data Sources, Cohort and Target Definitions

### 2.1 Source and availability

The study uses [GazeBase v2.0](https://doi.org/10.6084/m9.figshare.12912257), **Round 1 / Session 1**, with four recording tasks: FXS (fixations), HSS (horizontal saccades), RAN (random saccades) and TEX (reading). The source audit identified **100 eligible participants and 400 original task recordings**. Derived outputs are in GitHub, while the complete original GazeBase archive is not duplicated here.

Original CSVs are read from [`data/raw/gazebase/`](../data/README.md). Reuse the maintained `gaze_source` source adapter from Eye-Tracking Foundations as **read-only infrastructure**, without using any derived Eye-Tracking Foundations output CSV. Source data are subject to CC BY 4.0; repository code has a separate MIT licence.

### 2.2 Cohort sizing and target mapping

Notebook 01 executed with `MAX_PARTICIPANTS = 100` and seed 17. The original archive supplied the first **100 eligible complete people** in source-ID order, ranging **1001–1108**. Eight incomplete or unavailable source IDs were skipped: **1028, 1036, 1050, 1056, 1075, 1082, 1104 and 1105**. Their replacement by later eligible IDs maintained a cohort of **100 people and four tasks per person**. Both the per-record audit CSV and `01_protocol.csv` preserve the exact identities; no source records were selected using model performance.

The source archive passed recorded size and MD5 checks. All 400 selected original recordings contained valid samples; observed per-record valid fraction ranged approximately **0.478–1.000**, and the audit found no non-increasing timestamp transitions. The fixed class index order is FXS/HSS/RAN/TEX (0–3). **All eight candidate models have completed Validation, and the 20-person held-out main Test has been evaluated in Notebook 05.** These findings do not imply identical source quality or external generalisation.

Each record has one Task target. Each time window inherits that record target, so thousands of windows from one person **are not independent observations**. No person/Task/record ID, sample duration, vendor parser label, source file path or post-outcome covariate enters a model input.

### 2.3 Explicit independence boundary

The fixed, seeded (`seed=17`) **participant-disjoint** split assigned the highest 20% of eligible source IDs to Test, then assigned 20% of the remaining development people to Validation. The **100-person cohort** contains **64 Train**, **16 Validation** and **20 Test** people; all four task recordings for a person remain in the same partition. The 20-person Test was only scored after final model selection on Validation.

Changing the source cohort, split or modelling specification after observing Test outcomes would require a separately defined experiment; the reported Test must not be treated as an untouched confirmatory sample for such revisions.

## 3. Data Preparation and Representations

### Source acquisition and derived outputs

Notebook 01 downloaded and checksum-verified the original **6.71-GB GazeBase v2.0 archive**, enumerated actual Round 1 / Session 1 files and selectively extracted the **400 eligible FXS/HSS/RAN/TEX recordings**. The archived source is not stored among the derived study outputs. Source integrity and eligibility checks are separate from the learned-model results.

The source preflight generated `01_source_acquisition_preflight.csv` and the task-level summary `01_source_acquisition_task_summary.csv`. Across the 100 people, the source audit reports the following results:

| Source task | Records | Original sample rows | Mean valid-sample fraction |
| --- | ---: | ---: | ---: |
| FXS | 100 | 1,507,864 | 0.9601 |
| HSS | 100 | 10,132,594 | 0.9635 |
| RAN | 100 | 10,111,342 | 0.9695 |
| TEX | 100 | 6,024,366 | 0.9470 |

The processed main cohort produced **400 record-feature rows** and **9,600 temporal windows**, each containing 256 samples and four motion channels. All selected recordings contributed 24 windows. The split includes **64 Train people** (256 records; 6,144 windows), **16 Validation people** (64 records; 1,536 windows) and **20 Test people** (80 records; 1,920 windows). Source record IDs agree across the acquisition audit, inventory, split and feature tables. `01_protocol.csv` contains the selected source IDs, seed, split proportions and representation settings.

Notebook 01 saved **nine stage-owned derived outputs**, including the source inventory, eligibility audit, split assignments, record features, normalised window NPZ (approximately **19.7 MB**), normalisation statistics, protocol, task summary and source-distribution plot. Downstream stages consume these named files from `eye_movement_modelling/output/`; `verify_main_cohort` checks participant identities against the protocol.

**Recorded stage boundary:** All five notebooks use the same fixed 100-person cohort. Current `01_*` stores source inputs, `02_*` classical candidates, `03_*` CNN/TCN, `04_*` GRU/Transformer, and `05_*` the selected primary and held-out Test evidence. `01_protocol.csv` specifies the governing cohort.

### 3.1 Record feature construction

For each original source CSV, read timestamps in milliseconds; `x/y` in degrees of visual angle; `val=0` source-valid gaze; positive `dP` as **uncalibrated pupil measurement** where supplied. Compute: valid fraction, horizontal and vertical position spread, median/95th percentile angular displacement and speed for neighbouring valid steps no more than 5 ms apart, and pupil coverage/median. Keep `record_id`, `participant_id`, `task`, original row count and split as **audit identifiers, not predictors**.

The feature and window pipelines are independent of the separate Eye-Tracking Foundations processed outputs. Large step-speed values can arise from source measurements and should be treated as a quality and interpretation limitation rather than post-hoc grounds for changing the fixed feature definition.

### 3.2 Temporal sequences

Select complete source-valid time-contiguous windows of **256 original samples**, with at most **24 evenly distributed, non-overlapping windows** per source record. A window does not bridge invalid gaze or any adjacent timestamp gap greater than 5 ms. Each window contains four channels: median-centred x/y (dva) and their first differences (dva per source step); this is **not** a learnt gaze-velocity threshold, calibrated pupil diameter or event detector.

Fit per-channel mean and standard deviation on **Training windows only**; apply those values unchanged to Validation/Test. The temporal archive stores the original record keys and window labels for grouping, but model forward functions receive only the four-channel array.

### 3.3 Missingness and exclusions

Train-only median imputation is part of each classical scikit-learn Pipeline, with scaling for Logistic Regression. If any original source recording has no contiguous 256-sample valid window, raise an informative error rather than silently discard the record or mix incompatible sample sets. The selected subset remains source-ID-ordered and therefore not random/representative.

## 4. Experimental Design and Evaluation

### 4.1 Train, Validation and Test roles

Train participants fit model parameters, classical imputation/scaling and temporal normalisation. Validation participants inform neural checkpoint choice and the **pre-Test ranking**. The Test has **20 independent people (80 source recordings)** and was examined only after saving the Validation-selected primary model. The code's execution order and saved decision support this chronology, without implying formal preregistration or a prospective external study.

Evaluation is **record-based**: a classical model produces one prediction per source recording; a neural model produces probabilities for each window, which are then averaged by original recording before classification. Every model is compared on the same recording set in the relevant split.

### 4.2 Primary metrics and selection

Report fixed-label **Macro F1**, **Balanced Accuracy**, Accuracy, confusion matrices and, where available, performance by independent person. Validation selection sorts by Macro F1, then Balanced Accuracy, then the prespecified order: class prior, Logistic Regression, Random Forest, Gradient Boosting, CNN, TCN, GRU and Transformer. The code identifies the primary model **before its Test-scoring cell**. This order and selection flag are documented in the saved notebook; external chronological verification or formal preregistration has not been established.

All-class Test comparison is secondary and exploratory; a model with the best observed Test score does not retrospectively replace the Validation-selected primary candidate. Report sample size and participant support near every Test metric.

### 4.3 Independence and Test interpretation

The main cohort provides eight Validation evaluations on **64 recordings from 16 independent participants**, followed by a held-out Test with **80 recordings from 20 different people**. Each person contributes four task records, not four independent participants. The first eligible source IDs are not a random population sample; common session, stimuli and acquisition conditions limit generalisation.

## 5. Methods, Models and Configurations

| Notebook | Implemented approach | Fixed initial settings |
| --- | --- | --- |
| 01 | Source audit, person-disjoint split, record features and Train-only temporal normalisation | `MAX_PARTICIPANTS=100`; Round 1 / Session 1; `seed=17`; 256 samples × four channels; up to 24 windows/record |
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

## 7. Robustness, Uncertainty and Additional Analyses

The Validation-only sensitivity analysis masked `pupil_coverage` and `pupil_median_uncalibrated` for the fixed selected Logistic Regression pipeline without refitting. Macro F1 remained **1.000 → 1.000** on 64 Validation records from 16 people. The result is a limited descriptive perturbation, not evidence of causal feature irrelevance or a faithful explanation.

Scientific interpretation must consider stimulus/task-protocol confounding, sample eligibility (first 100 complete source people), source validity and anomalous step speeds, participant dependence among the four recordings per person, and the absence of session-disjoint or external testing. One Test split from **20 independent people** is useful but not a population-wide reliability or calibrated uncertainty guarantee. Do not treat 80 task records as 80 independent participants.

Potential future additions—different source rounds/sessions, stimulus-disjoint evaluation, a preregistered cognitive endpoint or an external dataset—would be new experiments and require their own leakage checks and independent source support.

## 8. Current Outputs and Reproducibility

Study outputs are stored under `eye_movement_modelling/output/` with stage prefixes. Current `01_*` contains the 100-person source registry, features and 9,600 windows; `02_*` the classical model pipelines and Validation scores; `03_*` and `04_*` the four temporal checkpoints, histories and Validation predictions; and `05_*` the saved primary decision, **640 Test predictions**, eight aggregate Test rows, **160 per-model participant rows**, Validation sensitivity table and four figures.

The derived features, predictions, fitted models and evaluation tables are available in [`output/`](output/), separate from the original GazeBase source archive. Each stage reads the required named inputs and writes its own results. The original source is subject to GazeBase's CC BY 4.0 licence.

The five notebooks contain compatible source, model and evaluation outputs. The **640 Test predictions** match the same 80 Test source records across eight models. Saved probabilities, aggregate scores and 160 model-by-person rows are internally consistent. Independent reconstruction would still require rerunning source preparation, training and evaluation.

## 9. Discussion, Limitations and Future Work

The implemented workflow is designed to compare simple feature-based models and compact temporal neural networks with participant-disjoint partitions, separate per-model outputs and explicit Test boundaries. Its scientific interpretation will remain narrow: **classification of the original recording task**. Differences in stimulus layout and recording duration, even when duration itself is excluded as a predictor, may drive Task identity. These scores would not demonstrate neurocognitive insight.

The fixed 100-person experiment supports recognition of the original recording protocols within GazeBase Round 1 / Session 1. Logistic Regression was selected on 64 Validation recordings before Test and scored **80/80** on 20 participant-disjoint Test people. Random Forest also scored 80/80 as a secondary comparison, without changing the selected primary.

The **100-person five-notebook workflow is complete**. The observed Test must not be treated as untouched for retrospective model changes. Future work could independently examine different sessions, stimuli, more representative participant sampling or external data under a separately defined evaluation protocol.

**Source attribution:** [GazeBase dataset](https://doi.org/10.6084/m9.figshare.12912257) and [Griffith, Lohr, Abdulin and Komogortsev (2021)](https://doi.org/10.1038/s41597-021-00959-y), [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
