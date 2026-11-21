# Eye-Movement Modelling & Advanced AI | Research Methods and Evidence

**Project:** Eye-Movement Modelling & Advanced AI within Eye-Tracking & Neurocognitive Intelligence. **Target:** four-class GazeBase recording-protocol identity. **Evidence:** Notebook 01 has recorded source preparation and Notebook 02 has recorded classical-model Validation evaluation in a ten-person pilot; neural training and final Test evaluation remain pending. See [README](README.md) and the [editable workflow](images/study_overview.svg).

## 1. Research Objective and Scope

The primary question is whether supervised models trained on gaze-derived features or temporal sequences can classify **which original GazeBase recording protocol** generated an unseen participant's sample: FXS, HSS, RAN or TEX. This is **task identity**, not a cognitive or medical label. The first-stage objective is a bounded ten-person feasibility and pipeline check; the intended more substantive comparison requires at least twenty complete people, a cohort-size decision before final Test, and an independent-person hold-out.

Five research questions guide the implementation: do simple priors/linear models offer meaningful baselines; do nonlinear classical models improve record-level Validation; do local/dilated temporal convolutions add information; do recurrent/attention models warrant their extra complexity; and how stable are selected predictions to withheld participants and input-channel removal?

This study does not train a fixation detector, infer reading comprehension, measure attention disorders, perform EEG fusion, predict disease or demonstrate external-dataset generalisation.

## 2. Data Sources, Cohort and Target Definitions

### 2.1 Source and availability

Use [GazeBase v2.0](https://doi.org/10.6084/m9.figshare.12912257) **Round 1, Session 1** with four tasks: FXS (fixations), HSS (horizontal saccades), RAN (random saccades) and TEX (reading). Ten complete participants and 40 source CSVs exist in the current Repository from Eye-Tracking Foundations; Notebook 01 has now recorded separate data preparation from these inputs.

Original CSVs are read from [`data/raw/gazebase/`](../data/README.md). Reuse the maintained `gaze_source` source adapter from Eye-Tracking Foundations as **read-only infrastructure**, without using any derived Eye-Tracking Foundations output CSV. Source data are subject to CC BY 4.0; repository code has a separate MIT licence.

### 2.2 Cohort sizing and target mapping

Notebook 01 deterministically selects up to `MAX_PARTICIPANTS` complete participants in original source-ID order; incomplete requests raise an error. For initial development use **10**. After inspecting source availability, choose **20 or 30** **before main Test** and rerun all notebooks. The four Task names are mapped to fixed class indices 0–3 using the exact order FXS/HSS/RAN/TEX.

Each record has one Task target. Each time window inherits that record target, so thousands of windows from one person **are not independent observations**. No person/Task/record ID, sample duration, vendor parser label, source file path or post-outcome covariate enters a model input.

### 2.3 Explicit independence boundary

Use a seeded (`seed=17`) **person-disjoint** Train/Validation/Test registry. In the ten-person pilot, the split is fixed and exploratory. Once 20 or more people are available, choose the highest complete source IDs for Test; those people are distinct from the original pilot cohort. Remaining people are divided into Train/Validation. The main cohort size must be fixed **before its Test is opened**.

A larger experiment reassigns and reconstructs the entire model cohort and may alter Train/Validation distribution, model weights and Test composition; it is **not merely a larger copy of the Pilot score**. If the main Test is examined at 20 people, extending to 30 for a better score invalidates a naive 'never observed' claim.

## 3. Data Preparation and Representations

### Colab-only source acquisition and named downstream files

Section 02 of Notebook 01 combines original GazeBase v2.0 acquisition
with data-source selection. It checks source IDs 1001–1020 in Round 1 /
Session 1, four task files per person. Files already in the Colab checkout
are reused; missing CSVs are obtained from Figshare file `27039812`,
with a checksum-verified exact-source fallback, and extracted selectively.
The original archive is approximately 6.7 GB. Its full download and
extraction have not yet been verified in a live Colab session.

Source quality and availability are recorded in
`01_source_acquisition_preflight.csv` and
`01_source_acquisition_task_summary.csv`. These data-audit tables are
not predictions and do not extend the approved development pilot, which
currently selects 10 complete people with four source recordings each.
Other outputs are the source inventory, participant-disjoint split
registry, recording features, temporal-window NPZ, train-derived channel
normalisation, protocol CSV and source-distribution figure.

The last cell of Notebook 01 verifies all **nine named current outputs**
before calling the existing stage packager. It produces exactly one
`/content/eye_movement_modelling_Notebook_01_Outputs.zip`, containing
the nine derived files used for quality review and subsequent notebook
inputs. The original CSVs and large source archive are excluded. Thus,
downstream Notebooks 02–05 can run without original gaze files when
using the current, compatible stage-01 output archive.

Each later notebook reads the files directly when the same Colab Runtime
is active. In a new Runtime, the previously downloaded numbered ZIP must
be placed in `/content/`. The `ensure_stage_inputs` helper restores
only named required files, avoiding re-execution of unrelated earlier
stages. Notebook 05 also requires compatible stage-02, stage-03 and
stage-04 outputs containing model checkpoints and model-selection metrics.
Stage-output packages must come from a compatible cohort/feature/split
contract; success of file restoration alone does not prove scientific
compatibility.

The pilot's observed original-data outputs and classical Validation
results predate the integration and remain labelled as pilot evidence.
A larger 100-person experiment requires a separately established
sampling, partition and evaluation protocol, then refreshed derived
outputs and model fitting. Runtime-local source files may be lost after
Colab disconnects; saving the processed ZIP does not back up original
raw data for a fresh Notebook 01 run.


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

Train participants fit model parameters, imputer/scaler values and temporal normalisation. Validation participants inform early stopping and the final **pre-Test** model selection. Test participants are opened **only in Notebook 05**, when models have been trained, candidates compared on Validation and the selected model saved.

Evaluation is **record-based**: a classical model produces one prediction per source recording; a neural model produces probabilities for each window, which are then averaged by original recording before classification. Every model is compared on the same recording set in the relevant split.

### 4.2 Primary metrics and selection

Report fixed-label **Macro F1**, **Balanced Accuracy**, Accuracy, confusion matrices and, where available, performance by independent person. Validation selection sorts by Macro F1, then Balanced Accuracy, then the prespecified order: class prior, Logistic Regression, Random Forest, Gradient Boosting, CNN, TCN, GRU and Transformer. The selected primary model must be identified **before** reading any Test results.

All-class Test comparison is secondary and exploratory; a model with the best observed Test score does not retrospectively replace the Validation-selected primary candidate. Report sample size and participant support near every Test metric.

### 4.3 Pilot versus main evaluation

At n=10, two Test people cannot provide a reliable population accuracy estimate or robust model-ranking inference. Treat all scores as **developmental pilot diagnostics**. A main n=20/30 Test still has small subject support and stimulus Task confounding; any broader generalisation claim requires appropriate independent validation beyond this project.

## 5. Methods, Models and Configurations

| Notebook | Implemented approach | Fixed initial settings |
| --- | --- | --- |
| 01 | Complete-source discovery, group partitions, feature extraction and train-only normalisation | `MAX_PARTICIPANTS=10`; source R1/S1; `seed=17`; 256 samples × 4 channels; ≤24 windows/record |
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

Models are intentionally small; that does **not** guarantee an observed Colab runtime, numerical stability or fair compute cost across architectures. Each neural model independently saves its state dictionary, learning curve, record-level Validation predictions and confusion plot. Parameter counts and selected epochs will be recorded upon execution.

## 6. Results and Comparative Analysis

**Recorded Notebook 01 preparation:** GazeBase Round 1 / Session 1, 10 complete participants, 40 recordings, 40 record feature rows and 960 temporal windows `(960, 256, 4)`. Seed-17 Train/Validation/Test: 6/2/2 people and 24/8/8 source records.

**Recorded Notebook 02 Validation:** Four fixed classical candidates fitted to 24 Train records (6 people) and evaluated on **the same eight Validation records (2 independent people)**. The class-prior baseline scored Accuracy **0.25**, fixed-label Macro F1 **0.10**, Balanced Accuracy **0.25**. Logistic Regression, Random Forest and Gradient Boosting each scored **1.00** on all three metrics, correctly classifying 2/2 recordings in each of FXS/HSS/RAN/TEX. These are observed *notebook output values*, not independently recomputed from downloadable raw predictions in this review.

**Interpretation and limits:** The class-prior model predicted FXS for all 8 Validation records (2 correct). The three feature-based models have tied perfect pilot Validation scores, but there are only **two independent Validation people**. Acquisition protocol and stimulus differences may dominate discrimination. These scores do not establish unbiased selected-model performance, generalisation, causal gaze mechanisms or cognitive inference. Notebook 05 has not yet made the predefined selection or opened the pilot Test.

## 7. Robustness, Uncertainty and Additional Analyses

The implementation includes a prescribed Validation-only input perturbation: pupil features masked for a classical selected model, or relative x/y channels masked for a selected temporal model. This assesses **sensitivity to an input change**, not causal feature attribution or a faithful explanation. A class-prior model necessarily ignores gaze predictors.

Additional scientific review should examine leakage due to shared people, Task stimuli, temporal overlap, unequal windows per record, source validity and anomalous step speeds. No independent expert event annotations are used. With ten pilot people, resampling Test people into a precise confidence interval would be misleading; report individual-person variability without overstating uncertainty.

Potential future additions—different source rounds/sessions, stimulus-disjoint evaluation, a preregistered cognitive endpoint or an external dataset—would be new experiments and require their own leakage checks and independent source support.

## 8. Current Outputs and Reproducibility

Notebook outputs are written under `eye_movement_modelling/output/` using unique two-digit prefixes. Notebook 01 has produced its source inventory, split table, feature CSV, `01_protocol.csv` and train-normalised temporal NPZ. Notebook 02 has recorded four fitted Pipeline outputs, Validation probability tables, confusion matrices and aggregate `02_validation_metrics.csv` plus its comparison plot. Notebooks 03 and 04 produce model-specific PyTorch state dictionaries, learning histories, record-level predictions, confusion figures and aggregate Validation tables. Notebook 05 produces pre-Test selection metadata, frozen Test predictions, record and person-level metrics, a Validation-only perturbation table and summary figures.

All generated outputs and ZIP transfer packages are **ignored by Git by default**; the repository's separate policy blocks unapproved blobs above 50,000,000 bytes. The original GazeBase CSVs already tracked in the public repository retain their original CC BY 4.0 attribution. Stages can reuse files in one Colab runtime or restore the exact numbered, current ZIP via `ensure_stage_inputs`.

The notebook implementations are published. Notebooks 01 and 02 retain original-data execution outputs, including the new CSV protocol contract and the classical model Validation tables/figures. Numeric output ZIPs and model files are Git-ignored; this review relied on the recorded notebook outputs, not a fresh source-data execution. Several Notebook 02 code cells retain outputs while `execution_count` is null, limiting recovery of the exact execution order. Temporal models and Test evaluation are not yet verified.

## 9. Discussion, Limitations and Future Work

The implemented workflow is designed to compare simple feature-based models and compact temporal neural networks with participant-disjoint partitions, separate per-model outputs and explicit Test boundaries. Its scientific interpretation will remain narrow: **classification of the original recording task**. Differences in stimulus layout and recording duration, even when duration itself is excluded as a predictor, may drive Task identity. These scores would not demonstrate neurocognitive insight.

The first ten-person run is a functional feasibility exercise. Expanding to 20/30 people requires **authorised additional original CSVs**, a decided main cohort size, new split assignments and complete source-to-model reruns. The initial ten-person pilot is not an independent external validation. Final-model selection must remain Validation-based, with the main Test held out until the design is frozen.

Current evidence supports continuing to Notebook 03 after confirming that the locally retained `01_*` and `02_*` outputs belong to this same ten-person run. Keep the Validation/Test boundary, inspect potential extreme step-speed influence in a subsequent robustness analysis, and do not infer future Test results from the pilot Validation scores.

**Source attribution:** [GazeBase dataset](https://doi.org/10.6084/m9.figshare.12912257) and [Griffith, Lohr, Abdulin and Komogortsev (2021)](https://doi.org/10.1038/s41597-021-00959-y), [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
