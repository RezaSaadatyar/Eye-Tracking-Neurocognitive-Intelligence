# Eye-Movement Modelling & Advanced AI | Research Methods and Evidence

**Project:** Eye-Movement Modelling & Advanced AI within Eye-Tracking & Neurocognitive Intelligence. **Target:** four-class GazeBase recording-protocol identity. **Evidence:** Notebooks 01–05 have recorded complete source-to-Test execution of a ten-person task-classification pilot, including the Validation-selected primary model, person-disjoint Test and one Validation-only perturbation. **Notebook 01 has now been executed on 100 main-cohort participants**. The 100-person data preparation and participant split are recorded; the numerical model results below still belong exclusively to the committed ten-person pilot. See [README](README.md) and the [editable workflow](images/study_overview.svg).

## 1. Research Objective and Scope

The primary question is whether supervised models trained on gaze-derived features or temporal sequences can classify **which original GazeBase recording protocol** generated an unseen participant's sample: FXS, HSS, RAN or TEX. This is **task identity**, not a cognitive or medical label. The executed ten-person pilot established the pipeline and exposed its exploratory Test. The next model comparison will use the **already prepared 100-person cohort** with a reserved 20-person main Test. Model fitting, development results and final Test remain pending; no new Test-informed changes should be made to this frozen participant assignment.

Five research questions guide the implementation: do simple priors/linear models offer meaningful baselines; do nonlinear classical models improve record-level Validation; do local/dilated temporal convolutions add information; do recurrent/attention models warrant their extra complexity; and how stable are selected predictions to withheld participants and input-channel removal?

This study does not train a fixation detector, infer reading comprehension, measure attention disorders, perform EEG fusion, predict disease or demonstrate external-dataset generalisation.

## 2. Data Sources, Cohort and Target Definitions

### 2.1 Source and availability

Use [GazeBase v2.0](https://doi.org/10.6084/m9.figshare.12912257) **Round 1, Session 1** with four tasks: FXS (fixations), HSS (horizontal saccades), RAN (random saccades) and TEX (reading). Ten source participants were previously used in the pilot. The new Notebook 01 execution verified and processed **100 complete participants and 400 original task recordings**, selected from the same GazeBase Round 1 / Session 1 source release. The Git checkout contains derived stage-01 outputs; the complete original archive is not distributed with them.

Original CSVs are read from [`data/raw/gazebase/`](../data/README.md). Reuse the maintained `gaze_source` source adapter from Eye-Tracking Foundations as **read-only infrastructure**, without using any derived Eye-Tracking Foundations output CSV. Source data are subject to CC BY 4.0; repository code has a separate MIT licence.

### 2.2 Cohort sizing and target mapping

Notebook 01 executed with `MAX_PARTICIPANTS = 100` and seed 17. The original archive supplied the first **100 eligible complete people** in source-ID order, ranging **1001–1108**. Eight incomplete or unavailable source IDs were skipped: **1028, 1036, 1050, 1056, 1075, 1082, 1104 and 1105**. Their replacement by later eligible IDs maintained a cohort of **100 people and four tasks per person**. Both the per-record audit CSV and `01_protocol.csv` preserve the exact identities; no source records were selected using model performance.

The archive passed the published size and MD5 checks in the recorded Colab execution. The resulting 400 original records were audited and all had at least some valid samples; this does not imply uniform quality. The recorded source-valid fraction ranged from approximately **0.478 to 1.000**, and no non-increasing timestamp transitions were reported. The next model evaluation remains unexecuted. A 300-person design would be a separate cohort and must first establish source eligibility and its own frozen Test definition. Four task labels are mapped to indices 0–3 in the fixed order FXS/HSS/RAN/TEX.

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

**Current stage boundary:** Notebook 01 is executed on the new main cohort. The fitted model files and scores named `02_*` to `05_*` still come from the earlier ten-person pilot until those stages are rerun; they must not be combined with the 100-person split. The previously tracked `01_protocol.json` is a retired ten-person pilot artefact, not the governing main-study protocol.

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

Train participants fit model parameters, imputer/scaler values and temporal normalisation. Validation participants inform early stopping and the final **pre-Test** model selection. Pilot Test participants were opened in **Notebook 05 of the completed ten-person pilot** after the code's Validation-based selection. In the **new 100-person run**, highest-source-ID Test participants are reserved and must remain unobserved until after model selection on the current Validation group.

Evaluation is **record-based**: a classical model produces one prediction per source recording; a neural model produces probabilities for each window, which are then averaged by original recording before classification. Every model is compared on the same recording set in the relevant split.

### 4.2 Primary metrics and selection

Report fixed-label **Macro F1**, **Balanced Accuracy**, Accuracy, confusion matrices and, where available, performance by independent person. Validation selection sorts by Macro F1, then Balanced Accuracy, then the prespecified order: class prior, Logistic Regression, Random Forest, Gradient Boosting, CNN, TCN, GRU and Transformer. The code identifies the primary model **before its Test-scoring cell**. This order and selection flag are documented in the saved notebook; external chronological verification or formal preregistration has not been established.

All-class Test comparison is secondary and exploratory; a model with the best observed Test score does not retrospectively replace the Validation-selected primary candidate. Report sample size and participant support near every Test metric.

### 4.3 Pilot versus main evaluation

At n=10, two Test people cannot provide a reliable population accuracy estimate or robust model-ranking inference. Treat all scores as **developmental pilot diagnostics**. The configured 100-person run uses a deterministic archive-completeness eligibility rule and split percentages, but **no new original-data main Test has been executed**. Pilot feedback and task-stimulus confounding must be disclosed; cross-session, external and clinical generalisation remain unsupported.

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

Models are intentionally small; that does **not** guarantee an observed Colab runtime, numerical stability or fair compute cost across architectures. Each neural model independently saves its state dictionary, learning curve, record-level Validation predictions and confusion plot. Parameter counts and selected epochs are recorded for CNN, TCN, GRU and Transformer; the exploratory pilot Test has now been recorded.

## 6. Results and Comparative Analysis

**Executed main-cohort Notebook 01:** 100 complete GazeBase Round 1 / Session 1 people; 400 original task recordings, 400 record features and 9,600 normalised windows of shape `(9,600, 256, 4)`. The recorded Train/Validation/reserved Test allocation is **64/16/20 people**. No main-cohort training, Validation accuracy or Test score has been observed yet.

**Historical ten-person pilot Notebook 01 preparation:** GazeBase Round 1 / Session 1, 10 complete participants, 40 recordings, 40 record feature rows and 960 temporal windows `(960, 256, 4)`. Seed-17 Train/Validation/Test: 6/2/2 people and 24/8/8 source records.

**Recorded Notebook 02 Validation:** Four fixed classical candidates fitted to 24 Train records (6 people) and evaluated on **the same eight Validation records (2 independent people)**. The class-prior baseline scored Accuracy **0.25**, fixed-label Macro F1 **0.10**, Balanced Accuracy **0.25**. Logistic Regression, Random Forest and Gradient Boosting each scored **1.00** on all three metrics, correctly classifying 2/2 recordings in each of FXS/HSS/RAN/TEX. These are observed *notebook output values*, not independently recomputed from downloadable raw predictions in this review.

**Interpretation and limits:** The class-prior model predicted FXS for all 8 Validation records (2 correct). The three feature-based models have tied perfect pilot Validation scores, but there are only **two independent Validation people**. Acquisition protocol and stimulus differences may dominate discrimination. These scores do not establish unbiased selected-model performance, generalisation, causal gaze mechanisms or cognitive inference. Notebook 05 has now applied the predefined selection and opened the two-person pilot Test; this does not support an unbiased estimate of broad model superiority.

**Recorded Notebook 03 Validation:** CNN and TCN both achieved **Accuracy = Macro F1 = Balanced Accuracy = 1.00**, on **eight Validation recordings from two independent people**. CNN had **11,492 trainable parameters** and selected epoch **8**; TCN had **12,772** and selected epoch **5**. Although the TCN Validation window loss decreased after epoch 5, its record-level Macro F1 declined, so checkpoint choice follows the declared selection metric. Both models' checkpoints, histories, Validation predictions and figures were recorded; the last cell reported **10 stage-owned files**. These are original notebook output values, not independently reproduced training or evidence of generalisation.

**Recorded Notebook 04 Validation:** GRU (3,780 trainable parameters; selected **epoch 1**) achieved **Accuracy 0.25, Macro F1 0.10 and Balanced Accuracy 0.25**. It assigned TEX to every one of the eight Validation recordings, correctly predicting only two TEX records. Transformer (33,764 parameters; selected **epoch 2**) achieved **Accuracy 0.50, Macro F1 0.375 and Balanced Accuracy 0.50**, correctly identifying both HSS and both RAN records, but missing FXS and TEX. Both use **the same eight recordings from two independent Validation people** as earlier candidates. GRU trained for four epochs and Transformer five; checkpoint selection followed record-level Macro F1, rather than lowest window-level loss. Both checkpoints, histories, prediction tables and assessment figures are now available in the repository, with **ten stage-04 files** recorded. These are observed pilot results, not independent training reproduction or reliable evidence of architectural superiority. No held-out Test results were calculated.

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

The executed **Validation-only input perturbation** masked `pupil_coverage` and `pupil_median_uncalibrated` for the frozen Logistic Regression pipeline, without refitting. Its Macro F1 was **1.00 with the original features and 1.00 with masked pupil features** on the same eight Validation recordings. This is a limited sensitivity check, not causal feature attribution, a faithful model explanation or proof that pupil measurements are generally irrelevant. A class-prior model necessarily ignores gaze predictors.

Additional scientific review should examine leakage due to shared people, Task stimuli, temporal overlap, unequal windows per record, source validity and anomalous step speeds. No independent expert event annotations are used. With ten pilot people, resampling Test people into a precise confidence interval would be misleading; report individual-person variability without overstating uncertainty.

Potential future additions—different source rounds/sessions, stimulus-disjoint evaluation, a preregistered cognitive endpoint or an external dataset—would be new experiments and require their own leakage checks and independent source support.

## 8. Current Outputs and Reproducibility

Notebook outputs are written under `eye_movement_modelling/output/` using unique two-digit prefixes. Notebook 01 has produced its source inventory, split table, feature CSV, `01_protocol.csv` and train-normalised temporal NPZ. Notebook 02 has recorded four fitted Pipeline outputs, Validation probability tables, confusion matrices and aggregate `02_validation_metrics.csv` plus its comparison plot. Notebook 03 has recorded two PyTorch state dictionaries, histories, record-level Validation predictions, assessment figures and aggregate metrics; Notebook 04 has recorded GRU and Transformer model checkpoints, histories, record-level Validation predictions, evaluation figures and aggregate metrics. Notebook 05 has produced its model-selection CSV, eight-model Test prediction table, pooled and per-person Test metrics, Validation-only perturbation table, and four assessment figures (nine current stage-owned files).

Study outputs can now be staged after review. `.gitignore` cannot apply size rules; the separate Git policy blocks unapproved blobs larger than **50,000,000 bytes**. Raw source folders remain ignored. ZIPs generated in `/content/` are not automatically copied into Git. The original GazeBase CSVs already tracked in the public repository retain their original CC BY 4.0 attribution. Stages reuse files in one Colab Runtime or explicitly upload the matching stage ZIP in `/content/` for `ensure_s2_stage_inputs`; uploaded ZIP files supersede older tracked pilot stage outputs. The main-cohort gate stops an unintentional 10-person read.

The ten-person pilot execution remains preserved in earlier commits. Notebook 01 has now been **rerun and committed with its actual 100-person execution outputs and counters**; Notebooks 02–05 are still unexecuted on the new cohort and their saved model artefacts remain historical pilot evidence. No source-data re-execution or model fitting was performed as part of this documentation review. Stage-01 through stage-04 output files are present in the GitHub tree, and saved Notebook 04 prediction CSVs and metrics were examined alongside recorded notebook output. This does not reproduce training or independently verify the user-downloaded ZIP archive. The latest committed Notebook 02 shows execution counts 1–9 and reports a successful 14-file ZIP package. CNN, TCN, GRU and Transformer execution is recorded. Notebook 05 has recorded Validation-based selection and pilot Test metrics. Saved numerical tables were checked for schema and internal consistency, but no full re-execution of the original-data training/evaluation pipeline was performed here.

## 9. Discussion, Limitations and Future Work

The implemented workflow is designed to compare simple feature-based models and compact temporal neural networks with participant-disjoint partitions, separate per-model outputs and explicit Test boundaries. Its scientific interpretation will remain narrow: **classification of the original recording task**. Differences in stimulus layout and recording duration, even when duration itself is excluded as a predictor, may drive Task identity. These scores would not demonstrate neurocognitive insight.

The ten-person modelling run remains a historical feasibility exercise. The **100-person source cohort and participant-disjoint split have now been executed and recorded**; the next dependency is classical model training and Validation in Notebook 02, followed by newly fitted temporal models and a later Validation-selected main Test. The initial ten-person pilot is not an independent external validation. Final-model selection must remain Validation-based, with the main Test held out until the design is frozen.

Notebook 01–05 evidence now supports **closure of the ten-person pilot execution sequence**, preserving all recorded outputs and their limitations. The new main data-preparation stage is implemented and executed, but **the 100-person study is not complete** until models are fitted, compared on current Validation records and evaluated against the reserved main Test. The named stage-01 through stage-04 outputs are tracked in the current GitHub tree; downloaded ZIP files have not been independently verified. For future work, retain the complete stage-owned outputs and provenance. Carry forward the fixed 100-person source IDs, participant-disjoint Test assignment and nine current stage-01 outputs. Inspect source validity and task/stimulus confounding in the development evidence, then freeze the current Validation-based selection before opening the reserved main Test. Do **not** treat either pilot Test person as untouched if pilot feedback informs subsequent choices; retrain and regenerate all dependent outputs after changing the cohort.

**Source attribution:** [GazeBase dataset](https://doi.org/10.6084/m9.figshare.12912257) and [Griffith, Lohr, Abdulin and Komogortsev (2021)](https://doi.org/10.1038/s41597-021-00959-y), [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
