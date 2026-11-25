# Reading Comprehension & Neurocognitive Modelling | Research Methods and Evidence

**Study:** Independent OneStop paragraph-reading application within Eye-Tracking & Neurocognitive Intelligence. **Target:** correctness of a subsequent comprehension response. The implementation is available in five notebooks; original-data results have not been recorded. See the [study README](README.md) and [study-wide workflow](images/study_overview.svg).

## 1. Research Objective and Scope

The main question is whether paragraph-phase eye movements improve prediction of a subsequent correct or incorrect response relative to text-only and Training-prevalence comparators. Three associated questions concern descriptive gaze–correctness relationships, the value of ordered fixation sequences, and generalisation to held-out people and articles.

The predictor observes the presented paragraph and gaze up to the end of reading, before the question/answer period. It does not receive the subsequent question text, answer options, selected answer, answer positions, feedback, question-specific span annotations or overall session outcomes. The experiment is therefore a paragraph-conditioned prediction of the recorded response to the assigned question, not a question-conditioned answer-selection model.

The study is scientifically independent of GazeBase, ZuCo and MPIIGaze. It does not infer latent cognitive ability, diagnose a disorder or model EEG.

## 2. Data Sources, Cohort and Target Definitions

### 2.1 Original source and access

OneStop is described in Berzak et al. (2025), [Scientific Data](https://doi.org/10.1038/s41597-025-06272-2). The published corpus comprises 360 English L1 participants, about 152 hours of eye tracking and multiple reading regimes. Source documentation distinguishes ordinary reading, information seeking and repeated reading. Ordinary reading includes 9,718 regular trials according to the publisher; exact usable cohort counts require original-data execution.

Use the official [ordinary-reading OSF component](https://osf.io/zn9sq/) and [report schema](https://lacclab.github.io/OneStop-Eye-Movements/variables). The publisher download script identifies the fixation archive as `ne4az` and the interest-area archive as `xkgfz`. The local adapter downloads and extracts the two report CSVs or reuses the corresponding local files. The reports are detected as tab- or comma-separated; required documented fields are checked before use.

### 2.2 Trial identity and eligibility

One sample corresponds to one participant reading one paragraph and answering its assigned question. The pair `participant_id, trial_index` identifies the source trial; article ID is qualified by article batch, paragraph by article and question by paragraph. These identities support matching and grouped evaluation only.

Both reports must agree on the trial metadata. The implemented cohort retains matched ordinary first-reading, non-practice trials, valid answers A–D, consistent `is_correct` labels and at least one finite positive-duration fixation with finite start time and coordinates. The source's correctness definition is answer A; missing selected answers are excluded even if an annotation is present. Exclusions are counted once using ordered, mutually exclusive reasons.

The source documentation reports missing trials/answers, some typographical and display-layout issues, and participant-ID qualifications in its [known issues](https://lacclab.github.io/OneStop-Eye-Movements/known_issues.html). These do not justify silently correcting source data or treating the reported corpus size as an assertion about available rows.

### 2.3 Outcome and information boundary

The binary label is source `is_correct` (0 incorrect, 1 correct). Selected answer is inspected only to establish eligibility and label consistency; it is not exported into modelling tables. Question identity is retained for provenance, not used as a model feature. No participant identifier, article identifier, trial position, question identity or post-answer field enters a prediction model.

## 3. Data Preparation and Representations

### 3.1 Report reconciliation

[Notebook 01](code/notebooks/01_onestop_dataset_and_protocol.ipynb) produces the cohort and fixed participant assignments. [Notebook 02](code/notebooks/02_reading_behaviour_and_statistics.ipynb) checks the current source cohort against the saved trial index before feature extraction. The source adapter and feature module are [onestop_io.py](code/src/onestop_io.py) and [reading_features.py](code/src/reading_features.py). Original reports are not tracked.

### 3.2 Trial-level representations

| Family | Model inputs and interpretation |
| --- | --- |
| Fixation timing | Count, total fixation duration, mean and SD of duration; milliseconds, not total elapsed reading time |
| Spatial | SD of horizontal/vertical fixation coordinates and sum of Euclidean steps between valid fixations in screen pixels |
| Reading transitions | Proportion of backward word-index transitions among finite nonzero changes; not independently classified regressive saccades |
| Interest areas | Fraction of areas with nonzero reported fixation count and mean word dwell time (ms) |
| Static paragraph | Whitespace-token count, mean token length, punctuation fraction and Advanced-versus-Elementary version |
| Lexical baseline | Paragraph unigram/bigram TF-IDF, vocabulary fitted on Training only |

Interest-area IDs and fixation word indices are used for the transition and coverage summaries. Invalid numeric values are treated as missing; missing scalar model features are imputed on Training only. The source word frequency and GPT-2 surprisal annotations are summarised descriptively when the IA count matches paragraph token count; they are not model predictors because segmentation and report support may differ between trials. No dwell-weighted lexical features enter the text-only baseline.

### 3.3 Ordered fixation sequences

Valid fixations are sorted by source start time and represented by duration, x, y and word-area index. Unmapped indices use −1. At most the first 256 valid fixations are retained per trial, with a Boolean mask and explicit truncation indicator; padding is excluded from sequence scaling and pooling. Trial IDs are stored in the local NPZ and checked against the feature table. Scalar features summarise the full valid fixation sequence rather than the truncated input.

## 4. Experimental Design and Evaluation

### 4.1 Fixed participant protocols

| Protocol | Training and Validation | Outer Test |
| --- | --- | --- |
| A — Unseen people | Approximately 60% / 20% of eligible people within each source article batch, seed 17 | Approximately 20% of people in each batch; no participant overlap |
| B — Unseen batch | Disjoint Training and Validation people in the first two ordered batches (Validation approximately 20% within each) | All eligible participants and articles from the final ordered batch |

Splits use participant identity and batch, not labels for balancing or performance. Every trial of a participant shares one role per protocol. Both outcome classes must be present in each role. Protocol B also requires article disjointness between development and Test. The two protocols are separate analyses; they can share source people across their different roles and must not be pooled.

### 4.2 Fitting and selection

Preprocessing, TF-IDF vocabulary, class weights and model fitting are Training-only. Neural checkpoint selection uses Validation Macro F1, with earliest best epoch retained. Classical configurations are fixed. All scores use threshold 0.5. Notebook 05 fixes a primary candidate per protocol using Validation Macro F1, with alphabetical tie-breaking, before calculating Test metrics. Other Test results are descriptive secondary comparisons and do not change the selected candidate.

### 4.3 Metrics

Macro F1 over labels 0 and 1 is primary. Report accuracy, balanced accuracy, AUROC, incorrect-response AUPRC using 1 − score, incorrect-class precision/recall and Brier score. Report participant counts and trial support alongside aggregate scores. Undefined single-class AUROC, AUPRC and balanced accuracy remain missing; fixed-label Macro F1 remains defined with zero for undefined class-specific F1. Class weighting can distort probability calibration, so model outputs are treated as scores.

## 5. Methods, Models and Configurations

### 5.1 Exploratory statistical analysis

[reading_statistics.py](code/src/reading_statistics.py) fits separate approximate Bayesian logistic mixed-effects models on Protocol A Training data. Each model uses one standardised gaze feature and difficulty as fixed effects, with crossed participant and paragraph random intercepts. Variational posterior means, SDs, approximate 95% intervals and optimiser status are saved. Feature-wise complete cases and repeated tests limit interpretation; these are exploratory associations, not causal or confirmatory findings.

### 5.2 Classical predictors

| Candidate | Configuration |
| --- | --- |
| Prior | Unweighted Training prevalence of correct responses |
| Logistic Regression | C = 1, maximum 1,000 iterations |
| Random Forest | 200 trees, minimum leaf size 5 |
| XGBoost | 150 trees, depth 3, learning rate 0.05 |
| TF-IDF Logistic Regression | Unigram/bigram, minimum document frequency 2, maximum 5,000 features, C = 1 |

Each scalar estimator is compared on gaze-only, text-only and combined input allowlists. TF-IDF is compared with static paragraph features alone and with added gaze features. Median imputation, standardisation and inverse-frequency sample weighting are fitted on Training only. Scalar-model permutation importance uses Validation Macro F1 decrease (three repeats, seed 17), without treating importance as a causal effect.

### 5.3 Sequence models and fusion

[reading_models.py](code/src/reading_models.py) defines three compact gaze encoders: masked 1D CNN (two convolutional layers, 32 channels, kernels 5 and 3), GRU (hidden size 32) and two-layer Transformer encoder (embedding size 32, four heads, feed-forward size 64, learned positions). Valid-position mean pooling produces a fixed representation. The fusion variants concatenate the gaze representation with Training-standardised static paragraph features; a text-only MLP supplies a neural comparator.

All neural candidates use a 32-unit hidden prediction head, dropout 0.1, AdamW (learning rate 0.001, weight decay 0.0001), batch 64, maximum 15 epochs and patience three. The positive-class loss weight is Training negative/positive support. Seed 17 is used for initialisation and data loading. GPU is recommended; actual resource demand and software versions await execution. CUDA results are not claimed to be bitwise deterministic.

## 6. Results and Comparative Analysis

### 6.1 Data and cohort evidence

No original OneStop report has been downloaded or decoded during implementation here. Notebook 01 will establish actual report dimensions, exclusions, retained trial counts and outcome support per participant protocol.

### 6.2 Feature and association evidence

Notebook 02 will report feature support, truncation counts, Training-only participant summaries and exploratory mixed-effects estimates. No direction or magnitude of association is assumed.

### 6.3 Prediction results

The candidate list contains 12 classical and lexical procedures (prior, nine scalar variants and two TF-IDF variants) and seven neural procedures (three gaze-only, three fusion and one text-only MLP). Each protocol uses the same eligible trial identities across all candidates. Notebook 05 will report Validation selection and held-out metrics for the fixed set. No performance, model ranking, improvement or generalisation result has been recorded yet.

## 7. Robustness, Uncertainty and Additional Analyses

### 7.1 Participant and article generalisation

Participant metrics and difficulty strata describe variation in the held-out cohort. Protocol B changes both participant group and article batch; a difference between protocols is not an isolated unseen-text effect. The source difficulty manipulation may be associated with layout, wording and question difficulty.

### 7.2 Conditional uncertainty and modality contrasts

[reading_evaluation.py](code/src/reading_evaluation.py) resamples participant-level confusion counts jointly across candidate scores (1,000 draws, seed 17), reporting descriptive percentile intervals for pooled Macro F1. Fusion-versus-text and fusion-versus-gaze contrasts are matched within scalar model families; TF-IDF fusion versus TF-IDF text is also included. The fitted models are not retrained and the bootstrap does not resample articles. These intervals do not estimate full training, article-population or external uncertainty.

### 7.3 Calibration and interpretation

Fixed ten-bin score summaries compare mean score and observed correct-response fraction; Brier score is recorded. No probability recalibration or Test-driven threshold optimisation is performed. Permutation importance and mixed-effects estimates have different interpretations and do not establish that changes in gaze cause changes in comprehension.

## 8. Current Outputs and Reproducibility

| Producer | Named current output files after execution |
| --- | --- |
| Notebook 01 | `01_source_inventory.csv`, `01_trial_exclusions.csv`, `01_trial_index.csv`, `01_participant_splits.csv`, `01_split_support.csv`, `01_cohort_support.png` |
| Notebook 02 | `02_trial_features.csv`, `02_training_behaviour.csv`, `02_training_associations.csv`, `02_reading_behaviour.png`, `02_reading_associations.png`; local `02_fixation_sequences.npz` and `02_paragraph_text.csv` |
| Notebook 03 | `03_classical_predictions.csv`, `03_validation_metrics.csv`, `03_validation_feature_importance.csv`, `03_validation_comparison.png` |
| Notebook 04 | `04_neural_predictions.csv`, `04_training_history.csv`, `04_model_configurations.csv`, `04_validation_metrics.csv`, `04_validation_curves.png` |
| Notebook 05 | `05_validation_metrics.csv`, `05_selected_models.csv`, `05_test_metrics.csv`, `05_participant_metrics.csv`, `05_difficulty_metrics.csv`, `05_calibration_bins.csv`, `05_participant_intervals.csv`, `05_paired_modality_intervals.csv`, `05_test_comparison.png`, `05_selected_confusion.png`, `05_score_calibration.png` |

The [requirements](requirements.txt) specify the initial compatible dependency range. Actual runtime versions, data availability and run outputs must be confirmed in Colab. [Synthetic tests](code/tests/test_reading_pipeline.py) exercise the cohort and evaluation contracts; they do not validate the original report or the scientific prediction task. Raw files and reconstruction caches are kept under the repository's ignored data directories. Notebook 05 reads the derived CSVs only.

## 9. Discussion, Limitations and Future Work

This study asks a bounded behavioural prediction question. Answer correctness depends on the assigned question, text, participant knowledge and other factors in addition to gaze. It cannot be equated with general comprehension ability, critical thinking, attention state or clinical status.

The source dataset includes multiple reading regimes, but the principal experiment is ordinary first reading. The single held-out batch analysis is internal, with participant/text confounding. Fixation reports are vendor-derived rather than independently adjudicated eye events; backward word-index changes are a proxy rather than validated regressions. Text-based feature summaries do not exhaust all available linguistic information. Sequence truncation may affect longer trials. Training-only associations are approximate and are not causal.

Potential later extensions include explicitly question-conditioned modelling with a separate information boundary, source-aligned lexical annotations, reading-goal and repeated-reading studies, independent text/person datasets and a separately justified EEG–gaze study. These are not current results.

**References and licensing:** [OneStop paper](https://doi.org/10.1038/s41597-025-06272-2); [official release](https://lacclab.github.io/OneStop-Eye-Movements/); [data](https://osf.io/2prdq/); [Shubi et al. (2024)](https://aclanthology.org/2024.emnlp-main.198/). Eye-tracking data use CC BY 4.0; underlying OneStopQA materials and auxiliary annotations use CC BY-SA 4.0. The local paragraph-text cache is not published. Derived data remain subject to applicable source terms.
