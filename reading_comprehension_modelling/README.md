# Reading Comprehension & Neurocognitive Modelling | OneStop

![OneStop paragraph-reading study workflow](images/study_overview.svg)

*Figure 1. Source-to-evaluation workflow implemented in five notebooks. The diagrams are methodological schematics; the recorded cohort, reading-behaviour and classical Validation evidence is reported below.*

**Aim:** Examine whether pre-answer eye movements provide information about the correctness of a subsequent reading-comprehension response beyond information available from the paragraph itself.

**Dataset:** The published [OneStop Eye Movements corpus](https://lacclab.github.io/OneStop-Eye-Movements/) includes 360 participants and approximately 152 hours of recordings across reading regimes. Its ordinary-reading component has 9,718 regular trials according to the source documentation. These are publisher-reported counts, not the retained study cohort.

**Status:** Notebooks 01–04 have verified saved cohort, feature, association and classical/neural Validation evidence. The five original Notebook 04 outputs are available and reconciled. Notebook 05 has executed and all eleven outputs are present, but final evidence reconciliation remains pending because the TF-IDF metrics differ from the committed producer predictions. This study predicts the recorded correctness of a particular response, not general cognitive ability.

**Scientific account:** [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Dataset and Experimental Protocol

| Item | Implemented study definition |
| --- | --- |
| Source | [OneStop ordinary reading](https://osf.io/zn9sq/), [dataset documentation](https://lacclab.github.io/OneStop-Eye-Movements/variables) |
| Input period | Paragraph fixation and interest-area reports only |
| Eligibility | Matched first-reading trials without question preview or practice; valid A–D response and usable fixation support |
| Prediction unit | Participant × reading trial; retain article, paragraph, question and difficulty identities |
| Target | Source `is_correct`: 1 = correct, 0 = incorrect |
| Protocol A | Participant-disjoint Training/Validation/Test, approximately 60/20/20 within each article batch |
| Protocol B | Final article batch and its participants held out; remaining batches supply disjoint Training and Validation participants |
| Selection | Validation Macro F1; fixed 0.5 threshold; model-name tie-break |
| Primary metric | Test Macro F1; secondary balanced accuracy, AUROC, incorrect-class precision/recall/AUPRC and Brier score |

Answer choices, selected answers, response-phase gaze, feedback, session-level accuracy, question-specific critical/distractor spans and participant/text identity codes are not predictors. Both protocols are internal to OneStop.

## 2. Experimental Programme

| Notebook | Scientific role | Output contract |
| --- | --- | --- |
| [01 — Dataset and Protocol](code/notebooks/01_onestop_dataset_and_protocol.ipynb) | Source acquisition, trial reconciliation, eligibility and participant splits | Source inventory, exclusions, trial index, splits, support counts, cohort plot |
| [02 — Reading Behaviour and Statistics](code/notebooks/02_reading_behaviour_and_statistics.ipynb) | Pre-answer feature extraction, ordered fixation arrays, participant summaries and crossed mixed-effects analysis | Feature and association tables, two plots; local NPZ and paragraph-text table |
| [03 — Classical ML and Text Baselines](code/notebooks/03_classical_ml_and_text_baselines.ipynb) | Training prevalence, scalar models and lexical TF-IDF comparators | Prediction scores, Validation metrics, permutation importance and comparison plot |
| [04 — Sequential and Multimodal Models](code/notebooks/04_sequential_and_multimodal_models.ipynb) | CNN, GRU, Transformer and text-only MLP; gaze–text fusion | Scores, epoch histories, selected configurations, Validation metrics and curves |
| [05 — Generalisation and Final Evaluation](code/notebooks/05_generalisation_and_final_evaluation.ipynb) | Matched Validation selection, held-out comparisons, person/difficulty summaries, bootstrap and calibration | Eight CSV tables and three plots |

Notebooks 03 and 04 share the trial features and participant assignments from Notebooks 01–02; they independently produce candidate scores for Notebook 05. The six editable five-band SVGs illustrate the scientific design.

## 3. Main Results

Notebook 01 reconciled 10,078 report trial identities and retained 9,718 eligible trials from 180 participants and 30 articles; 360 trials were excluded as `practice_or_unknown`. The retained cohort contains 7,890 correct responses (81.19%) and 1,828 incorrect responses (18.81%).

| Protocol | Training (people / trials) | Validation (people / trials) | Test (people / trials) |
| --- | ---: | ---: | ---: |
| A — Unseen people | 108 / 5,831 | 36 / 1,943 | 36 / 1,944 |
| B — Unseen batch | 96 / 5,183 | 24 / 1,295 | 60 / 3,240 |

Saved trial identities and participant assignments reproduce all class-support counts, with disjoint participants within each protocol and disjoint development/Test articles in Protocol B. See the [cohort plot](output/01_cohort_support.png), [exclusions](output/01_trial_exclusions.csv) and [split support](output/01_split_support.csv). Notebook 02 retains all 9,718 trial identities with ten gaze and four static text predictors; the recorded sequence shape is `(9718, 256, 4)` and 193 sequences (1.99%) are truncated. On Protocol A Training (5,831 trials, 108 participants, 162 paragraphs), all ten separate exploratory mixed-effects fits report convergence. Backward word-index transition rate has the largest positive posterior mean (0.137 log odds per SD; approximate 95% posterior interval 0.068 to 0.206). See the [feature table](output/02_trial_features.csv), [participant summaries](output/02_training_behaviour.csv), [association estimates](output/02_training_associations.csv) and [association plot](output/02_reading_associations.png). The estimates are exploratory, not causal or confirmatory. 

Notebook 03 recorded 101,064 matched Validation and Test scores across 12 classical candidates per protocol. Validation Macro F1 is highest for TF-IDF fusion in Protocol A (0.5582) and TF-IDF text in Protocol B (0.5698), compared with prevalence references of 0.4472 and 0.4487. Adding gaze to the TF-IDF comparator changes Macro F1 by +0.0171 in A and -0.0060 in B. The added value of gaze is not consistent across model families. See the [Validation metrics](output/03_validation_metrics.csv), [comparison plot](output/03_validation_comparison.png) and [permutation importance](output/03_validation_feature_importance.csv). Protocol B Validation is within the development article batches; final Test reporting remains pending reconciliation.

Notebook 04 records 14 neural fits and 58,954 unique, aligned prediction rows. The highest neural Validation Macro F1 is GRU gaze in A (0.5251) and Transformer fusion in B (0.5458), below the corresponding classical leaders. All 126 numeric Validation entries and 14 checkpoint configurations were reconciled against the scores, fixed partitions and 130 history rows. See the [neural Validation metrics](output/04_validation_metrics.csv), [learning curves](output/04_validation_curves.png) and [model configurations](output/04_model_configurations.csv). Both score tables now supply 160,018 scores for 19 candidates per protocol. No Test performance has been calculated in this review.

The primary comparison asks whether the gaze–text candidates improve over matched text-only comparators. The lexical TF-IDF baseline complements the simpler paragraph-statistic baseline. Participant-level and paired uncertainty analyses will distinguish observed differences from stronger generalisation claims.

## 4. Running and Reviewing the Study

1. Update the local repository with `git pull --ff-only origin main` without discarding local edits. Use the [Getting Started guide](../GETTING_STARTED.md) to open the notebook in VS Code/Colab.
2. The executed [Notebook 01](code/notebooks/01_onestop_dataset_and_protocol.ipynb) and its six current outputs are available. For reproduction, execute it on CPU; the two ordinary-reading ZIPs are obtained from OSF using the documented download IDs, or extracted local reports are reused. Do not rerun it merely to prepare Notebook 02 when its current cohort and split tables are available.
3. The executed [Notebook 02](code/notebooks/02_reading_behaviour_and_statistics.ipynb) and its five derived outputs are available. For reproduction, execute it on CPU with the same original reports and cohort. Keep its local `data/processed/onestop/02_paragraph_text.csv` and `02_fixation_sequences.npz` in the active runtime or a private durable copy. Neither is in the output ZIP or Git. Notebook 03 needs the text cache; Notebook 04 needs the sequence cache. No rerun is needed when these compatible inputs are available.
4. The executed [Notebook 03](code/notebooks/03_classical_ml_and_text_baselines.ipynb) and its four current outputs are available. The executed [Notebook 04](code/notebooks/04_sequential_and_multimodal_models.ipynb) and its five reconciled outputs are available. Preserve both original score tables for Notebook 05; neither model refitting nor the local caches are required for final evaluation.
5. [Notebook 05](code/notebooks/05_generalisation_and_final_evaluation.ipynb) has executed on CPU-compatible code and all eleven outputs were committed at `4b2aa28`. Independent recalculation matched 68 of 76 model–protocol–role combinations; TF-IDF text and fusion differ in both protocols and roles. The saved setup records a deferred Git update. Rerun only Notebook 05 from a fresh checkout using the committed score tables, then replace all eleven stage outputs together. Do not refit or change model selection rules after inspecting Test.

Use the shared output-download utility to transfer each numbered ZIP's contents to the study output directory. Keep the original reports, local text and sequence arrays outside Git. Each run writes its named current outputs. If the source or cohort changes, regenerate the downstream stages rather than combining incompatible results.

## 5. Files and Saved Outputs

| Location | Responsibility |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Five runnable research notebooks |
| [`code/src/`](code/src/) | Source IO, features, statistical models, prediction models and final evaluation |
| [`code/tests/`](code/tests/) | Synthetic cohort, feature and prediction-contract tests |
| [`images/`](images/) | Study overview and five notebook SVGs |
| [`output/`](output/) | Verified Notebook 01–04 outputs; Notebook 05 outputs present but pending fixed-input reconciliation |
| [`requirements.txt`](requirements.txt) | Study dependencies; actual runtime versions should be recorded during execution |
| `data/raw/onestop/ordinary/` (repository-local) | Source ZIPs and original paragraph reports; Git-ignored |
| `data/processed/onestop/` (repository-local) | Local sequence NPZ and paragraph text; Git-ignored |
| [`RESEARCH_METHODS_AND_EVIDENCE.md`](RESEARCH_METHODS_AND_EVIDENCE.md) | Full scientific methods and evidence boundaries |

Run the synthetic checks from the repository root with `python -m unittest discover -s reading_comprehension_modelling/code/tests -v`.

## 6. Interpretation and Scope

Answer correctness is an observed behavioural outcome for a particular question, not a direct measurement of understanding, attention or clinical cognition. The ordinary-reading restriction removes the explicit question-preview condition from the principal experiment but does not eliminate stimulus, difficulty or participant-related confounding.

Protocol A may share articles and questions between roles. Protocol B holds out one article batch together with its participants, so changes in text and reader population are not separable; one batch is not an external validation corpus. Bootstrap intervals are conditional on the fitted predictions and do not account for article clustering or training variability.

**Sources:** Berzak et al., [OneStop dataset paper, Scientific Data (2025)](https://doi.org/10.1038/s41597-025-06272-2); [OneStop release and variables](https://lacclab.github.io/OneStop-Eye-Movements/variables); Shubi et al., [Fine-Grained Prediction of Reading Comprehension from Eye Movements, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.198/). The eye-tracking data are provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the underlying OneStopQA text and auxiliary annotations are subject to [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Repository software licensing does not supersede those terms.
