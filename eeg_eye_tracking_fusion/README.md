# EEG–Eye-Tracking Multimodal Research | ZuCo 2.0

![Six-stage EEG–eye-tracking workflow with frozen model selection and held-out Test evaluation](images/study_overview.svg)

*Figure 1. The six-stage ZuCo 2.0 study, from 36 source runs through paired features, model comparison and independent final Test. Gaze Random Forest was selected by Validation before it was scored on the four-person Test.*

**Aim:** Compare EEG-only, gaze-only and EEG–gaze fusion models for the original **Normal Reading (NR)** versus **Task-Specific Reading (TSR)** labels using participant-disjoint data. The six notebooks executed on **18 people, 36 source runs and 630 paired windows**. Ten predefined candidates were ranked on three-person Validation; **Gaze Random Forest** was frozen at Validation Macro F1 **0.6244** and then scored on four independent Test participants, achieving **Macro F1 0.4997** and **Accuracy 0.5250**.

**Independence:** This is a separate ZuCo 2.0 study within the repository. Reusable infrastructure may be shared, but no GazeBase measurements, participant splits, model results or processed features are imported from Eye-Tracking Foundations or Eye-Movement Modelling.

**Evidence:** The 18-person experiment produces 49 derived files, including final Test results from 160 windows and four participants. Test evaluation follows Validation-based model selection, without refitting or reselection.

## 1. Dataset and Experimental Protocol

| Item | Implemented definition and boundary |
| --- | --- |
| Dataset | [ZuCo 2.0](https://www.nemar.org/dataset/nm000279), NEMAR `nm000279` release `v1.0.0`; 18 eligible participants with original NR/TSR Run-1 EEG and EyeLink files |
| Source recordings | **36 runs**, one NR and one TSR run per person; 36/36 recorded source packages, approximately **3,373.4 MB** retrieved |
| Measurement metadata | Original EEG sidecars report **128 channels**; nominal EEG and gaze sampling **500 Hz**; eye `StartTime` offsets **−0.046 to +0.033 s**. Model features use **16 actual EEG channels** |
| Prediction target | BIDS reading-task identity: **NR versus TSR**. Not comprehension, measured cognition, diagnosis or clinical status |
| Analysis unit | A **12-second**, non-overlapping paired EEG/gaze window; up to **20 windows per source run**, with six consecutive **two-second** feature bins |
| Window eligibility | BIDS-relative EEG/gaze overlap and at least **0.60 finite gaze x/y fraction**. This is availability, not independently verified clock synchronisation or calibration |
| Classical predictors | Four EEG means: theta/alpha/beta log bandpower and RMS; four gaze means: finite-coordinate fraction, median pixel speed, median pupil, vendor fixation-overlap fraction |
| Temporal predictors | One EEG and one gaze tensor, each **(630, 6, 4)**, keyed to identical window IDs; model fitting and scaling use Train only |
| Group separation | All runs and windows for one person remain in exactly one Train, Validation or Test partition; IDs, paths, task and role are never predictors |
| Frozen partition | Seed **17**, retaining the six previously assigned participant roles: **11 Train people / 371 windows**, **3 Validation people / 99 windows**, **4 reserved Test people / 160 windows** |
| Model selection | Fixed-label **Validation Macro F1**, then Balanced Accuracy, then predefined candidate priority; selection recorded in `06_model_selection.csv` |
| Final evaluation | **`RUN_FINAL_TEST=True`** for the recorded final run; **160 Test windows from four independent people**, primary Macro F1 **0.4997**, Accuracy and Balanced Accuracy **0.5250**; not external validation |

NR and TSR differ in source instructions and may differ in stimulus/text characteristics. Task classification cannot by itself isolate cognitive processing. BIDS timebase alignment does not independently verify EEG–EyeLink clock drift or gaze calibration. A cohort or feature-definition change requires reviewing downstream compatibility before reusing current model scores.

## 2. Experimental Programme

The six notebooks form **one** multimodal experiment, not six separate studies. Notebooks 01–05 produced the frozen development data and models; Notebook 06 completed the Validation selection and the independently held-out final Test without changing the selected candidate.

| Notebook | Main experiment and observed evidence | Owned current outputs |
| --- | --- | --- |
| [01 — Source Audit and Protocol](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) | **18 people; 36 NR/TSR Run-1 source packages**; frozen 11/3/4 participant roles and acquisition metadata | Seven `01_*` files: source manifest, eligibility, role assignments, metadata, protocol, download inventory and overview |
| [02 — Alignment and Quality](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) | **630 eligible paired 12-s windows** across 36 runs; 371/99/160 Train/Validation/Test windows; BIDS-relative overlap, unverified drift | Five `02_*` files: window registry, alignment and person/task quality, summary and plot |
| [03 — Features and Neurophysiology](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) | **630 aligned rows**, eight finite mean predictors, two **(630, 6, 4)** arrays, zero extraction exclusions; exploratory raw fixation diagnostic | Seven `03_*` files: feature table, temporal NPZ, quality/exclusion tables and two plots |
| [04 — Classical Baselines and Fusion](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) | Seven candidates, **693 common-window Validation predictions**; best classical model is Gaze Random Forest (Macro F1 **0.6244**) | Twelve `04_*` files: six fitted pipelines, fusion weights, metrics/predictions and two plots |
| [05 — Temporal and Advanced Fusion](code/notebooks/05_temporal_and_advanced_fusion.ipynb) | Three GRUs, **297 Validation predictions**; Gated Fusion GRU leads this family (Macro F1 **0.6107**) | Nine `05_*` files: three checkpoints, Train scaling, histories, predictions, metrics and curves |
| [06 — EEG–Eye-Tracking Model Selection](code/notebooks/06_robustness_and_final_evaluation.ipynb) | **990 Validation and 1,600 Test predictions**; frozen Gaze Random Forest selection, four-person Test Macro F1 **0.4997**, person-level metrics and clustered uncertainty | **Nine** `06_*` files: four development and five final Test outputs |

One editable [study-wide workflow](images/study_overview.svg) appears above. Each notebook also has its **own** stage-specific scientific illustration in [`images/`](images/), retained beside the relevant notebook content rather than repeatedly embedded in this README.

## 3. Main Results

**Source-to-feature evidence (Notebooks 01–03):** All 18 eligible people contributed both source task runs. The 36 runs yielded **630 paired windows**: 341 NR and 289 TSR. Of these, **371** belong to Train (201 NR, 170 TSR), **99** to Validation (60 NR, 39 TSR), and **160** to reserved Test (80 NR, 80 TSR). Every eligible window has a corresponding EEG/gaze feature row; there were **zero feature-extraction exclusions**. The `gaze_valid_fraction` feature is constant on Train, so it supplies no within-Train discrimination.

**Classical and temporal Validation (Notebooks 04–05):** All ten candidates were compared against the same **99 windows from three independent Validation people**, after fitting on **371 Train windows from 11 different people**. The recorded results are development scores, not final Test estimates.

| Candidate | Validation Macro F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| **Gaze Random Forest (selected)** | **0.6244** | **0.6231** | **0.6465** |
| Gated Fusion GRU | 0.6107 | 0.6096 | 0.6465 |
| Gaze GRU | 0.5938 | 0.5929 | 0.6263 |
| EEG GRU | 0.5938 | 0.5929 | 0.6263 |
| Equal-weight Late Fusion | 0.5643 | 0.5641 | 0.5859 |
| Gaze Logistic Regression | 0.5472 | 0.5474 | 0.5657 |
| EEG Random Forest | 0.5139 | 0.5282 | 0.5152 |
| EEG Logistic Regression | 0.4545 | 0.4737 | 0.4545 |
| Early-Fusion Logistic Regression | 0.4518 | 0.4603 | 0.4545 |
| Class Prior | 0.3774 | 0.5000 | 0.6061 |

**Selection and person-level variation (Notebook 06):** The predefined Validation rule selected **`gaze_forest`** over Gated Fusion GRU by **0.0137 Macro F1**. The selected model correctly classified **44/60 NR** and **20/39 TSR** Validation windows. Its person-level Macro F1 was **0.7654** (sub-YDR, 31 windows), **0.6306** (sub-YRH, 38) and **0.3478** (sub-YTL, 30); pooled scores conceal this heterogeneity. These three independent participants are insufficient for a strong population ranking.

**Secondary masking check:** For the fixed early-fusion logistic model, Validation Macro F1 was **0.4518** with both modalities, **0.5343** with EEG masked and **0.4907** with gaze masked. The Train-fitted imputer was unchanged and the model was not retrained. This describes sensitivity to modified inputs, not causal modality importance or a reason to revise the selected model.

**Held-out Test (Notebook 06):** The frozen Gaze Random Forest achieved **Macro F1 0.4997**, **Balanced Accuracy 0.5250** and **Accuracy 0.5250** on **160 windows from four Test people** (80 NR, 80 TSR). Its confusion matrix is **60/80 NR correctly classified** and **24/80 TSR correctly classified**. Per-person Macro F1 was **0.5238** (`sub-YAC`), **0.3333** (`sub-YAG`), **0.4505** (`sub-YMD`) and **0.3333** (`sub-YRP`). A 500-resample participant-cluster bootstrap gives a descriptive **95% percentile interval of 0.3333–0.5616** for Macro F1; four clusters provide limited precision.

**Secondary Test comparisons:** Equal-weight Late Fusion scored **0.5204** and Gated Fusion GRU **0.5157** Macro F1. These are descriptive findings from the same Test windows, **not** replacements for the Validation-selected gaze forest. The full [Test metrics](output/06_test_metrics.csv), [predictions](output/06_test_predictions.csv) and [primary confusion plot](output/06_primary_test_confusion.png) are saved. See [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md) for interpretation.

## 4. Running and Reviewing the Study

The six notebooks and saved results support inspection of the complete study without source acquisition or model retraining.

1. Start with [Notebook 06](code/notebooks/06_robustness_and_final_evaluation.ipynb), the [frozen model decision](output/06_model_selection.csv), [Validation evidence](output/06_validation_evidence.png) and [held-out Test metrics](output/06_test_metrics.csv). The selected primary was not changed after Test.
2. Inspect [`output/`](output/) for the 630-window registry, paired features, matched Validation predictions, **1,600 Test predictions**, participant-level Test metrics and the primary confusion matrix.
3. For original-data reproduction, follow [Notebooks 01–06](code/notebooks/) in numerical order, starting from the source release and frozen participant assignments. Use the study [requirements](requirements.txt) and the existing Colab `setup_colab("eeg_eye_tracking_fusion")` workflow. Raw ZuCo files are **not** bundled with Git; Colab source caches can be ephemeral. Rerunning a stage replaces its stage-owned current outputs and may require rebuilding downstream stages.
4. Review the [research methods](RESEARCH_METHODS_AND_EVIDENCE.md) for source access, final model selection, participant-level results and scientific limitations. The final scoring used the saved models and Train-fitted preprocessing without refitting.

**Saved-model compatibility:** The Stage-04 `.joblib` pipelines require **scikit-learn 1.6.1**. Install the [study requirements](requirements.txt) before importing the model libraries. If necessary, restart the Python kernel after installation. The [Test metrics](output/06_test_metrics.csv) and [Notebook 06](code/notebooks/06_robustness_and_final_evaluation.ipynb) provide the final evaluation evidence.

A new cohort, timing correction, feature set or evaluation objective is a separate scientific decision; results must not be silently mixed with the current 18-person experiment.

## 5. Files and Saved Outputs

| Location | Purpose |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Six structured Jupyter notebooks, source audit through held-out Test evaluation |
| [`code/src/`](code/src/) | Source reader, alignment, features, models and the [final-test execution adapter](code/src/run_final_test.py) |
| [`images/study_overview.svg`](images/study_overview.svg) | One editable six-stage overview diagram for this README |
| [`images/`](images/) | Six individual editable stage figures used in their notebooks |
| [`output/`](output/) | **49 current outputs**: seven `01_*`, five `02_*`, seven `03_*`, twelve `04_*`, nine `05_*` and nine `06_*` |
| [`requirements.txt`](requirements.txt) | Study dependencies, including scikit-learn 1.6.1 compatibility with stored pipelines |
| [`tests/test_eeg_eye_tracking_fusion.py`](../tests/test_eeg_eye_tracking_fusion.py) | Synthetic research-contract checks; not original-data reproduction |
| [`README.md`](README.md) | Concise study entry point and results |
| [`RESEARCH_METHODS_AND_EVIDENCE.md`](RESEARCH_METHODS_AND_EVIDENCE.md) | Full protocol, model settings, saved evidence and interpretation |

Original BrainVision EEG and EyeLink recordings are available from the [ZuCo source](https://www.nemar.org/dataset/nm000279). The final evaluation uses saved features and fitted models.

## 6. Interpretation and Scope

**The six-stage ZuCo 2.0 study now includes final held-out Test results.** Gaze Random Forest was selected from ten candidates on three-person Validation (Macro F1 **0.6244**); on four independent Test people it achieved Macro F1 **0.4997** and Balanced Accuracy **0.5250**. The Test score is lower than the development score. The best fusion candidate on Validation, Gated Fusion GRU (**0.6107**), did not exceed the gaze forest there. Its secondary Test Macro F1 was **0.5157**, compared with **0.5204** for Late Fusion; neither Test ranking changes the original model choice.

**Cohort and task-design limits:** The 18-person cohort was divided into 11 Train, three Validation and four Test people. Multiple windows from the same person are correlated, so 160 Test windows do not represent 160 independent participants; run lengths also differ. NR/TSR are original reading-task labels, and possible differences in instructions or text prevent attributing classification uniquely to comprehension or a cognitive process.

**Measurement and evaluation limits:** EEG–gaze windows were aligned using BIDS timing metadata and nominal sample rates, without an independent clock-drift measurement. Finite gaze coordinates establish availability, not independently verified calibration accuracy. Ocular EEG artefacts remain possible, and the raw fixation-locked diagnostic has an uncorrected offset; it is **not** a validated fixation-related potential. The Test was executed on independent people **within ZuCo 2.0**, not on an external, cross-session, prospective or clinical cohort. These are documented limits on interpretation, **not missing execution stages**; their causes and consequences are detailed in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md), Section 7.2.

**Source attribution:** Hollenstein, Tröndle, Zhang and Langer (2020), [*ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation*](https://aclanthology.org/2020.lrec-1.18/); [NEMAR dataset DOI](https://doi.org/10.82901/nemar.nm000279). The original dataset reports [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) terms; repository code has its separate licence.
