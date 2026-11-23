# EEG–Eye-Tracking Multimodal Research | ZuCo 2.0

![Six-stage ZuCo 2.0 EEG and eye-tracking study workflow, paired-window preparation, model comparison and reserved Test boundary](images/study_overview.svg)

*Figure 1. One six-stage workflow for the 18-person ZuCo 2.0 study: source audit, EEG–gaze alignment, feature construction, classical and temporal comparisons, and Validation-only model selection. The selected Gaze Random Forest precedes any final Test evaluation; the four Test participants remain unscored.*

**Aim:** Determine whether combining EEG and gaze improves discrimination of source-defined **Normal Reading (NR)** and **Task-Specific Reading (TSR)** conditions compared with the strongest EEG-only or gaze-only model, using matched windows and participant-disjoint development data. **Notebooks 01–06 have completed the 18-person Validation-development workflow**: 36 source runs, 630 paired windows and ten fixed candidates. **Gaze Random Forest** was selected on 99 Validation windows from three people (Macro F1 **0.6244**, Balanced Accuracy **0.6231**, Accuracy **0.6465**). No Test performance or improvement from fusion over the strongest unimodal candidate has been established.

**Independence:** This is a separate ZuCo 2.0 study within the repository. Reusable infrastructure may be shared, but no GazeBase measurements, participant splits, model results or processed features are imported from Eye-Tracking Foundations or Eye-Movement Modelling.

**Current evidence:** The 44 committed `01_*`–`06_*` outputs belong to the same expanded 18-person experiment. The historical six-person pilot is superseded as the current evaluation. The frozen Test partition contains **four other people and 160 windows**, none scored in this development run.

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
| Frozen partition | Seed **17**, retaining the six original pilot roles: **11 Train people / 371 windows**, **3 Validation people / 99 windows**, **4 reserved Test people / 160 windows** |
| Model selection | Fixed-label **Validation Macro F1**, then Balanced Accuracy, then predefined candidate priority; selection recorded in `06_model_selection.csv` |
| Final evaluation | `RUN_FINAL_TEST=False`. The selected configuration is saved, but there are **no Test predictions, Test metrics or independent external results** |

NR and TSR differ in source instructions and may differ in stimulus/text characteristics. Task classification cannot by itself isolate cognitive processing. BIDS timebase alignment does not independently verify EEG–EyeLink clock drift or gaze calibration. A cohort or feature-definition change requires reviewing downstream compatibility before reusing current model scores.

## 2. Experimental Programme

The six notebooks form **one** multimodal experiment, not six separate studies. All six executed and produced the current Validation-development evidence; the final held-out Test is a separate, presently unexecuted step.

| Notebook | Main experiment and observed evidence | Owned current outputs |
| --- | --- | --- |
| [01 — Source Audit and Protocol](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) | **18 people; 36 NR/TSR Run-1 source packages**; frozen 11/3/4 participant roles and acquisition metadata | Seven `01_*` files: source manifest, eligibility, role assignments, metadata, protocol, download inventory and overview |
| [02 — Alignment and Quality](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) | **630 eligible paired 12-s windows** across 36 runs; 371/99/160 Train/Validation/Test windows; BIDS-relative overlap, unverified drift | Five `02_*` files: window registry, alignment and person/task quality, summary and plot |
| [03 — Features and Neurophysiology](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) | **630 aligned rows**, eight finite mean predictors, two **(630, 6, 4)** arrays, zero extraction exclusions; exploratory raw fixation diagnostic | Seven `03_*` files: feature table, temporal NPZ, quality/exclusion tables and two plots |
| [04 — Classical Baselines and Fusion](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) | Seven candidates, **693 common-window Validation predictions**; best classical model is Gaze Random Forest (Macro F1 **0.6244**) | Twelve `04_*` files: six fitted pipelines, fusion weights, metrics/predictions and two plots |
| [05 — Temporal and Advanced Fusion](code/notebooks/05_temporal_and_advanced_fusion.ipynb) | Three GRUs, **297 Validation predictions**; Gated Fusion GRU leads this family (Macro F1 **0.6107**) | Nine `05_*` files: three checkpoints, Train scaling, histories, predictions, metrics and curves |
| [06 — EEG–Eye-Tracking Model Selection](code/notebooks/06_robustness_and_final_evaluation.ipynb) | **990 matched predictions across ten candidates**; frozen Gaze Random Forest choice, 30 model/person rows and fixed early-fusion masking; **Test not run** | Four `06_*` files: selection, person metrics, sensitivity and Validation evidence plot |

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

No held-out Test score is available; final model generalisation remains unmeasured. Detailed calculations and qualifications are in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

## 4. Running and Reviewing the Study

**The current six-notebook Validation-development run is already saved.** Reviewing its evidence does not require re-downloading ZuCo or rerunning model fitting.

1. Start with the [completed Notebook 06](code/notebooks/06_robustness_and_final_evaluation.ipynb), [saved model decision](output/06_model_selection.csv) and [Validation evidence](output/06_validation_evidence.png). The selection is deliberately independent of Test.
2. Inspect the [`output/` directory](output/) for the 630-window registry, paired features, ten models' Validation predictions and current quality/sensitivity results.
3. For original-data reproduction, follow [Notebooks 01–06](code/notebooks/) in numerical order, starting from the source release and frozen participant assignments. Use the study [requirements](requirements.txt) and the existing Colab `setup_colab("eeg_eye_tracking_fusion")` workflow. Raw ZuCo files are **not** bundled with Git; Colab source caches can be ephemeral. Rerunning a stage replaces its stage-owned current outputs and may require rebuilding downstream stages.
4. Review the [research methods](RESEARCH_METHODS_AND_EVIDENCE.md) for source access, representations, model settings and evaluation limits. The recorded run is **Validation-only**, with final Test scoring disabled.

A new cohort, timing correction, feature set or evaluation objective is a separate scientific decision; results must not be silently mixed with the current 18-person experiment.

## 5. Files and Saved Outputs

| Location | Purpose |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Six structured Jupyter notebooks, source audit through Validation selection |
| [`code/src/`](code/src/) | ZuCo reader, alignment, EEG/gaze features, classical and temporal model functions |
| [`images/study_overview.svg`](images/study_overview.svg) | One editable six-stage overview diagram for this README |
| [`images/`](images/) | Six individual editable stage figures used in their notebooks |
| [`output/`](output/) | **44 current outputs**: seven `01_*`, five `02_*`, seven `03_*`, twelve `04_*`, nine `05_*` and four `06_*` |
| [`requirements.txt`](requirements.txt) | Study-specific dependencies |
| [`tests/test_eeg_eye_tracking_fusion.py`](../tests/test_eeg_eye_tracking_fusion.py) | Synthetic research-contract checks; not original-data reproduction |
| [`README.md`](README.md) | Concise study entry point and results |
| [`RESEARCH_METHODS_AND_EVIDENCE.md`](RESEARCH_METHODS_AND_EVIDENCE.md) | Full protocol, model settings, saved evidence and interpretation |

Original BrainVision EEG/EyeLink recordings remain outside ordinary Git history. Output files hold the compatible 18-person experiment; the earlier six-person pilot is historical Git evidence, not an alternative active output directory.

## 6. Interpretation and Scope

**The six-stage ZuCo 2.0 Validation-development workflow is complete, but the study's reserved final Test has not been scored.** Across ten fixed candidates and 99 matched Validation windows from three people, **Gaze Random Forest** was the prespecified development selection (Macro F1 **0.6244**). Gated Fusion GRU was the strongest fusion candidate (**0.6107**). Thus the current evidence **does not show a Validation advantage for EEG–gaze fusion over the strongest gaze-only model**. It does not establish that fusion is harmful or that gaze will dominate on new people.

Interpretation is limited by three independent Validation people, repeated windows, unequal run support, possible stimulus/task differences, uncertain gaze calibration and inter-device drift, ocular artefacts, and an uncorrected raw fixation-locked EEG offset. The raw diagnostic is **not** a validated fixation-related potential or explanation of cognition. No cross-session, external, prospective or clinical generalisation has been established.

**Source attribution:** Hollenstein, Tröndle, Zhang and Langer (2020), [*ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation*](https://aclanthology.org/2020.lrec-1.18/); [NEMAR dataset DOI](https://doi.org/10.82901/nemar.nm000279). The original dataset reports [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) terms; repository code has its separate licence.
