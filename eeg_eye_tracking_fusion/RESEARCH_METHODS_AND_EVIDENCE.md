# EEG–Eye-Tracking Multimodal Research | Research Methods and Evidence

**Project:** EEG–Eye-Tracking Multimodal Research (Project 3) within Eye-Tracking & Neurocognitive Intelligence. **Target:** ZuCo 2.0 source-defined Normal Reading (NR) versus Task-Specific Reading (TSR). **Scope:** 18 participants, 36 original NR/TSR Run-1 recordings, 630 paired windows, ten fixed model candidates and a participant-disjoint **11 Train / 3 Validation / 4 reserved Test** split. The completed six-notebook development workflow selected **Gaze Random Forest** using 99 Validation windows from three people (Macro F1 **0.6244**); the 160-window Test remains **unscored**. The 44 current stage outputs document this experiment. See [README](README.md) and the [single study workflow](images/study_overview.svg).

## 1. Research Objective and Scope

The primary question is whether **paired EEG and eye-movement features** improve discrimination of NR and TSR over the strongest gaze-only or EEG-only comparator when evaluated on identical windows from held-out participants. The outcome is a **source instruction/task label**, not reading comprehension, attentional state, neurological diagnosis or a clinical endpoint.

The study examines five connected questions: can original EEG and gaze be paired on a defensible timebase; can eligible windows yield comparable classical and temporal representations; how do simple and single-modality baselines perform; does early, late or gated fusion provide a Validation advantage; and how consistent are model rankings and fixed-input perturbations across independent people?

The implemented six-stage programme provides **Validation-based model selection**, not an independently estimated final Test score. The study does not validate physiological fixation-related potentials, perform causal modality attribution or establish cross-device, cross-session or external-dataset generalisation. No GazeBase-derived measurements, labels or outcomes are transferred from the other eye-tracking studies in this repository.

## 2. Data Sources, Cohort and Target Definitions

### 2.1 Source and access

The source is [ZuCo 2.0](https://www.nemar.org/dataset/nm000279), NEMAR dataset `nm000279` release **v1.0.0**. The BIDS metadata reference ZuCo 2.0 (Hollenstein, Tröndle, Zhang and Langer, 2020), BIDS 1.9.0 and **CC BY 4.0** data terms. The indexed source has **18 participants** with nominal **500-Hz** EEG and EyeLink left-eye sampling. EEG acquisition metadata identify **128 channels**, though per-run sidecars can contain differing manufacturer or reference descriptors; the analysed EEG feature representation uses **16 source channels**.

The source reader accesses specific BrainVision EEG files (`.eeg`, `.vhdr`, `.vmrk`), EEG metadata/channel/event tables and EyeLink compressed `physio.tsv.gz` and JSON sidecars through the NEMAR dataset gateway and [BIDS file index](https://github.com/nemarDatasets/nm000279). The recorded Stage-01 execution retrieved **36/36 original run packages**, one NR and one TSR Run-1 recording per person, totalling **3,373.4 MB**. The raw EEG and gaze data are not bundled in this repository; an ephemeral Colab cache must not be mistaken for durable storage.

### 2.2 Cohort, labels and independent units

BIDS task identity maps `nr` to class **0** and `tsr` to class **1**. The **participant** is the independent split and uncertainty unit. Each person contributes two source runs; each source run contributes eligible, non-overlapping **12-second windows**; each window becomes one binary prediction row and six consecutive **two-second** temporal bins.

The frozen protocol (seed **17**) assigns **11 Train, three Validation and four reserved Test participants**. It preserves all six historical pilot participant roles, including the originally reserved Test people, before fitting the expanded experiment. Source IDs, labels, run IDs, paths and split roles provide provenance, not predictive features. The 18-person source registry, role table and current window registry agree.

| Partition | Independent people | NR windows | TSR windows | Total windows |
| --- | ---: | ---: | ---: | ---: |
| Train | 11 | 201 | 170 | **371** |
| Validation | 3 | 60 | 39 | **99** |
| Reserved Test | 4 | 80 | 80 | **160** |
| **Total** | **18** | **341** | **289** | **630** |

The split is participant-disjoint, but each participant contributes repeated windows. Neither 630 windows nor 99 Validation windows represent that many independent people. Task instructions and text/stimulus composition may differ between NR and TSR and therefore remain plausible confounders.

## 3. Data Preparation and Representations

### 3.1 BIDS-relative EEG–gaze alignment

Notebook 02 uses BrainVision EEG time and the EyeLink BIDS-relative sampling time, calculated from eye `StartTime` plus sample index divided by sampling frequency. Observed start offsets range **−0.046 to +0.033 seconds**. Runs must have usable relative-time overlap and valid event bounds. Within overlap, the implementation forms **non-overlapping 12-second windows**, capped at **20 windows per run**, with at least **0.60 finite x/y gaze fraction**.

All **36 participant/task runs** contributed at least one eligible window. The final registry contains **630 unique paired IDs**; `sub-YAK` NR contributes only one eligible window from approximately **15.196 seconds** of EEG. The fixed cap and unequal run durations mean that pooled-window metrics weight participants and tasks unequally.

BIDS-relative overlap is a **timebase assumption**, not independently measured EEG–EyeLink synchronisation. Every recorded run-level drift flag remains unverified. Finite gaze coordinates indicate availability, not accurate calibration, true fixations or absence of eye artefacts.

### 3.2 EEG and gaze features

For the selected **16 EEG channels**, each 12-second window is channel-centred and passed through a **1–40 Hz Butterworth filter**. Welch estimates within each two-second bin yield channel-averaged **theta (4–8 Hz), alpha (8–13 Hz) and beta (13–30 Hz) log bandpower**, plus **EEG RMS in µV**. This model representation is distinct from the raw single-channel fixation-locked diagnostic.

Gaze bins contain finite-coordinate fraction, median two-dimensional speed (**pixels/s**), median vendor pupil measurement and the fraction overlapping vendor fixation events. No calibrated pupil-diameter or fixation ground-truth interpretation is assumed.

Each paired window contributes **eight window-mean features** (four EEG plus four gaze), stored in `03_features.csv`, and two modality-specific six-bin tensors, **EEG (630, 6, 4)** and **gaze (630, 6, 4)**, stored together in `03_temporal_sequences.npz`. The arrays preserve source-window IDs and numeric `float32` features; identifier/label fields use fixed-width Unicode rather than object arrays. Consumers check that the temporal-bin means match the corresponding CSV means.

### 3.3 Eligibility, missingness and diagnostic boundaries

The recorded extraction retained **all 630 windows**, with **zero extraction exclusions** and no non-finite values among the eight exported window means. However, `gaze_valid_fraction` has **zero Train standard deviation** and therefore cannot discriminate Train conditions. Its observed constancy is an informative data-quality constraint, not evidence of independent gaze accuracy.

A **Train-only**, fixation-locked *raw single-channel* EEG average exhibits an **uncorrected offset near −10,100 µV**. This diagnostic was not used to define the centred, filtered prediction features. Without validated EEG reference/baseline, event timing, overlap treatment and ocular artefact control, it is **not** evidence of a physiological fixation-related potential (FRP), neural localisation or cognitive processing effect.

## 4. Experimental Design and Evaluation

### 4.1 Train, Validation and Test roles

Train participants fit classical estimators, imputation/scaling, neural parameters and temporal feature scaling. The **three Validation participants** inform GRU checkpoint selection and the cross-family model decision. All ten candidates were compared on the **same 99 Validation window IDs and class labels**, generating **990 model/window prediction records**. The final four Test participants and **160 Test windows** were not used to select, tune, score or revise the development winner.

The current Stage-06 code executed in `run_mode=development` with **`RUN_FINAL_TEST=False`** and saved `test_used_in_selection=False`. There are no current final Test predictions, confusion matrices, scores or participant-level Test intervals. Future Test work requires a separately authorised execution, and the frozen Validation choice must not be retrospectively changed because of an observed Test result.

### 4.2 Primary metrics and fixed selection rule

The primary selection maximises **fixed-label pooled Validation Macro F1**, breaking ties by **Balanced Accuracy**, then by the recorded **predefined candidate priority**. Accuracy and per-person Macro F1 support interpretation but do not replace this selection order. All predictions use the same NR/TSR label order and source-window identities.

Pooled-window scores describe the sampled windows. Independent-person uncertainty must recognise participants as clusters; naive resampling of individual windows would overstate independent support. With only three Validation people, any apparent small ranking difference is primarily descriptive.

### 4.3 Fair comparison and leakage boundary

Classical and temporal candidates have different representations but identical participant roles, target labels and Validation windows. Input identifiers, paths, split role and task label are excluded from predictors. Train-derived preprocessing is applied unchanged to Validation. Saved prediction probabilities, label order and checkpoint metadata were reconciled during Stage 06. A single within-source Validation split does not establish equal optimisation budgets, stimulus-independent task recognition or a general population-level fusion effect.

## 5. Methods, Models and Configurations

| Notebook | Candidate or operation | Implemented settings and role |
| --- | --- | --- |
| 01 | Source selection and participant-disjoint protocol | 18 ZuCo 2.0 people, Run 1 NR/TSR, seed **17**, six original pilot roles retained |
| 02 | EEG–EyeLink pairing | BIDS-relative overlap; non-overlapping **12-s windows**; finite x/y fraction ≥**0.60**; maximum **20** windows/run |
| 03 | EEG/gaze representations | 16-channel per-window-centred **1–40 Hz** EEG; theta/alpha/beta log power and RMS; four gaze measures; six **2-s** bins |
| 04 | Class Prior | Training class-frequency baseline; no physiological features |
| 04 | Gaze Logistic / Gaze Random Forest | Four gaze means; Train-fitted median imputation, scaling where used, regularised logistic model or **150-tree** forest |
| 04 | EEG Logistic / EEG Random Forest | Four EEG means; corresponding fixed classical pipelines |
| 04 | Early-Fusion Logistic Regression | Joint eight-feature EEG+gaze vector, with Train-fitted preprocessing |
| 04 | Equal-weight Late Fusion | Arithmetic mean of fixed EEG and gaze logistic probabilities; **0.5/0.5** weights |
| 05 | Gaze GRU / EEG GRU | Six-bin single-modality sequence; hidden width **16**, dropout **0.2** |
| 05 | Gated Fusion GRU | Paired six-bin EEG/gaze sequences; sigmoid-gated modality blending; no causal interpretation of gates |
| 06 | Selection and fixed masking | Ten Validation candidates, pooled Macro F1 and tie-breaks; feature masking in fitted **early-fusion logistic** model only; Test disabled |

**Temporal training:** The three compact GRUs use AdamW with learning rate **0.001**, weight decay **0.001**, batch size **64**, up to **15 epochs**, early stopping patience **4** and seed **17**. Train-derived temporal scaling is saved in `05_scaling.npz`. Gaze and EEG GRUs each have **1,090 parameters**; gated fusion has **2,354**. Training produced **15/7/14** recorded epochs respectively, with best checkpoints selected at **12/3/10**.

The model comparison is a set of configured pipelines, not a matched computational-budget experiment. All model weights and recorded training histories are preserved as current outputs; no new fit is implied by the documentation revision.

## 6. Results and Comparative Analysis

### 6.1 Completed acquisition, alignment and feature extraction

Notebooks 01–03 recorded **18 people**, **36 NR/TSR Run-1 packages**, **630 eligible paired windows**, **630 complete feature rows** and two **(630, 6, 4)** temporal tensors. The frozen split contains **371 Train**, **99 Validation** and **160 reserved Test** windows. Window IDs, participant roles, task labels and source runs reconcile across saved Stage-01/02/03 evidence. There are **zero feature-extraction exclusions**, but independent drift/calibration validation remains absent.

### 6.2 Classical and simple-fusion Validation (Notebook 04)

Seven candidates produced **693 predictions** on the same **99 Validation windows** of three independent people. Gaze Random Forest led this family. The reported values are from [`04_validation_metrics.csv`](output/04_validation_metrics.csv).

| Classical candidate | Validation Macro F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| **Gaze Random Forest** | **0.6244** | **0.6231** | **0.6465** |
| Equal-weight Late Fusion | 0.5643 | 0.5641 | 0.5859 |
| Gaze Logistic Regression | 0.5472 | 0.5474 | 0.5657 |
| EEG Random Forest | 0.5139 | 0.5282 | 0.5152 |
| EEG Logistic Regression | 0.4545 | 0.4737 | 0.4545 |
| Early-Fusion Logistic Regression | 0.4518 | 0.4603 | 0.4545 |
| Class Prior | 0.3774 | 0.5000 | 0.6061 |

Gaze Random Forest correctly classified **44 of 60 NR** and **20 of 39 TSR** Validation windows. The best classical fusion candidate, equal-weight Late Fusion, scored **0.0600** less Macro F1 than gaze forest. This is not evidence of a population-wide fusion disadvantage.

### 6.3 Temporal GRU Validation (Notebook 05)

Three GRUs produced **297 predictions** on the same 99 Validation windows. The results in [`05_validation_metrics.csv`](output/05_validation_metrics.csv) are:

| Temporal candidate | Validation Macro F1 | Balanced Accuracy | Accuracy | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| **Gated Fusion GRU** | **0.6107** | **0.6096** | **0.6465** | 10 |
| Gaze GRU | 0.5938 | 0.5929 | 0.6263 | 12 |
| EEG GRU | 0.5938 | 0.5929 | 0.6263 | 3 |

The gated network led the temporal family by **0.01695 Macro F1** over either unimodal GRU. It did **not** exceed Gaze Random Forest.

### 6.4 Ten-model development selection (Notebook 06)

Stage 06 reconciled **990 matched Validation predictions** across all ten candidates, plus **30 candidate/person metric rows**. The fixed rule selected [**Gaze Random Forest**](output/06_model_selection.csv), with Macro F1 **0.624390243902439**, Balanced Accuracy **0.6231** and Accuracy **0.6465**. Its closest competitor was Gated Fusion GRU (**0.6107**), a Macro F1 difference of approximately **0.0137**. The selected model's performance is a **Validation estimate**, not final Test generalisation.

The selected gaze forest exhibited substantial person-level variation: **0.7654** for `sub-YDR` (31 windows), **0.6306** for `sub-YRH` (38), and **0.3478** for `sub-YTL` (30). Pooling 99 correlated windows does not make this a large independent sample.

**Observed result:** Adding EEG did **not** improve the best pooled Validation Macro F1 over the strongest gaze-only candidate in this particular development experiment. This does not establish that EEG is generally uninformative, that fusion is harmful or that the same ordering will hold on Test participants.

## 7. Robustness, Uncertainty and Additional Analyses

### 7.1 Fixed early-fusion input masking

Stage 06 perturbed the already fitted **early-fusion logistic regression** by masking one feature group and using its existing **Train-derived median imputer**. Neither the model nor preprocessing was refitted. On the same **99 Validation windows from three people**, [`06_validation_sensitivity.csv`](output/06_validation_sensitivity.csv) records:

| Fixed early-fusion input | Validation Macro F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| Original EEG + gaze | 0.4518 | 0.4603 | 0.4545 |
| EEG features masked | **0.5343** | **0.5571** | **0.6263** |
| Gaze features masked | 0.4907 | 0.5385 | 0.4949 |

Both artificial masks improved Macro F1 for this relatively weak *particular* fixed classifier. These results concern shifted inputs after imputation, **not** independently retrained modality ablations, causal sensor contributions or the selected Gaze Random Forest. Gated GRU weights have no physiological or causal interpretation either.

### 7.2 Independent-person uncertainty and unresolved signal issues

Only **three people** support Validation model choice, and their gaze forest scores differ markedly. The four Test people remain reserved and unscored. A participant-clustered interval or subgroup analysis on such a small independent sample would be exploratory; no independent population confidence claim is established by window counts.

BIDS timing does not verify independent EEG–EyeLink drift; finite eye coordinates do not validate calibration. The Train-constant finite-coordinate fraction cannot discriminate source task. EEG may carry ocular contamination. Unequal run lengths, repeated windows and NR/TSR stimulus/instruction differences restrict interpretation. The raw fixation-locked single-channel diagnostic has an uncorrected **approximately −10,100 µV** offset and cannot support FRP inference without reference, baseline, event overlap and artefact investigation.

No independent Test, cross-session, external-dataset or prospective robustness evaluation has been performed for the expanded experiment.

## 8. Current Outputs and Reproducibility

All **44** stage-owned results belong to the same current **18-person, 630-window** run, in [`output/`](output/). The former six-person pilot survives in Git history but does not define an alternative set of current files.

| Stage | Current owned evidence | Key producer-to-consumer relationship |
| --- | --- | --- |
| 01 · Source (7 files) | `01_source_manifest.csv`, `01_split_assignments.csv`, `01_protocol.json`, eligibility, download and acquisition evidence, overview plot | Original person/run IDs and frozen roles define downstream groups |
| 02 · Pairing (5 files) | `02_window_registry.csv`, `02_window_summary.csv`, alignment and person/task QA, plot | Unique 630 EEG/gaze window IDs and time bounds |
| 03 · Features (7 files) | `03_features.csv`, `03_temporal_sequences.npz`, feature/modality/exclusion QA and two plots | Paired mean rows and two six-bin tensors remain ID-aligned |
| 04 · Classical (12 files) | Six `.joblib` pipelines, `04_late_fusion.json`, `04_validation_metrics.csv`, 693 predictions, person metrics and plots | Train-only fitting; seven same-window Validation results |
| 05 · Temporal (9 files) | Three `.pt` GRU checkpoints, `05_scaling.npz`, histories, 297 predictions, metrics and plot | Train-derived temporal scaling and checkpoint selection |
| 06 · Development decision (4 files) | `06_model_selection.csv`, `06_validation_person_metrics.csv`, `06_validation_sensitivity.csv`, `06_validation_evidence.png` | Ten-model matched Validation choice, no Test results |

All six notebooks have executed on the 18-person original-data workflow, and their saved outputs are committed. This establishes recorded results and internal source/identifier reconciliation, **not** independent reproduction in the current environment. Earlier reported synthetic tests and GitHub Actions successes are separate from new original-data execution; no model was trained as part of this documentation update.

The study requires the [original ZuCo release](https://www.nemar.org/dataset/nm000279), [study dependencies](requirements.txt) and the established Colab `setup_colab("eeg_eye_tracking_fusion")` workflow. Source recordings may have to be downloaded again if a Colab cache disappears. A reproducible original-data run proceeds through [Notebooks 01–06](code/notebooks/) in numerical order; each stage overwrites its own current named outputs. Changing cohort, roles, alignment, feature definitions or representation shapes can invalidate downstream models and must trigger a dependency review.

The [README](README.md) contains the single six-stage overview. Each notebook retains a separate editable five-band figure in [`images/`](images/). The Test branch was deliberately disabled; no ordinary documentation or reporting operation should silently launch it.

## 9. Discussion, Limitations and Future Work

The six-stage development workflow demonstrates an end-to-end **source-audited EEG–gaze pairing and matched Validation comparison**. In the expanded ZuCo 2.0 experiment, Gaze Random Forest was chosen using three independent Validation people and achieved pooled Macro F1 **0.6244**; the strongest fusion GRU achieved **0.6107**. This is a negative or mixed result for the proposed *added value of EEG over the strongest gaze comparator* in the specific Validation setting, rather than a claim that multimodal learning fails generally.

The central limitations are the small number of independent Validation participants, repeated and unevenly distributed windows, source task/text confounding, incomplete EEG/gaze calibration and clock-drift evidence, potential ocular EEG artefacts, the constant gaze-validity predictor and the unvalidated raw fixation-locked EEG diagnostic. No causal, cognitive, clinical, external or population-wide claim follows from the available results.

**Current boundary:** the 18-person Validation-development study has completed Notebooks **01–06**, all ten candidates have matched Validation results and the development winner is fixed. The final four-person Test remains unscored and is **not** a routine documentation-clean-up task. Any future Test evaluation, stimulus-matched control, drift study, physiologically grounded FRP analysis or external validation should be specified and authorised as scientific work, preserving the present source roles and model-selection history.

**Source attribution:** Hollenstein, N., Tröndle, M., Zhang, C., & Langer, N. (2020). *ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation*. [LREC 2020](https://aclanthology.org/2020.lrec-1.18/); [NEMAR dataset DOI](https://doi.org/10.82901/nemar.nm000279). BIDS source metadata report [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) terms.
