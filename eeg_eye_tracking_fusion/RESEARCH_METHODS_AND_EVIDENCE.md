# EEG–Eye-Tracking Multimodal Research | Research Methods and Evidence

**Project:** EEG–Eye-Tracking Multimodal Research (Project 3) within Eye-Tracking & Neurocognitive Intelligence. **Target:** ZuCo 2.0 source-defined Normal Reading (NR) versus Task-Specific Reading (TSR). **Scope:** 18 participants, 36 original NR/TSR Run-1 recordings, 630 paired windows, ten fixed model candidates and a participant-disjoint **11 Train / 3 Validation / 4 reserved Test** split. The six notebooks selected **Gaze Random Forest** on three-person Validation (Macro F1 **0.6244**) and scored the frozen candidate on **160 windows from four independent Test people** (Macro F1 **0.4997**, Accuracy **0.5250**). The **49 current stage outputs** document this experiment. See [README](README.md) and the [single study workflow](images/study_overview.svg).

## 1. Research Objective and Scope

The primary question is whether **paired EEG and eye-movement features** improve discrimination of NR and TSR over the strongest gaze-only or EEG-only comparator when evaluated on identical windows from held-out participants. The outcome is a **source instruction/task label**, not reading comprehension, attentional state, neurological diagnosis or a clinical endpoint.

The study examines five connected questions: can original EEG and gaze be paired on a defensible timebase; can eligible windows yield comparable classical and temporal representations; how do simple and single-modality baselines perform; does early, late or gated fusion provide a Validation advantage; and how consistent are model rankings and fixed-input perturbations across independent people?

The implemented six-stage programme includes **Validation-based selection followed by held-out Test evaluation without retraining or re-selection**. Its Test result is within the same ZuCo 2.0 source, not external validation. The study does not validate physiological fixation-related potentials, perform causal modality attribution or establish cross-device or cross-session generalisation. No GazeBase-derived measurements, labels or outcomes are transferred from the other eye-tracking studies in this repository.

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

Train participants fit classical estimators, imputation/scaling, neural parameters and temporal feature scaling. The **three Validation participants** inform GRU checkpoint selection and the cross-family model decision. All ten candidates were compared on the **same 99 Validation window IDs and class labels**, generating **990 model/window prediction records**. The final four Test participants and **160 Test windows** were not used in fitting, checkpoint choice or Validation-based selection; they were scored only after the development winner was frozen.

Stage 06 first recorded the development choice **`selected_model=gaze_forest`** with **`test_used_in_selection=False`**. The authorised final execution set **`RUN_FINAL_TEST=True`** and recorded **`run_mode=final`**, without altering the selected model. It evaluated the saved classical pipelines and GRU checkpoints, producing 1,600 Test predictions for ten candidates on 160 matching windows. Test scores were not used to choose a new model or adjust preprocessing.

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
| 06 | Frozen selection and final Test | Ten Validation candidates and fixed ranking; secondary early-fusion masking on Validation; final Test scoring on four held-out people without model refitting |

**Temporal training:** The three compact GRUs use AdamW with learning rate **0.001**, weight decay **0.001**, batch size **64**, up to **15 epochs**, early stopping patience **4** and seed **17**. Train-derived temporal scaling is saved in `05_scaling.npz`. Gaze and EEG GRUs each have **1,090 parameters**; gated fusion has **2,354**. Training produced **15/7/14** recorded epochs respectively, with best checkpoints selected at **12/3/10**.

The candidates differ in input representation, model capacity and training budget. The study does not isolate these factors experimentally; saved checkpoints and histories document the configurations that were executed.

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

**Validation finding:** Adding EEG did **not** improve the best pooled Validation Macro F1 over the strongest gaze-only candidate in this development experiment. This does not establish that EEG is generally uninformative or that the same ordering holds on other people.

### 6.5 Final held-out Test of the frozen model (Notebook 06)

The **Gaze Random Forest** remained the primary model selected by the fixed Validation rule. Its **160 Test windows** came from four independent people, with 80 NR and 80 TSR windows. The same trained pipeline and feature order were applied without refitting. On these Test observations, the primary achieved **Accuracy 0.5250**, **Macro F1 0.4997** and **Balanced Accuracy 0.5250**. Its [Test confusion matrix](output/06_primary_test_confusion.png) contains **60 correct NR, 20 NR labelled TSR, 56 TSR labelled NR and 24 correct TSR**. Thus TSR recall was **24/80 = 0.30**.

| Candidate (Test, secondary except primary) | Macro F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| **Gaze Random Forest (frozen primary)** | **0.4997** | **0.5250** | **0.5250** |
| Equal-weight Late Fusion | 0.5204 | 0.5750 | 0.5750 |
| Gated Fusion GRU | 0.5157 | 0.5750 | 0.5750 |
| Early-Fusion Logistic Regression | 0.5014 | 0.5625 | 0.5625 |
| Gaze GRU | 0.4910 | 0.5313 | 0.5313 |
| Gaze Logistic Regression | 0.4764 | 0.5125 | 0.5125 |
| EEG Random Forest | 0.4764 | 0.5500 | 0.5500 |
| EEG Logistic Regression | 0.3861 | 0.4813 | 0.4813 |
| Class Prior | 0.3333 | 0.5000 | 0.5000 |
| EEG GRU | 0.3333 | 0.5000 | 0.5000 |

The [complete Test metrics](output/06_test_metrics.csv) and [1,600 matched predictions](output/06_test_predictions.csv) preserve the original candidate identities. Although Late Fusion and Gated Fusion GRU score slightly above the preselected primary on Test, these are **secondary comparisons**, not grounds for changing the selected model. The lower primary Test score relative to Validation (0.4997 versus 0.6244) limits claims of strong between-person generalisation.

## 7. Robustness, Uncertainty and Additional Analyses

### 7.1 Fixed early-fusion input masking

Stage 06 perturbed the already fitted **early-fusion logistic regression** by masking one feature group and using its existing **Train-derived median imputer**. Neither the model nor preprocessing was refitted. On the same **99 Validation windows from three people**, [`06_validation_sensitivity.csv`](output/06_validation_sensitivity.csv) records:

| Fixed early-fusion input | Validation Macro F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| Original EEG + gaze | 0.4518 | 0.4603 | 0.4545 |
| EEG features masked | **0.5343** | **0.5571** | **0.6263** |
| Gaze features masked | 0.4907 | 0.5385 | 0.4949 |

Both artificial masks improved Macro F1 for this relatively weak *particular* fixed classifier. These results concern shifted inputs after imputation, **not** independently retrained modality ablations, causal sensor contributions or the selected Gaze Random Forest. Gated GRU weights have no physiological or causal interpretation either.

### 7.2 Sources and consequences of scientific limitations

The study used **18 participants**, split by participant into **11 Train, three Validation and four Test people**. Each Test participant contributed **40 windows** (20 NR and 20 TSR). The selected gaze forest's Test participant Macro F1 values were **0.5238** (`sub-YAC`), **0.3333** (`sub-YAG`), **0.4505** (`sub-YMD`) and **0.3333** (`sub-YRP`). A **500-resample participant-cluster bootstrap** yielded a descriptive Macro F1 **95% percentile interval of 0.3333–0.5616**. The number of *independent people*, rather than the 160 correlated Test windows, limits the precision of this interval and broader generalisation claims.

| Source of limitation in the recorded study | Consequence for interpretation |
| --- | --- |
| **Cohort and split:** Only three people informed model selection and four independent people contributed to Test; run lengths also varied. | Repeated windows are not independent participants. Validation rankings and Test uncertainty reflect a small, unevenly sampled cohort, not a population-wide estimate. |
| **EEG–gaze time alignment:** Paired windows were constructed using BIDS `StartTime` and nominal sample rates. Independent EEG–EyeLink clock drift was **not measured**. | The recorded temporal overlap is supported by source metadata, but drift-free synchronisation cannot be claimed. |
| **Gaze measurement:** The eligibility check requires finite x/y coordinates, not an independent eye-tracker calibration assessment. The gaze-validity feature was constant on Train. | Numerical availability does not establish gaze-position accuracy or freedom from ocular artefacts; the constant feature cannot discriminate the Train task labels. |
| **NR/TSR task definition:** Labels describe original reading conditions; instructions and presented texts may differ. No stimulus-matched control was evaluated. | Successful task classification cannot be attributed uniquely to comprehension or a specific cognitive process. |
| **EEG diagnostic:** Ocular contamination was not independently quantified. The exploratory fixation-locked raw EEG trace has an uncorrected **approximately −10,100 µV** offset. | The trace does not establish a validated fixation-related potential (FRP) without further reference, baseline, event-overlap and artefact checks. |
| **Validation setting:** Train, Validation and Test all originate from **ZuCo 2.0**. | The held-out participant Test assesses within-source transfer, not external-dataset, cross-session, prospective or clinical generalisation. |

These are **limitations of the available cohort, source measurements and experimental scope**, not evidence that Notebook 06 or the held-out Test was left unexecuted. Addressing them would require additional source-level verification, controlled comparisons or new independent data; those investigations are distinct from the completed evaluation reported here.

## 8. Current Outputs and Reproducibility

The 49 derived files in [`output/`](output/) represent the 18-person, 630-window experiment.

| Stage | Current owned evidence | Key producer-to-consumer relationship |
| --- | --- | --- |
| 01 · Source (7 files) | `01_source_manifest.csv`, `01_split_assignments.csv`, `01_protocol.json`, eligibility, download and acquisition evidence, overview plot | Original person/run IDs and frozen roles define downstream groups |
| 02 · Pairing (5 files) | `02_window_registry.csv`, `02_window_summary.csv`, alignment and person/task QA, plot | Unique 630 EEG/gaze window IDs and time bounds |
| 03 · Features (7 files) | `03_features.csv`, `03_temporal_sequences.npz`, feature/modality/exclusion QA and two plots | Paired mean rows and two six-bin tensors remain ID-aligned |
| 04 · Classical (12 files) | Six `.joblib` pipelines, `04_late_fusion.json`, `04_validation_metrics.csv`, 693 predictions, person metrics and plots | Train-only fitting; seven same-window Validation results |
| 05 · Temporal (9 files) | Three `.pt` GRU checkpoints, `05_scaling.npz`, histories, 297 predictions, metrics and plot | Train-derived temporal scaling and checkpoint selection |
| 06 · Selection and final Test (9 files) | Four development outputs (`06_model_selection.csv`, Validation person metrics, sensitivity and evidence plot) plus `06_test_metrics.csv`, `06_test_predictions.csv`, `06_person_metrics.csv`, `06_cluster_bootstrap.csv` and `06_primary_test_confusion.png` | Frozen Validation selection, 1,600 Test predictions, per-person metrics and selected-primary uncertainty |

Notebooks 01–05 cover original-data acquisition, feature extraction and model fitting. Notebook 06 evaluates the saved features, classical pipelines and GRU checkpoints on the held-out participants. Model, participant and window identities, class probabilities and Test metrics were checked against the saved predictions. These results describe held-out inference; they do not establish independent reproduction of acquisition or model fitting.

Full reproduction requires the [original ZuCo release](https://www.nemar.org/dataset/nm000279) and [study dependencies](requirements.txt). For final inference, the Stage-03 features and sequences, Stage-04 pipelines, Stage-05 checkpoints and scaling, and Notebook 06 are sufficient. The classical pipelines require **scikit-learn 1.6.1**; the recorded headless evaluation used Python 3.11. Run Notebooks 01–06 in order for source-to-result reproduction. Changes to cohort, roles, alignment or features require regeneration of affected results and a justified evaluation.

The [README](README.md) contains the study overview, and the individual notebooks link to their scientific diagrams in [`images/`](images/). The [headless evaluation adapter](code/src/run_final_test.py) provides an alternative to running Notebook 06 interactively, using the same scientific cells.

## 9. Discussion, Limitations and Future Work

The six-stage study now records both **source-audited paired EEG–gaze modelling and final participant-disjoint Test evaluation**. Gaze Random Forest was selected by three-person Validation (Macro F1 **0.6244**) and then achieved Macro F1 **0.4997** on four Test people. Although equal-weight Late Fusion and Gated Fusion GRU scored **0.5204** and **0.5157** on Test, these are post-selection comparisons, not new winners. The primary Test result is only modestly above the equal-class accuracy baseline of 0.5; this is a limited within-source task result, not proof of multimodal or general cognitive decoding.

The limitations have identifiable sources. **Cohort and design constraints** include three Validation and four Test participants, correlated windows and possible differences in NR/TSR reading instructions or stimuli. **Measurement and verification constraints** include BIDS-relative rather than independently drift-verified EEG–EyeLink alignment, unverified gaze calibration, potential ocular EEG contamination and an exploratory fixation-locked trace with an uncorrected offset. **Evaluation scope** is restricted to participant-disjoint testing within ZuCo 2.0; no independent external or cross-session dataset was evaluated. These constraints qualify the scientific claims but do not negate the completed Test execution. Section 7.2 states their individual causes and consequences; the four-person clustered interval remains exploratory.

**Current boundary:** Notebooks **01–06** are completed, the development candidate was frozen before held-out scoring, and the four-person Test is now evaluated and documented. Further investigations of stimuli, timing, FRP, cross-session or external data are separate research extensions and must not retroactively alter the recorded Test-selected procedure.

**Source attribution:** Hollenstein, N., Tröndle, M., Zhang, C., & Langer, N. (2020). *ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation*. [LREC 2020](https://aclanthology.org/2020.lrec-1.18/); [NEMAR dataset DOI](https://doi.org/10.82901/nemar.nm000279). BIDS source metadata report [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) terms.
