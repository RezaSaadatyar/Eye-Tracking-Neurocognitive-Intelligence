# EEG–Eye-Tracking Multimodal Research

A six-notebook research workflow testing whether EEG and gaze provide complementary evidence for distinguishing ZuCo 2.0 **Normal Reading (NR)** from **Task-Specific Reading (TSR)**. **Expanded 18-person Stages 01–04 are executed and published:** 36 original source runs, 630 eligible matched 12-second windows, 630 EEG/gaze feature rows and seven fixed classical/simple-fusion Validation candidates. The frozen split is **371 Train / 99 Validation / 160 reserved Test windows**, from **11/3/4 independent participants**. In Stage 04, **Gaze Random Forest** leads classical candidates with **Validation Macro F1 0.6244**, evaluated on three held-out people. **Temporal Stage 05 and ten-model selection Stage 06 remain pending.** Earlier six-person pilot scores are historical, not the expanded experiment. Test remains unscored.

**Recorded evidence:** Stage 01 retrieved all **36 NR/TSR Run-1 source packages** from 18 people (3,373.4 MB), retaining all six previous pilot roles. Stage 02 produced **630 BIDS-relative eligible paired windows**, and Stage 03 retained **630 EEG/gaze feature rows** and separate **(630, 6, 4)** temporal arrays, with no extraction exclusions. `gaze_valid_fraction` is constant (1.000) in the current Train data. Stage 04 generated **693 seven-model Validation predictions**, **21 descriptive person/model metric rows**, six fitted estimators and its fixed 0.50/0.50 late-fusion result. Independent EEG–EyeLink clock drift and eye calibration remain unverified.

![Overview of EEG and eye-tracking data processing and evaluation](images/study_overview.svg)

*Figure 1. Expanded Stages 01–04 are executed and published; Gaze Random Forest leads the current seven-candidate classical Validation comparison. The three temporal GRUs and final Validation-only model selection have not yet run for 18 people.*

## 1. Dataset and Experimental Protocol

The source is **ZuCo 2.0**, NEMAR dataset [`nm000279`, release `v1.0.0`](https://www.nemar.org/dataset/nm000279). The current source cohort contains **18 paired-file-eligible participants** with **36 original NR/TSR Run-1 recordings**, split at participant level into **11 Train, 3 Validation and 4 reserved Test people**, retaining all six previously assessed pilot assignments. EEG/eye sidecars report nominal **500-Hz sampling**, **128 EEG channels**, and relative eye `StartTime` offsets between **−0.046 and +0.033 s**. Stage 02 used this source registry to produce **630 eligible paired windows**. Independent clock drift and calibration validity remain unverified.

Source acquisition reads only selected BIDS participants and NR/TSR Run-1 records through the public NEMAR source gateway. To select a different currently supported cohort size, change **`MAX_PARTICIPANTS`** in Notebook 01; the participant-role helper retains the original six-person assignments. `RUNS_PER_TASK=1` is an approved scientific setting, **not** a routine population-size control. The full original archive is not required in advance, and raw EEG/eye physiology is not committed to GitHub.

The initial design uses **one run per task per eligible person**, 12-second non-overlapping paired windows, six two-second feature bins per window, and a seeded **participant-disjoint** Train/Validation/Test split. Run and task labels are verified from BIDS metadata and are never inferred from EEG amplitudes or gaze patterns. Source eligibility rules and scientific boundaries are described in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

**Important limitations:** identical nominal EEG and gaze sample rates do not prove independent-clock synchronisation. The current mapping uses BIDS `StartTime` and sample frequencies, with drift unverified unless independent markers become available. NR versus TSR differs in instructions and possibly text/stimulus content; performance cannot automatically be attributed to cognitive processing or multimodal complementarity.

## 2. Experimental Programme

**Execution status:** Expanded **Notebooks 01–04 have executed and are published** for 18 people and 630 aligned feature windows. The six Stage-04 fitted estimators, **seven summary metric rows**, **693 model/window predictions** and **21 per-model/person rows** are recorded and reconciled. Gaze Random Forest leads the classical subset (**Macro F1 0.6244**), but the best model across all ten candidates is **not yet established**. The earlier six-person pilot completed the six-notebook development procedure with no Test scoring (historical reference commit `8f2e206`). Expanded Notebook 05 is prepared, not yet executed; Notebook 06 is pending.

**Current files and provenance:** The active output directory contains **31 compatible expanded results**: seven `01_*` files, five `02_*`, seven `03_*` and **12 `04_*`** files. Historical six-person model results are accessible through Git history but are not mixed into these 18-person outputs. The previous pilot's gaze-logistic Macro F1 **0.8862** came from 31 Validation windows of one person and must **not** be used as the expanded result. The four current Test people and their 160 windows remain reserved.

| Stage | Notebook | Scientific role | Intended named outputs |
| --- | --- | --- | --- |
| 01 | [ZuCo Source Audit and Protocol](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) | **Expanded source stage complete:** 18 participants, 36/36 source packages, 11/3/4 pilot-preserving participant roles, no Test scoring | Seven reconciled current `01_*` files: manifest, assignments, eligibility, download/sidecar evidence, protocol and source overview |
| 02 | [EEG–Gaze Alignment and Quality](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) | **Executed, published:** 36 source runs, 630/630 finite-coordinate-eligible windows; 371/99/160 window split, drift unverified | Five committed `02_*` artefacts, including per-person/task QA |
| 03 | [Preprocessing, Features and Neurophysiology](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) | **Executed, published:** 630/630 matched feature rows, eight finite mean features, two 630×6×4 arrays, zero exclusions, one Train-constant gaze feature | Seven committed `03_*` artefacts: feature CSV, NPZ, exclusion/quality tables and two figures |
| 04 | [Unimodal Baselines and Classical Fusion](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) | **Executed, published:** seven fixed models on 371 Train windows, compared on 99 Validation windows from three people; **Gaze Random Forest Macro F1 0.6244**, no Test scoring | Twelve `04_*` files: six fitted models, 693 predictions, seven pooled rows, 21 per-person rows, late-fusion weights and two plots |
| 05 | [Temporal and Advanced Fusion](code/notebooks/05_temporal_and_advanced_fusion.ipynb) | **Ready for expanded run:** three GRU candidates with Train-derived scaling; 371 Train / 99 Validation; fixed checkpoint rule; Test reserved | Three model checkpoints, Train scaler, Validation history/predictions/metrics, new per-person metrics and learning-curve figure |
| 06 | [Validation Selection and Robustness](code/notebooks/06_robustness_and_final_evaluation.ipynb) | **Pending expanded run:** compare ten candidates using Validation-only selection and descriptive robustness; no Test inference | Validation-only model selection and sensitivity evidence; Test branch remains disabled |

Each stage has an editable five-band scientific SVG in [`images/`](images/). The current [source](images/01_zuco_source_audit_and_protocol.svg), [alignment](images/02_eeg_gaze_alignment_and_quality.svg), [feature](images/03_preprocessing_features_and_neurophysiology.svg) and [Stage-04 classical](images/04_unimodal_baselines_and_classical_fusion.svg) figures represent recorded **18-person evidence**. The [study overview](images/study_overview.svg) distinguishes completed Stages 01–04 from Stage 05/06 pending work. The [Stage-05 diagram](images/05_temporal_and_advanced_fusion.svg) is **prospective** and makes no unexecuted GRU performance claim.

## 3. Main Results

*Stages **01–04** below describe the currently **executed 18-person study**. The previous six-person pilot's temporal and all-ten-model selection scores are **historical**, not a proxy for new-cohort results. The expanded 160 Test windows remain unscored.*

**Executed expanded source audit (Notebook 01):** All **18 indexed participants** had complete original BIDS filename sets for one NR and one TSR Run-1 recording. The **36/36 packages** totalled **3,373.4 MB (3.14 GiB)**. Source sidecars report **128 EEG channels**, nominal **500 Hz** for EEG and EyeLink, and eye-start offsets from **−0.046 to +0.033 seconds**. The frozen partition has **11 Train, 3 Validation and 4 reserved Test people**; six previously assessed pilot roles remain unchanged. The CSV/JSON provenance and identities agree. Source presence does not establish final usable EEG–gaze windows, independent clock-drift verification or model performance.

The relevant comparison is whether a fusion model improves over the strongest **EEG-only or gaze-only** comparator on *identical*, participant-disjoint examples. A negative or mixed fusion result is scientifically informative. Primary model selection is based on Validation only and must **not** be revised using future Test scores.

**Recorded expanded alignment and window audit (Notebook 02 execution, files pending):** The saved Colab notebook processed **36 original EEG–EyeLink run pairs from 18 people**, generated **630 distinct 12-second windows**, and recorded **630/630 passing** the `eye_valid_fraction ≥ 0.60` finite x/y rule. The mean and minimum finite-coordinate fractions were **1.000 across all 36 person/task groups**. `sub-YAK / NR` contributed **one window** because its EEG source lasts approximately **15.196 seconds**; many other runs reached the predefined **20-window cap**. The grouping and original tasks are:

| Partition | NR windows | TSR windows | Total windows | Independent people |
| --- | ---: | ---: | ---: | ---: |
| Train | 201 | 170 | **371** | 11 |
| Validation | 60 | 39 | **99** | 3 |
| Test (reserved) | 80 | 80 | **160** | 4 |
| **Overall** | **341** | **289** | **630** | **18** |

These figures are now supported by **five committed 18-person Stage-02 files**, including the 630-window registry, per-run alignment evidence, per-person/task QA and the saved figure. The finite x/y criterion is not independent eye-calibration or drift verification. All 36 original person/task groups produced at least one eligible window.

![Updated scientific illustration of expanded paired-window availability](images/02_eeg_gaze_alignment_and_quality.svg)

*Figure 2. Executed Stage-02 evidence: 18 participants, 630 eligible windows and BIDS-relative alignment limits. No independent inter-device drift validation is claimed.*

**Executed Stage-03 feature extraction (Notebook 03):** All **630** paired Stage-02 windows were processed, preserving **371 Train, 99 Validation and 160 reserved Test** rows from **11/3/4 people**. The saved `03_features.csv` contains **four EEG and four gaze window-mean fields**, all finite; the original source metadata and all 630 unique window IDs match Stage 02. Two temporal arrays have shape **(630, 6, 4)** per modality, as printed by the executed notebook. All **36 participant/task groups** remain represented, and no extraction failure was reported. The eight-row Train-derived quality table identifies **`gaze_valid_fraction` as constant (1.000)**, whereas the other seven mean predictors vary on Train. These are feature-availability findings, not model results or calibration validation.

The exploratory fixation-locked raw EEG plot has a large uncorrected baseline offset (approximately **−10,100 µV**). It is **not a validated fixation-related potential or filtered EEG feature plot** and must not support physiological conclusions.

**Executed classical and simple-fusion Validation comparison (Notebook 04).** All seven predefined candidates were fitted using **371 Train windows from 11 people** (except fixed equal-weight late fusion, which mixes two trained models) and evaluated on the **same 99 Validation windows from three people** (**60 NR / 39 TSR**). The saved **693 model/window predictions** agree with the seven metric rows and the **21 descriptive person/model records**. The best classical candidate is **Gaze Random Forest**, not the previously favoured pilot gaze logistic model:

| Candidate | Validation Macro F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| **Gaze Random Forest** | **0.6244** | **0.6231** | **0.6465** |
| Equal-weight late fusion | 0.5643 | 0.5641 | 0.5859 |
| Gaze logistic regression | 0.5472 | 0.5474 | 0.5657 |
| EEG random forest | 0.5139 | 0.5282 | 0.5152 |
| EEG logistic regression | 0.4545 | 0.4737 | 0.4545 |
| Early-fusion logistic regression | 0.4518 | 0.4603 | 0.4545 |
| Class prior | 0.3774 | 0.5000 | 0.6061 |

The gaze-forest confusion matrix contains **44/60 NR** and **20/39 TSR** correctly classified windows. Its per-person Validation Macro F1 is **0.7654** (`sub-YDR`, 31 windows), **0.6306** (`sub-YRH`, 38) and **0.3478** (`sub-YTL`, 30). This variation cautions against interpreting the pooled-window score as population-level generalisation. Neither early nor fixed late EEG–gaze fusion exceeded the strongest unimodal candidate **on this Validation set**; no general fusion-harm or causal conclusion follows.

![Expanded Stage-04 classical-model scientific figure](images/04_unimodal_baselines_and_classical_fusion.svg)

*Figure 3. Executed expanded seven-candidate Validation comparison, with three held-out participants and no Test scoring. Scores come from saved source-matched Validation predictions.*

**Temporal and final development stages:** The **three compact GRUs** and their nine planned person/model Validation rows are **not yet observed** on the expanded cohort. The earlier six-person EEG-GRU Macro F1 **0.6890** and pilot all-ten-candidate selection are available as [historical development evidence](https://github.com/RezaSaadatyar/Eye-Tracking-Neurocognitive-Intelligence/tree/8f2e2064272955782400c5fb5ce15dca76640d13/eeg_eye_tracking_fusion/output), but cannot determine current model choice.

**Final-evaluation status:** The current study has completed **classical development Validation**, not temporal comparison, all-ten-model selection or final Test. The four Test participants' **160 windows are reserved**; `RUN_FINAL_TEST=False` is required throughout current development.

## 4. Running and Reviewing the Study

**Next execution: Notebook 05.** The **31 published Stage-01–04 output files** now share the same **18-person source cohort and 630-window registry**. Open the latest [Notebook 05](code/notebooks/05_temporal_and_advanced_fusion.ipynb) in a fresh Colab Runtime. Its Setup is the **approved short Git checkout and shared `setup_colab("eeg_eye_tracking_fusion")`**, without a custom synchronisation manager. It checks Stage-02 source IDs against Stage-03 arrays and fits EEG/gaze scaling on **Train only**.

Notebook 05 retains the approved architectures and fixed hyperparameters: **Gaze GRU, EEG GRU and Gated Fusion GRU**, six two-second bins, hidden width 16, AdamW and up to 15 epochs with predefined early stopping. It should evaluate **99 Validation windows from three people**, yielding **297 model/window predictions** and a new **nine-row `05_validation_person_metrics.csv`**. These are **expected file sizes**, not results that have already executed. Test remains excluded from fitting, checkpoint choice and scoring.

After execution, review, commit and reconcile the stage-owned `05_*` files, then proceed to Notebook 06 for Validation-only comparison of all ten models. Keep **`RUN_FINAL_TEST=False`** unless a separate final-evaluation decision authorises otherwise. Historic six-person model scores are not current 18-person results.

The raw data path is not persistent between Colab sessions. Stage 02/03 source retrieval is intentionally limited to the selected runs rather than the full 18-GB release. GPU use is optional for the compact temporal models; free Colab resource availability is not guaranteed. The original source's attribution and use conditions remain applicable.

## 5. Files and Saved Outputs

**Current expanded output set:** Stage 01 has seven files, Stage 02 five, Stage 03 seven, and Stage 04 **12 files**, including the new `04_validation_person_metrics.csv`. The pending Stage 05 will create its own `05_*` outputs and the new `05_validation_person_metrics.csv`; expanded Stage 06 still has no current results.

```text
eeg_eye_tracking_fusion/
├── README.md
├── RESEARCH_METHODS_AND_EVIDENCE.md
├── requirements.txt
├── code/
│   ├── notebooks/              # Six ordered Colab notebooks
│   └── src/                    # Source, alignment, features and model helpers
├── images/                     # Overview and six role-based SVG figures
├── output/                     # Stage-01–06 development evidence files
└── data/                       # Source cache is external to Git by design
```

Notebook outputs in [`output/`](output/) are owned by their stages. The active directory currently holds **31 compatible 18-person files**: seven `01_*`, five `02_*`, seven `03_*` and twelve `04_*`. No expanded `05_*` or `06_*` outputs have yet been published; historical six-person comparisons remain accessible in Git history. Stage 04 metrics come from three independent Validation people and must not be conflated with final Test evidence.

## 6. Interpretation and Scope

This project studies original reading-task labels from paired EEG and eye movements. A classifier may exploit task instructions, source texts, session conditions, eye movements and other nuisance differences; high task discrimination **does not directly demonstrate language comprehension, neurocognitive state, clinical validity or causal EEG contributions**.

Temporal EEG features derive from a fixed 16-channel subset without anatomical localisation claims. The published exploratory raw fixation-locked trace has a large uncorrected offset and does not substitute for a referenced/filtered, artefact-controlled ERP/FRP analysis. Participant-disjoint validation controls person overlap, **not cross-session or stimulus confounding**. Independent timing drift and external generalisation remain unverified unless explicitly assessed.

**Data and attribution:** [ZuCo 2.0 (Hollenstein et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.18/), [NEMAR release nm000279](https://doi.org/10.82901/nemar.nm000279), licence reported in the BIDS metadata as CC BY 4.0. Code and methodological outputs in this repository are separate from the original source recordings.
