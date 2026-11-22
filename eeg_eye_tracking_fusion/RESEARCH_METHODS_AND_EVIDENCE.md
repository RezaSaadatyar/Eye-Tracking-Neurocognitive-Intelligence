# EEG–Eye-Tracking Multimodal Research: Research Methods and Evidence

**Current status:** Notebook 01 has been executed, with **seven saved source-audit outputs** for **six selected ZuCo 2.0 people and twelve NR/TSR runs**. Alignment, usable-window selection, feature extraction, model fitting and final evaluation have **not yet been run**. The recorded Stage-01 results describe data availability and acquisition metadata, not model performance.

## 1. Research Objective and Scope

**Research question:** For paired EEG and gaze windows obtained from ZuCo 2.0, can multimodal predictors distinguish documented reading-task conditions more effectively than gaze-only and EEG-only models under unseen-participant evaluation?

The primary supervised target is **NR versus TSR** (source task protocol). It is not direct comprehension prediction, cognition measurement or a clinical endpoint. We explicitly test whether fusion adds value over each individual modality on the *same eligible windows*, avoiding an automatic assumption that an additional modality improves classification. Advanced fusion is exploratory given the small number of independent people.

The study extends existing eye-tracking infrastructure with EEG reading, source-specific co-registration and grouped inference. No features, labels or results from GazeBase are transferred into this analysis.

## 2. Data Sources, Cohort and Target Definitions

### Sources and access

Source: [ZuCo 2.0](https://www.nemar.org/dataset/nm000279), BIDS version 1.9.0, NEMAR release `nm000279` **v1.0.0**. The source's BIDS description attributes the dataset to Hollenstein, Tröndle, Zhang and Langer (2020), with reported **CC-BY-4.0** terms. The release lists 18 people and nominally 128-channel EEG (EGI) and EyeLink 1000 left-eye data at 500 Hz. Some *per-run* EEG sidecars use different manufacturer/reference strings from general task metadata; preprocessing must follow the run actually read, and the acquisition reference cannot be inferred from the source-level summary.

The repository does not bundle original EEG/eye source signals. The source helper uses the public NEMAR `data.nemar.org/nm000279/v1.0.0/` file gateway to retrieve just named BIDS run files; GitHub's [`nemarDatasets/nm000279`](https://github.com/nemarDatasets/nm000279) index provides the actual file listing. Each selected run needs its BrainVision `.eeg` / `.vhdr` / `.vmrk`, EEG JSON, channel table, event table, eye `physio.tsv.gz` and eye JSON. File retrieval is repeated as needed after Colab Runtime resets. The selected source subset contains **sub-YAC, sub-YAG, sub-YAK, sub-YDG, sub-YDR and sub-YFR**, with **one NR and one TSR run per participant**. Enlarging the cohort changes the experiment and requires regeneration of dependent outputs.

### Cohort and analytical units

The original person is the **independent partition and uncertainty unit**. Each person may contribute paired NR/TSR runs, each run supplies non-overlapping **12-second temporal windows**, and each window supplies **six two-second bins**. Prediction rows are windows, not independent participants; windows from the same person never cross Train/Validation/Test roles.

The initial development partition uses a seeded (`seed=17`) random permutation of eligible people, with 20% Test and 20% of the remainder Validation (minimum one each), leaving the rest for Train. The recorded participant split contains **3 Train, 1 Validation and 2 Test people** (6, 2 and 4 original runs respectively). Actual eligible paired-window counts will only be established after Notebook 02. Small development runs must not be used as evidence of reliable generalisation.

### Inputs and targets

**Task** is read from the BIDS filename/metadata (`nr` / `tsr`), mapped to class indices 0/1. Person ID, run ID, source file name and split role are **provenance/grouping fields**, never predictors. Candidate predictors include gaze coordinates, pupil and vendor fixation events, and EEG bandpower/RMS features within the same valid time window. Source text content is not used as a predictor in the implemented benchmark. Text, task/run duration and other stimulus differences could still confound this classification; a text-only baseline would require an independently specified compatible mapping.

## 3. Data Preparation and Representations

**Observed Stage 01 evidence:** The source index lists **18 participants**, with the first **six** selected for one NR and one TSR run each. All **twelve original run packages** were retrieved, totalling **1,135.6 MB**, and the eight BIDS files per run were recorded as present. All selected EEG/eye sidecars report nominal **500-Hz** sampling and **128 EEG channels**; gaze start offsets range from **−0.046 to +0.010 seconds**. The smallest run package, `sub-YAK / NR` (~**3.95 MB**), has a source EEG duration of about **15.2 seconds**; its actual common EEG–gaze support remains unverified. Seven stage-owned source outputs are available without committing the raw recordings.

Stage 02 reads the lazy BrainVision EEG and compressed EyeLink table for each selected run; timebase construction is **EEG-relative sample time** versus **EyeLink `StartTime + sample_number / gaze_sampling_frequency`**. Only common temporal overlap is considered. Fixation/saccade/blink and trigger rows are checked for plausible onset within recording duration, but the independent inter-device clock drift has **not** been established from shared raw triggers; the report explicitly records `drift_verified=False`. Eligible 12-s windows are non-overlapping, limited to 20/run by default, and must have at least 60% finite gaze x/y samples. Non-eligible windows remain in the audit registry with exclusion reasons.

Stage 03 reads 16 actual EEG channels (`E1` onwards when available, for compact analysis). Per-window channel centring acts as a reference and a 1–40 Hz Butterworth filter is applied to the loaded EEG window. For each two-second bin, Welch spectra yield channel-mean theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) log bandpower and EEG RMS (µV). Eye features are finite-coordinate fraction, median pixel speed, median pupil size and fraction of time overlapping vendor fixation events. These eight features are separate modality groups and each paired window is represented both as a mean-feature vector and as six time bins per modality. All units and missingness are preserved; Train-only scaling/imputation belongs to modelling stages.

A **Train-only**, fixation-locked sample EEG average is saved as an exploratory visual diagnostic. Because reading fixations overlap, ocular artefacts are possible, and no deconvolution or controlled reference contrast is implemented, this plot is **not** a validated ERP/FRP effect or an explanation of reading cognition.

## 4. Experimental Design and Evaluation

The fixed participant grouping is the protection against direct same-person overlap. Every candidate must use identical eligible windows and the same Train/Validation role definition. Class prior, gaze-only, EEG-only, early feature fusion and equal-weight late probability fusion are compared; neural candidates use the same paired temporal arrays.

**Primary selection (before Test):** maximise **fixed-label Validation Macro F1**, then Balanced Accuracy, then fixed candidate priority. The selected model and cohort metadata are saved as `06_model_selection.csv` before any final Test scoring. Test predictions are never used to revise model selection or architecture.

**Iterative execution:** Notebook 06 defaults to `RUN_FINAL_TEST=False` for the initial six-person engineering subset; this saves Validation selection and a limited sensitivity analysis but does not compute Test scores. After an appropriately reviewed and frozen cohort and candidate set, enabling that explicit flag permits one final held-out evaluation. Subsequent tuning using this Test cannot be presented as an independent confirmation.

## 5. Methods, Models and Configurations

| Candidate | Input | Configuration and role |
| --- | --- | --- |
| Class Prior | No physiological predictor | Training class frequency baseline |
| Gaze Logistic / Gaze Forest | Gaze-only mean features | Train-only median imputation; regularised logistic or 150-tree random forest |
| EEG Logistic / EEG Forest | EEG-only spectral/RMS means | Same fixed preprocessing and model families |
| Early Fusion Logistic | Joined EEG+gaze means | Same Train-fitted median/scaler/logistic pipeline |
| Late Equal Fusion | Gaze and EEG logistic probabilities | Fixed `0.5/0.5` mean; no Test-tuned weight |
| Gaze GRU / EEG GRU | Six two-second modality bins | Hidden size 16, dropout 0.2, AdamW, maximal 15 epochs, early stopping on Validation |
| Gated Fusion GRU | Paired EEG and gaze bins | Sigmoid blend of modality embeddings plus compact GRU; gating is not causal attribution |

The temporal models apply Train-only feature-wise medians, means and standard deviations; the saved `05_scaling.npz` supports consistent inference. Neural checkpoint ranking uses Validation Macro F1, Balanced Accuracy and lower Validation cross-entropy. Common seed 17 is recorded. Architectures are intentionally compact for the limited independent subject count and are not guaranteed to outperform classical baselines.

## 6. Results and Comparative Analysis

**Recorded source audit:** six people, twelve downloaded original NR/TSR runs, a fixed 3/1/2 participant split and consistent EEG/eye acquisition metadata. **No EEG–gaze quality, paired-window count, feature extraction or model-evaluation results are yet available.** The next original-data execution is Notebook 02. The full evaluation will eventually report all models' Test Accuracy, fixed-label Macro F1 and Balanced Accuracy with **independent person counts**, correctly separating windows from participants.

A genuine negative fusion outcome (unimodal outperforming fusion) will be reported. The documented published ZuCo analyses do not licence borrowing their scores as results of this pipeline. No empirical fusion gain, high accuracy, sample size or clinical inference is claimed here.

## 7. Robustness, Uncertainty and Additional Analyses

Stage 06 includes a **Validation-only** early-fusion masking check for EEG and gaze feature groups without refitting, used as a descriptive sensitivity probe rather than feature-importance proof. When the final Test has at least three independent people, the selected model's Macro F1 is summarised with a **participant-cluster bootstrap** (500 resamples) instead of treating each 12-s window as independent. Per-person scores are saved for transparency; uncertainty with one or two Test people is not assigned misleading confidence limits.

Task/stimulus confounding, unequal quality, eye artefacts in EEG, reading-event overlap, timebase assumptions, limited electrode coverage and small independent sample size are material limitations. No cross-session or external validation is implemented in the initial six-notebook programme.

## 8. Current Outputs and Reproducibility

| Artefact | Path | Producing stage and intended use |
| --- | --- | --- |
| **Recorded source audit and frozen roles** | `output/01_source_manifest.csv`, `01_split_assignments.csv`, `01_source_eligibility.csv`, `01_download_inventory.csv`, `01_acquisition_metadata.csv`, `01_protocol.json`, `01_source_overview.png` | **Seven Stage-01 files:** original run paths, participant roles, retrieval volumes, sidecar metadata and source figure |
| Window alignment and quality | `output/02_window_registry.csv`, `02_alignment_quality.csv` | Eligible paired time windows and timing audit |
| Independent modality representations | `output/03_features.csv`, `03_temporal_sequences.npz` | Classical predictors and temporal sequences |
| Classical candidate evidence | `output/04_validation_metrics.csv`, `04_validation_predictions.csv`, `04_*.joblib` | Train-fitted baselines, Validation scores |
| Temporal candidate evidence | `output/05_validation_metrics.csv`, `05_validation_predictions.csv`, `05_*.pt`, `05_scaling.npz` | Neural checkpoints and Validation scores |
| Frozen final evidence | `output/06_model_selection.csv`, optional `06_test_metrics.csv`, `06_test_predictions.csv` | Pre-Test primary decision and eventual held-out evaluation |

All six notebooks and their scientific SVGs are authored; **seven Stage-01 source outputs have been committed**, while Stages 02–06 have not been executed on the original data. A stage replaces its named artefacts on rerun, and later stages consume only current named files. Upstream cohort/feature changes require rerunning affected downstream stages. Data access requires the original public dataset and a live Colab Runtime; a GitHub clone alone does not reconstruct raw data or execute the experiment.

## 9. Discussion, Limitations and Future Work

The intended contribution is a transparent, grouped comparison of EEG-only, gaze-only and fusion approaches with explicit source provenance, timebase uncertainty, model-selection discipline and groupwise evaluation. **The scientific comparison is not yet performed**, so no conclusion about fusion improvement is supported.

ZuCo NR and TSR differ in task instructions and likely source-text characteristics, meaning models may detect protocol conditions rather than cognitive processes. Source-file coverage, quality exclusions and independent-person counts must be verified before interpretation. The BIDS eye-stream `StartTime` is a documented offset, not proof of drift-free clock synchronisation. Fixation-locked EEG examples are descriptive and cannot establish cognitive or causal claims. Future work may consider source-matched text controls, independent timing markers, broader channel/montage analysis, additional sessions and genuine external datasets under new protocols.

**Attribution:** Hollenstein, N., Tröndle, M., Zhang, C., & Langer, N. (2020). *ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation*. LREC 2020; [publication](https://aclanthology.org/2020.lrec-1.18/), [NEMAR dataset DOI](https://doi.org/10.82901/nemar.nm000279). BIDS metadata identify the dataset licence as CC BY 4.0.
