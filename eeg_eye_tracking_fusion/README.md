# EEG–Eye-Tracking Multimodal Research

A reproducible six-notebook research workflow, with **Notebook 01 executed on six ZuCo 2.0 participants** and later EEG–gaze processing and modelling not yet executed for evaluating whether simultaneously recorded EEG and eye movements provide complementary information about **ZuCo 2.0 reading-task conditions**. The objective is to compare unimodal and fusion models on the *same* paired temporal windows from participant-disjoint Train, Validation and Test groups. The target is **normal reading (NR) versus task-specific reading (TSR)**, not comprehension, intelligence, a clinical outcome or cognitive ability.

![Overview of EEG and eye-tracking data processing and evaluation](images/study_overview.svg)

*Figure 1. Planned source audit, temporal co-registration, feature processing, unimodal comparators, compact fusion models and one frozen held-out evaluation. No experimental scores are depicted.*

## 1. Dataset and Experimental Protocol

The source is **ZuCo 2.0**, NEMAR dataset [`nm000279`, release `v1.0.0`](https://www.nemar.org/dataset/nm000279), with original BIDS EEG/eye recordings from simultaneous reading tasks. The release documents 18 participants, 128-channel EEG and EyeLink data at nominal 500 Hz. Notebook 01 selected **six participants and twelve original NR/TSR runs**. The number of scientifically eligible paired windows remains undetermined until Notebook 02.

Source acquisition reads only requested BIDS participants and NR/TSR runs using the public NEMAR file gateway. The same code supports an initial six-person development subset and an expanded source cohort by changing `MAX_PARTICIPANTS` (and, if needed, `RUNS_PER_TASK`) in Notebook 01. The complete original archive is not required in advance. Raw signals remain outside the Git repository, in Colab's temporary `/content/zuco2_source_cache` directory.

The initial design uses **one run per task per eligible person**, 12-second non-overlapping paired windows, six two-second feature bins per window, and a seeded **participant-disjoint** Train/Validation/Test split. Run and task labels are verified from BIDS metadata and are never inferred from EEG amplitudes or gaze patterns. Source eligibility rules and scientific boundaries are described in [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

**Important limitations:** identical nominal EEG and gaze sample rates do not prove independent-clock synchronisation. The current mapping uses BIDS `StartTime` and sample frequencies, with drift unverified unless independent markers become available. NR versus TSR differs in instructions and possibly text/stimulus content; performance cannot automatically be attributed to cognitive processing or multimodal complementarity.

## 2. Experimental Programme

All six notebooks are authored. **Notebook 01 has been executed and its seven outputs are committed**; Notebooks 02–06 have not yet been run on original ZuCo signals. They run sequentially in Google Colab. Each stage owns named output files that subsequent stages read; rerunning a stage regenerates its current outputs rather than collecting historical fallback results.

| Stage | Notebook | Scientific role | Intended named outputs |
| --- | --- | --- | --- |
| 01 | [ZuCo Source Audit and Protocol](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) | Inspect BIDS metadata, choose paired NR/TSR runs, download selected raw files and freeze person-level roles | `01_source_manifest.csv`, `01_split_assignments.csv`, `01_protocol.json`, audit tables and source plot |
| 02 | [EEG–Gaze Alignment and Quality](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) | Build paired time windows, inspect sample overlap, event boundaries and missing gaze | `02_window_registry.csv`, `02_alignment_quality.csv`, `02_window_summary.csv`, QA figure |
| 03 | [Preprocessing, Features and Neurophysiology](code/notebooks/03_preprocessing_features_and_neurophysiology.ipynb) | Derive separate EEG/gaze features and temporal arrays; produce an exploratory Train-only fixation-locked EEG diagnostic | `03_features.csv`, `03_temporal_sequences.npz`, feature/quality tables and figures |
| 04 | [Unimodal Baselines and Classical Fusion](code/notebooks/04_unimodal_baselines_and_classical_fusion.ipynb) | Compare task prior, EEG-only, gaze-only, early and fixed late fusion on Validation | `04_validation_metrics.csv`, `04_validation_predictions.csv`, fitted pipelines and plots |
| 05 | [Temporal and Advanced Fusion](code/notebooks/05_temporal_and_advanced_fusion.ipynb) | Compare compact EEG/gaze GRUs and a gated fusion GRU on identical Validation windows | `05_validation_metrics.csv`, `05_validation_predictions.csv`, checkpoints, scaler and histories |
| 06 | [Robustness and Final Evaluation](code/notebooks/06_robustness_and_final_evaluation.ipynb) | Freeze the Validation-selected model before optional final Test; assess participant-level uncertainty | `06_model_selection.csv`, Validation sensitivity, and, when enabled, Test predictions/metrics and figures |

Each notebook also has an editable five-band scientific SVG in [`images/`](images/) using the same role-based typography as the other projects in this repository. The Notebook 01 figure now reports observed source-acquisition counts; figures for later notebooks remain methodological and contain no invented scores.

## 3. Main Results

**Recorded source audit (Notebook 01):** NEMAR listed **18 participants**. Six were selected with NR and TSR Run 1, yielding **12 original source runs** and **1,135.6 MB** of downloaded EEG, EyeLink and BIDS files. The saved split contains **3 Train, 1 Validation and 2 Test people**. Source metadata report **128 EEG channels**, nominal **500-Hz EEG and eye sampling** and eye-start offsets from **−0.046 to +0.010 seconds**. These are acquisition observations, not verified synchronisation or model performance.

The relevant comparison is whether a fusion model improves over the strongest **EEG-only or gaze-only** comparator on *identical*, participant-disjoint examples. A negative or mixed fusion result is scientifically informative and will be reported without changing the primary selection based on the eventual Test scores.

**Pending:** Notebook 02 will determine EEG–gaze overlap, valid gaze fractions and eligible paired-window counts. The short `sub-YAK / NR` source (~**15.2 seconds** according to its EEG sidecar) warrants review. No feature, Validation or Test model results exist yet.

## 4. Running and Reviewing the Study

1. Review the [executed Notebook 01](code/notebooks/01_zuco_source_audit_and_protocol.ipynb) and its seven committed [`01_*` files](output/). The fixed source manifest and split are available without rerunning acquisition.
2. Run [Notebook 02](code/notebooks/02_eeg_gaze_alignment_and_quality.ipynb) using the same six-person cohort and the saved Stage-01 inputs. Original ZuCo EEG/eye files are read from the Colab cache or downloaded again if the Runtime was reset.
3. Inspect the Notebook 02 per-run duration, eye-coverage and eligible-window reports, especially the short NR recording. Then run Notebooks 03–05 on the resulting fixed, participant-disjoint paired windows.
4. For an initial development run, Notebook 06 uses `RUN_FINAL_TEST = False`. It verifies all Validation results and records a primary model **without scoring Test**. Only after finalising the source cohort and model specification should `RUN_FINAL_TEST` be set to `True` for the single final Test evaluation.
5. Review and commit the *named, reasonably sized* output files and executed notebook after each stage. Do not publish raw EEG or eye physiology files to this repository or claim completion of scientific results until they have been checked.

The raw data path is not persistent between Colab sessions. Stage 02/03 source retrieval is intentionally limited to the selected runs rather than the full 18-GB release. GPU use is optional for the compact temporal models; free Colab resource availability is not guaranteed. The original source's attribution and use conditions remain applicable.

## 5. Files and Saved Outputs

```text
eeg_eye_tracking_fusion/
├── README.md
├── RESEARCH_METHODS_AND_EVIDENCE.md
├── requirements.txt
├── code/
│   ├── notebooks/              # Six ordered Colab notebooks
│   └── src/                    # Source, alignment, features and model helpers
├── images/                     # Overview and six role-based SVG figures
├── output/                     # Seven committed source-audit files
└── data/                       # Source cache is external to Git by design
```

Notebook outputs in [`output/`](output/) use distinct `01_*` through `06_*` prefixes. Seven source-acquisition outputs are already committed; later stage files will appear after their respective executions. No separate output-directory README is required. The source cache is `/content/zuco2_source_cache` during Colab execution, not `data/raw` in the tracked repository.

## 6. Interpretation and Scope

This project studies original reading-task labels from paired EEG and eye movements. A classifier may exploit task instructions, source texts, session conditions, eye movements and other nuisance differences; high task discrimination **does not directly demonstrate language comprehension, neurocognitive state, clinical validity or causal EEG contributions**.

Temporal EEG features derive from a fixed 16-channel subset without anatomical localisation claims. A descriptive fixation-locked average does not substitute for a controlled ERP/FRP analysis. Participant-disjoint validation controls person overlap, **not cross-session or stimulus confounding**. Independent timing drift and external generalisation remain unverified unless explicitly assessed.

**Data and attribution:** [ZuCo 2.0 (Hollenstein et al., LREC 2020)](https://aclanthology.org/2020.lrec-1.18/), [NEMAR release nm000279](https://doi.org/10.82901/nemar.nm000279), licence reported in the BIDS metadata as CC BY 4.0. Code and methodological outputs in this repository are separate from the original source recordings.
