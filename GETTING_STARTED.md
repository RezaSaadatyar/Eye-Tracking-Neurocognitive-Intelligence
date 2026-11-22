# Getting Started | Eye-Tracking & Neurocognitive Intelligence

This guide covers the seven-notebook Eye-Tracking Foundations & Signal Analysis workflow and source-file reuse in the GitHub repository:

`https://github.com/RezaSaadatyar/Eye-Tracking-Neurocognitive-Intelligence`

The Eye-Tracking Foundations & Signal Analysis project contains seven notebooks, study-specific scientific modules (including a record-wise streaming reader), a ZIP restoration helper, seven editable five-band SVG figures, tests and study documentation. Notebook 01 has been executed on ten GazeBase participants; the repository currently also includes 40 original Session 1 task CSVs committed by the operator. The large stage-owned output tables are not committed. Notebook 02 has also been executed on this cohort, with 40 recordings and all 2,778,008 rows retained. Its peak diagnostic has been executed. Notebooks 03 and 04 have now also been executed on this cohort. Notebook 03 produced and displayed its class summary; Notebook 04 produced 40 record-level feature rows, a dictionary and a scanpath figure. Notebook 05 has now executed: 40 aligned source records, ten people and 16 participant-balanced statistics, with exploratory intervals. Its updated zoomed plot and output ZIP were rerun. Notebook 06 has completed its 120 source/threshold calculations, compact reference-support checks and revised plot; its ZIP was recorded. Notebook 07 has also executed all sections and produced its alignment table, one-row evidence summary and three-panel figure. The seven-notebook sequence is now executed, while scientific final review and broader repository closure remain separate.

## Bring the unchanged Human Movement infrastructure into this repository

The exact reference for the shared helpers is Human Movement Intelligence, Project 3 (`multimodal_movement_learning`), commit `67bc0803d2ea3fe0fcfcc618ea5e316d28d8cc29` on 18 November 2026. If preparing a standalone source overlay without the shared helpers, the included one-time importer can obtain the same unchanged helpers. For a current clone of this GitHub repository, they are already present and this command is not required:

```bash
python IMPORT_EXACT_HUMAN_MOVEMENT_HELPERS.py
```

This copies `colab_setup.py`, `repository_sync.py`, `plot_style.py`, `output_download.py`, `computer_device.py` and `__init__.py` **byte-for-byte** from the pinned reference revision. It does not copy NinaPro scientific algorithms, models, data, settings or results. The copied setup helper expects `requirements.txt` at repository root; this delivery includes that file. The original helper has a historical repository-specific optional `data_mode="refresh"` branch for a different study: do **not** use that mode in Eye-Tracking Foundations & Signal Analysis. The notebooks use its standard default `setup_colab("eye_tracking_foundations")` call and do not modify the helper code.

The imported output-download helper packages named current outputs into a stage ZIP for manual download in VS Code's Colab connection. The separately authored `stage_outputs.py` can restore one explicitly named current-stage ZIP at the start of a later notebook; this is not an edit to the original helper.

## Placement and publication

The Colab checkout reads the selected GitHub repository revision. Use a fresh checkout or the existing notebook's safe repository update before using new code; do not mix old loaded modules with updated source files. Keep the original MIT `LICENSE`, project outputs and unrelated files unchanged.

Inspect the changes, dependencies and staged file sizes before any authorised publication; do not publish `data/raw/`, private records, or generated research outputs indiscriminately.

## Data and execution

1. The current public source checkout already contains 40 original GazeBase Session 1 CSVs (FXS/HSS/RAN/TEX for subjects 1001–1010). If using a private checkout, retain the same folder structure and original filenames. The official release is [Figshare GazeBase](https://doi.org/10.6084/m9.figshare.12912257); no large archive download occurs automatically.
2. Notebook 01 has a committed real-data execution with 10 participants, 40 recordings and 2,778,008 sample rows. Do not rerun it to prepare Notebook 02 if its compatible current sample CSV is already available.
3. Notebook 02 has already created `02_preprocessed_samples.csv` (623.4 MiB), `02_preprocessing_summary.csv` and `02_velocity_profile.png`. Its new Section 04A inspects the three largest record-level velocity peaks by reading the saved processed CSV in bounded chunks. It writes `02_velocity_extreme_review.csv` and `02_velocity_extreme_trace.png`. The peak diagnostic has already been run: three selected peaks had 1-ms adjacent intervals and source validity code zero on both neighbouring samples, with reproduced computed maxima. This does not establish physiological movement and does not require a change to the source cohort or preprocessing settings.
4. At the end of each notebook, the unchanged `download_outputs_from_colab` helper creates `eye_tracking_foundations_Notebook_NN_Outputs.zip` in `/content/` for download using Colab Contents in VS Code.
5. If continuing in a new Colab runtime, upload the relevant *current* Notebook ZIP(s) to `/content/` before running the next notebook. The stage input helper loads existing named current outputs or restores the exact matching ZIP only when the required files are absent. It never silently reruns the earlier scientific stage.

Notebook 01 has been executed on original GazeBase recordings in Colab (as shown by the committed output cells). This is not independent reproduction. Notebook 02 has been executed in Colab, as documented by its committed notebook outputs. The peak review was executed, as shown by the committed Notebook 02 outputs. Notebook 03 was also executed and generated 21,728 event intervals from the full sample representation. The Notebook 03 class summary is executed. Notebook 04 has also produced real-cohort feature outputs. The Notebook 05 participant-bootstrap code has executed and its 16 statistical outputs are recorded. The updated Notebook 05 plot and ZIP have been rerun. Notebook 06 has saved three-threshold source results, threshold-dependent denominator support and successful 30-dva/s baseline continuity. Notebook 07 has completed the compact cross-notebook source-alignment and evidence-summary checks. Synthetic checks do not replace full original-data reproduction. No deep-learning or cognitive target is fitted in Eye-Tracking Foundations & Signal Analysis.

## After Notebook 02 execution

Inspect the recorded 40-record preprocessing summary and the `02_velocity_extreme_review.csv` / `02_velocity_extreme_trace.png` outputs from Section 04A. Check actual `val`, adjacent `Δt`, raw/smoothed jumps and run continuity at the selected peaks. Treat extreme computed speeds as requiring review, not as automatic exclusions. Keep the Notebook 02 ZIP privately; the unchanged `02_preprocessed_samples.csv` remains Notebook 03's input. Notebook 03 has now generated event evidence, but its physiological interpretation remains conditional on the unresolved extreme coordinate steps. The executed Notebook 03 and 04 outputs are already available. Notebook 05 has already used `01_quality_by_record.csv` and `04_features_by_record.csv` to verify record identity and source validity (maximum discrepancy 0.0) and estimate descriptive participant-level intervals. Notebook 05 is complete for the current scope and should not be rerun. Notebook 06 and Notebook 07 have now both executed successfully. Do not rerun their calculations solely to refresh the Markdown or introductory SVG diagrams: these changes preserve the saved numerical evidence. Notebook 06 owns `06_sensitivity.csv`, `06_sensitivity_summary.csv`, `06_reference_support.csv` and `06_threshold_sensitivity.png`. Notebook 07 owns `07_source_alignment.csv`, `07_evidence_summary.csv` and `07_study_overview.png`. These outputs should be retained privately for scientific final review. If a future input cohort or calculation changes, rerun the affected producer and its dependent consumers using the named current files. If switching runtimes, place the needed current Notebook ZIPs in `/content/` before starting.


## Git output and file-size protection

Reviewed derived outputs under `eye_tracking_foundations/output/` and
`eye_movement_modelling/output/` may now be staged in Git when appropriate.
The `.gitignore` file cannot filter by file size; a separate policy checks
Git blobs and blocks files above **50,000,000 bytes (50 decimal MB)** unless
explicitly approved. Certain large Eye-Tracking Foundations sample CSVs remain ignored by
their exact filenames. Raw/processed source data also remain ignored.

Before publishing selected results:

```bash
git add eye_movement_modelling/output/
python scripts/check_git_file_sizes.py --staged
```

Unstage oversized or unsuitable files using `git restore --staged <path>`.
The separately maintained local hooks can be enabled with:

```bash
git config core.hooksPath .githooks
```

GitHub Actions runs the file-size policy as well, but an unprotected
branch may accept a push before CI reports a problem. Do not commit
unauthorised source records, private data or unnecessarily large archives.

## Eye-Movement Modelling & Advanced AI

The [study README](eye_movement_modelling/README.md), [scientific methods and evidence](eye_movement_modelling/RESEARCH_METHODS_AND_EVIDENCE.md) and [five notebooks](eye_movement_modelling/code/notebooks/) document the **completed 100-person GazeBase Round 1 / Session 1 study**. The four-class recording-task target is FXS/HSS/RAN/TEX, not cognition or clinical status.

The fixed protocol selected **100 complete participants and 400 task recordings** and produced **9,600 normalised windows**. Independent-person Train/Validation/Test partitions contain **64/16/20 people** (256/64/80 recordings). Eight models were compared on the same 64 Validation records, and Logistic Regression was selected before scoring the held-out Test. The primary achieved **Accuracy and Macro F1 1.000 (80/80 correct across 20 Test people)**. This is within-GazeBase task classification, not cross-session or external generalisation.

All named current `01_*`–`05_*` outputs, including models and evaluation tables, are in [`eye_movement_modelling/output/`](eye_movement_modelling/output/). A clean checkout can inspect them directly. To reproduce the pipeline, run the five notebooks in numerical order with the same source cohort and settings; each stage writes its named outputs consumed by later stages. The original GazeBase source archive and raw recordings are distinct from the saved derived outputs; follow the [data provenance guide](data/README.md) for original source access.

The main held-out Test has already been evaluated. Any new cohort, model redesign informed by these Test outcomes, or different acquisition protocol requires a **separately defined evaluation**, rather than treating the recorded Test as untouched evidence.
