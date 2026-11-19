# S1: Getting Started and Source-File Reuse

This guide describes the current seven-stage S1 workflow in the GitHub repository:

`https://github.com/RezaSaadatyar/Eye-Tracking-Neurocognitive-Intelligence`

The repository contains seven S1 notebooks, study-specific scientific modules (including a record-wise streaming reader), a ZIP restoration helper, seven editable five-band SVG figures, tests and study documentation. Notebook 01 has been executed on ten GazeBase participants; the repository currently also includes 40 original Session 1 task CSVs committed by the operator. The large stage-owned output tables are not committed. Notebooks 02–07 have not yet been executed on this cohort.

## Bring the unchanged Human Movement infrastructure into this repository

The exact reference for the shared helpers is Human Movement Intelligence, Project 3 (`multimodal_movement_learning`), commit `67bc0803d2ea3fe0fcfcc618ea5e316d28d8cc29` on 18 November 2026. If preparing a standalone source overlay without the shared helpers, the included one-time importer can obtain the same unchanged helpers. For a current clone of this GitHub repository, they are already present and this command is not required:

```bash
python IMPORT_EXACT_HUMAN_MOVEMENT_HELPERS.py
```

This copies `colab_setup.py`, `repository_sync.py`, `plot_style.py`, `output_download.py`, `computer_device.py` and `__init__.py` **byte-for-byte** from the pinned reference revision. It does not copy NinaPro scientific algorithms, models, data, settings or results. The copied setup helper expects `requirements.txt` at repository root; this delivery includes that file. The original helper has a historical repository-specific optional `data_mode="refresh"` branch for a different study: do **not** use that mode in S1. The notebooks use its standard default `setup_colab("eye_tracking_foundations")` call and do not modify the helper code.

The imported output-download helper packages named current outputs into a stage ZIP for manual download in VS Code's Colab connection. The separately authored `stage_outputs.py` can restore one explicitly named current-stage ZIP at the start of a later notebook; this is not an edit to the original helper.

## Placement and publication

The Colab checkout reads the selected GitHub repository revision. Use a fresh checkout or the existing notebook's safe repository update before using new code; do not mix old loaded modules with updated source files. Keep the original MIT `LICENSE`, project outputs and unrelated files unchanged.

Inspect the changes, dependencies and staged file sizes before any authorised publication; do not publish `data/raw/`, private records, or generated research outputs indiscriminately.

## Data and execution

1. The current public source checkout already contains 40 original GazeBase Session 1 CSVs (FXS/HSS/RAN/TEX for subjects 1001–1010). If using a private checkout, retain the same folder structure and original filenames. The official release is [Figshare GazeBase](https://doi.org/10.6084/m9.figshare.12912257); no large archive download occurs automatically.
2. Notebook 01 has a committed real-data execution with 10 participants, 40 recordings and 2,778,008 sample rows. Do not rerun it to prepare Notebook 02 if its compatible current sample CSV is already available.
3. Open Notebook 02 in VS Code connected to Colab, run it top-to-bottom, and inspect the stage-02 summary and velocity figure before proceeding to Notebook 03. The stage-02 reader consumes `01_gaze_samples.csv` and `01_source_manifest.csv` and writes `02_preprocessed_samples.csv` one complete record at a time to limit memory use.
4. At the end of each notebook, the unchanged `download_outputs_from_colab` helper creates `eye_tracking_foundations_Notebook_NN_Outputs.zip` in `/content/` for download using Colab Contents in VS Code.
5. If continuing in a new Colab runtime, upload the relevant *current* Notebook ZIP(s) to `/content/` before running the next notebook. The stage input helper loads existing named current outputs or restores the exact matching ZIP only when the required files are absent. It never silently reruns the earlier scientific stage.

Notebook 01 has been executed on original GazeBase recordings in Colab (as shown by the committed output cells). This is not independent reproduction. The current stage-02 implementation has been checked with synthetic sample records, not executed on the actual ten-participant source; review its real output and RAM usage when running it. No deep-learning or cognitive target is fitted in S1.

## After Notebook 02 execution

Inspect the 40-record preprocessing summary, the total sample count, valid-run counts, undefined boundary velocities, example raw-versus-smoothed gaze and saved output size. Stage-02 outputs are local by default; retain the ZIP for later notebooks. Only after that review should Notebook 03 event detection be run. Later stages will assess vendor-reference coverage, event classes and parameter sensitivity without inventing unsupported findings.
