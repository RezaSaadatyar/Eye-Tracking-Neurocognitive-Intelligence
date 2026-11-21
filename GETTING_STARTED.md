# S1: Getting Started and Source-File Reuse

This guide describes the current seven-stage S1 workflow in the GitHub repository:

`https://github.com/RezaSaadatyar/Eye-Tracking-Neurocognitive-Intelligence`

The repository contains seven S1 notebooks, study-specific scientific modules (including a record-wise streaming reader), a ZIP restoration helper, seven editable five-band SVG figures, tests and study documentation. Notebook 01 has been executed on ten GazeBase participants; the repository currently also includes 40 original Session 1 task CSVs committed by the operator. The large stage-owned output tables are not committed. Notebook 02 has also been executed on this cohort, with 40 recordings and all 2,778,008 rows retained. Its peak diagnostic has been executed. Notebooks 03 and 04 have now also been executed on this cohort. Notebook 03 produced and displayed its class summary; Notebook 04 produced 40 record-level feature rows, a dictionary and a scanpath figure. Notebook 05 is ready but has not yet run, and Notebooks 06–07 remain pending.

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
3. Notebook 02 has already created `02_preprocessed_samples.csv` (623.4 MiB), `02_preprocessing_summary.csv` and `02_velocity_profile.png`. Its new Section 04A inspects the three largest record-level velocity peaks by reading the saved processed CSV in bounded chunks. It writes `02_velocity_extreme_review.csv` and `02_velocity_extreme_trace.png`. The peak diagnostic has already been run: three selected peaks had 1-ms adjacent intervals and source validity code zero on both neighbouring samples, with reproduced computed maxima. This does not establish physiological movement and does not require a change to the source cohort or preprocessing settings.
4. At the end of each notebook, the unchanged `download_outputs_from_colab` helper creates `eye_tracking_foundations_Notebook_NN_Outputs.zip` in `/content/` for download using Colab Contents in VS Code.
5. If continuing in a new Colab runtime, upload the relevant *current* Notebook ZIP(s) to `/content/` before running the next notebook. The stage input helper loads existing named current outputs or restores the exact matching ZIP only when the required files are absent. It never silently reruns the earlier scientific stage.

Notebook 01 has been executed on original GazeBase recordings in Colab (as shown by the committed output cells). This is not independent reproduction. Notebook 02 has been executed in Colab, as documented by its committed notebook outputs. The peak review was executed, as shown by the committed Notebook 02 outputs. Notebook 03 was also executed and generated 21,728 event intervals from the full sample representation. The Notebook 03 class summary is executed. Notebook 04 has also produced real-cohort feature outputs. The Notebook 05 participant-bootstrap code is authored but not yet executed; synthetic checks do not replace source review. No deep-learning or cognitive target is fitted in S1.

## After Notebook 02 execution

Inspect the recorded 40-record preprocessing summary and the `02_velocity_extreme_review.csv` / `02_velocity_extreme_trace.png` outputs from Section 04A. Check actual `val`, adjacent `Δt`, raw/smoothed jumps and run continuity at the selected peaks. Treat extreme computed speeds as requiring review, not as automatic exclusions. Keep the Notebook 02 ZIP privately; the unchanged `02_preprocessed_samples.csv` remains Notebook 03's input. Notebook 03 has now generated event evidence, but its physiological interpretation remains conditional on the unresolved extreme coordinate steps. The executed Notebook 03 and 04 outputs are already available. Use `01_quality_by_record.csv` and `04_features_by_record.csv` to run Notebook 05, which verifies record identity and source validity before estimating descriptive participant-level intervals. Later stages will assess vendor-reference coverage, event classes and parameter sensitivity without inventing unsupported findings.


## Git output and file-size protection

Generated Notebook CSV, PNG and ZIP files under `eye_tracking_foundations/output/`
are ignored by default, regardless of size. The separate Git blob check blocks
files larger than **50,000,000 bytes (50 MB)** unless the owner deliberately
adds the exact path to `.git-large-file-allowlist`. Enable the repository's
local pre-commit/pre-push hooks once with:

```bash
git config core.hooksPath .githooks
```

The GitHub Actions size check also reports oversized tracked files, but an
unprotected remote branch may still accept a push before that check runs.
To publish a reviewed output smaller than 50 MB, explicitly use `git add -f`
for that specific ignored file. Do not force-add the whole `output/` directory.
