# S1: Seven-Notebook Development Delivery

This directory is a **source-only companion copy** of the files for the existing GitHub repository:

`https://github.com/RezaSaadatyar/Eye-Tracking-Neurocognitive-Intelligence`

It contains seven S1 notebooks, six independent scientific modules, one ZIP restoration helper, seven conceptual editable SVG workflow figures, a focused test set and updated S1/data documentation. No original GazeBase recordings or numerical study outputs are distributed.

## Bring the unchanged Human Movement infrastructure into this repository

The exact reference for the shared helpers is Human Movement Intelligence, Project 3 (`multimodal_movement_learning`), commit `67bc0803d2ea3fe0fcfcc618ea5e316d28d8cc29` on 18 November 2026. If preparing this source-only archive independently of the GitHub checkout, run the included one-time importer in a network-enabled terminal to obtain the same unchanged helpers. If cloning the updated GitHub repository, the shared helpers are already present and this command is not required:

```bash
python IMPORT_EXACT_HUMAN_MOVEMENT_HELPERS.py
```

This copies `colab_setup.py`, `repository_sync.py`, `plot_style.py`, `output_download.py`, `computer_device.py` and `__init__.py` **byte-for-byte** from the pinned reference revision. It does not copy NinaPro scientific algorithms, models, data, settings or results. The copied setup helper expects `requirements.txt` at repository root; this delivery includes that file. The original helper has a historical repository-specific optional `data_mode="refresh"` branch for a different study: do **not** use that mode in S1. The notebooks use its standard default `setup_colab("eye_tracking_foundations")` call and do not modify the helper code.

The imported output-download helper packages named current outputs into a stage ZIP for manual download in VS Code's Colab connection. The separately authored `stage_outputs.py` can restore one explicitly named current-stage ZIP at the start of a later notebook; this is not an edit to the original helper.

## Placement and publication

Merge the files in this overlay into the existing checkout with the same relative paths, preserving the existing MIT `LICENSE` and all unrelated files. Do not delete existing files merely because they are not listed in this delivery. The Colab checkout reads the selected remote repository revision, so locally edited notebook and module files must also be available in that checkout before execution. The archive itself does not change the remote revision when copied to a different machine.

Inspect the changes, dependencies and staged file sizes before any authorised publication; do not publish `data/raw/`, private records, or generated research outputs indiscriminately.

## Data and execution

1. Obtain an authorised GazeBase subset from the [official Figshare release](https://doi.org/10.6084/m9.figshare.12912257). The archive is large; do not treat it as a small automatic Colab download.
2. Preserve the original CSV filenames and place selected files under `data/raw/gazebase/` in the Colab checkout. Four tasks (FXS/HSS/RAN/TEX) from Round 1 and Session 1 for the same participant are expected by the initial Notebook 01 selection. The loader accepts nested source folders.
3. Open `01_source_audit_and_gaze_quality.ipynb` in VS Code connected to Colab, then run top-to-bottom. Repeat for Notebook 02 through 07, in order. Do not manually insert invented missing-value or parser-label fields.
4. At the end of each notebook, the unchanged `download_outputs_from_colab` helper creates `eye_tracking_foundations_Notebook_NN_Outputs.zip` in `/content/` for download using Colab Contents in VS Code.
5. If continuing in a new Colab runtime, upload the relevant *current* Notebook ZIP(s) to `/content/` before running the next notebook. The stage input helper loads existing named current outputs or restores the exact matching ZIP only when the required files are absent. It never silently reruns the earlier scientific stage.

Notebook 01 does not contain a one-click download of the full GazeBase archive. No code has been executed against actual GazeBase files in the development environment: the executed pipeline verification used synthetic CSV records, and the five contract tests use synthetic fixtures. These checks are not scientific results.

## After real-data execution

Inspect source-file counts, sample interval anomalies, validity, vendor-reference coverage, event classes, parameter sensitivity, participant support, stage output consistency, and chart captions. Update the S1 README and methods/evidence document from the actual saved results rather than from this plan.
