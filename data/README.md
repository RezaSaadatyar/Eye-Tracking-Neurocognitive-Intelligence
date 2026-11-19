# Dataset Access and Provenance

S1 uses real, original GazeBase CSV recordings when they are supplied by the operator. No raw research dataset is redistributed with this delivery. This repository does not automatically download the large Figshare archive.

## GazeBase source

- Original dataset: https://doi.org/10.6084/m9.figshare.12912257
- Publication and variable dictionary: https://doi.org/10.1038/s41597-021-00959-y
- Published source file: `GazeBase_v2_0.zip`; GazeBase data are reported as CC BY 4.0 in the source publication. Check the current source package and terms before use.
- Initially selected tasks: `FXS`, `HSS`, `RAN`, and `TEX` from Round 1, Session 1, when available for the same participant.
- Dataset-specific source fields: `n` in ms, `x`/`y` in degrees of visual angle, `val` indicating missing samples, and optional `xT`/`yT`, `dP` (uncalibrated pupil area) and `lab` (EyeLink parser reference labels).
- Source parser labels (when present): `1` fixation, `2` saccade, `-1` blink; unavailable for some tasks. Vendor labels are not independent ground truth.

## Prepare bounded CSV files

Obtain the original GazeBase source from the official Figshare record. Extract a small number of task CSVs *without renaming them*, retaining, for example, source names of the form `S_1001_S1_FXS.csv`. Place them under:

```text
data/raw/gazebase/
    S_1001_S1_FXS.csv
    S_1001_S1_HSS.csv
    S_1001_S1_RAN.csv
    S_1001_S1_TEX.csv
    ...
```

The filenames above illustrate the documented naming convention; they do not assert that those particular files have been retrieved or checked. The loader searches nested directories as well. The Notebook 01 selection requires all four selected task files for a participant; any missing subset produces an actionable error instead of an invented cohort.

Do not extract the entire original archive merely to satisfy a formatting convention. The original compressed archive is large; use a suitable bounded subset, where permitted, and verify its contents. If a full archive must be fetched, plan storage and network requirements first.

## Saved notebook outputs and new Colab runtimes

After each notebook run, the unchanged Human Movement `download_outputs_from_colab()` helper packages that notebook's stage-prefixed current outputs. Download the ZIP from the Colab Contents pane in VS Code. To continue in a fresh runtime, upload the exact upstream ZIP or ZIPs to `/content/` before running the next notebook. A separate reusable helper restores only the named required current-stage files; it does not search historical outputs or rerun previous notebooks.

Each notebook writes its current results into `eye_tracking_foundations/output/`, normally excluded from Git. Original and reconstructed datasets under `data/raw/` and `data/processed/` must remain local unless separately reviewed and authorised for public redistribution.

## Prepared GazeBase folders (Round 1)

The repository contains *empty tracked directory placeholders* for `Subject_1001` through `Subject_1020`, with four task folders for Session 1 and four reserved task folders for Session 2. Their common root is:

```text
data/raw/gazebase/GazeBase_v2_0/Round_1/Subject_1001/
    S1_Fixations/
    S1_Horizontal_Saccades/
    S1_Random_Saccades/
    S1_Reading/
    S2_Fixations/
    S2_Horizontal_Saccades/
    S2_Random_Saccades/
    S2_Reading/
```

This layout repeats for subjects 1002–1020. Each task folder currently contains only a `.gitkeep` placeholder. **No GazeBase data have been committed.**

**Prepare Session 1 first:** To run the current seven S1 notebooks, select the four Session 1 tasks from up to ten complete participants, starting with 1001–1010. Later extend to 1011–1020 as needed. The Session 2 folders are reserved for a future explicitly designed cross-session analysis; do not transfer Session 2 yet unless you wish to keep a private local backup. The current Notebook 01 selects `session_id=1` and still has `max_participants=3`; this must be changed to the approved count before analysing ten or more participants.

Preserve the exact source CSV names and place each original CSV inside its matching subject/session/task directory. The current source discovery code searches all subdirectories under `data/raw/gazebase/` and matches filenames; it does not need the CSVs flattened into one directory. The example naming convention is `S_1001_S1_FXS.csv`, `S_1001_S1_HSS.csv`, `S_1001_S1_RAN.csv`, and `S_1001_S1_TEX.csv`. Verify names within the real archive; do not rename original source files to match an example.

**Storage and publication:** The placeholders are intentionally committed for directory visibility, while `/data/raw/` and `/data/processed/` remain excluded from ordinary local `git add` operations. GitHub's browser upload can bypass `.gitignore`: do **not** upload original CSVs to this public repository. Extract them into a local checkout or the Colab runtime, or keep a private ZIP in supported cloud storage. For Colab the full location begins with `/content/Eye-Tracking-Neurocognitive-Intelligence/data/raw/gazebase/`. The separate `data/processed/gazebase/` location is available for necessary local reconstruction artefacts, not for stage-owned results, which remain in `eye_tracking_foundations/output/`.
