# Dataset Access and Provenance

S1 uses original GazeBase source CSV records. As of the current repository revision, the public Git history includes 40 original Round 1 / Session 1 task CSVs (ten participants, four tasks each), committed by the operator. The raw archive is not automatically downloaded. Generated stage-owned tables remain untracked by default; future raw-data publication requires a deliberate rights/provenance decision.

## GazeBase source

- Original dataset: https://doi.org/10.6084/m9.figshare.12912257
- Publication and variable dictionary: https://doi.org/10.1038/s41597-021-00959-y
- Published source file: `GazeBase_v2_0.zip`. The [original GazeBase paper](https://doi.org/10.1038/s41597-021-00959-y) explicitly states the data licence is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The 40 tracked source CSVs are a subset of the archive rather than MIT-licensed code; credit Griffith, Lohr, Abdulin and Komogortsev, link the source and licence, and acknowledge the subset selection.
- Initially selected tasks: `FXS`, `HSS`, `RAN`, and `TEX` from Round 1, Session 1, when available for the same participant.
- Dataset-specific source fields: `n` in ms, `x`/`y` in degrees of visual angle, `val` indicating missing samples, and optional `xT`/`yT`, `dP` (uncalibrated pupil area) and `lab` (EyeLink parser reference labels).
- Source parser labels (when present): `1` fixation, `2` saccade, `-1` blink; unavailable for some tasks. Vendor labels are not independent ground truth.

## Next main-cohort acquisition for Eye-Movement Modelling

The modelling Notebook 01 now defaults to **100 participants** from
GazeBase v2.0 Round 1 / Session 1, four original FXS/HSS/RAN/TEX
recordings per person. The single `MAX_PARTICIPANTS` value can be set
to 300 for a new separately defined run. The selected original files
must all be present and valid; incomplete cohorts are rejected.

If needed, the official 6.7-GB GazeBase archive is downloaded and
verified **inside Colab**, and only requested original CSVs are extracted.
The completed archive is deleted after extraction. Derived outputs
are packaged as one stage ZIP for downstream modelling, not as a raw
source backup. No Google Drive mount or automatic raw-data publication
is used. The 100/300-person original-data run **has not yet occurred**.

## Prepare bounded CSV files

Obtain the original GazeBase source from the official Figshare record. Extract a small number of task CSVs *without renaming them*, retaining, for example, source names of the form `S_1001_S1_FXS.csv`. Place them under:

```text
data/raw/gazebase/GazeBase_v2_0/Round_1/
    Subject_1001/
        S1_Fixations/S_1001_S1_FXS.csv
        S1_Horizontal_Saccades/S_1001_S1_HSS.csv
        S1_Random_Saccades/S_1001_S1_RAN.csv
        S1_Reading/S_1001_S1_TEX.csv
    Subject_1002/ ... Subject_1010/
    Subject_1011/ ... Subject_1020/  # reserved
```

The four displayed filenames for Subject 1001 are present in the current GitHub tree. The loader searches nested directories as well. The Notebook 01 selection requires all four selected task files for a participant; any missing subset produces an actionable error instead of an invented cohort.

Do not extract the entire original archive merely to satisfy a formatting convention. The original compressed archive is large; use a suitable bounded subset, where permitted, and verify its contents. If a full archive must be fetched, plan storage and network requirements first.

## Saved notebook outputs and new Colab runtimes

After each notebook run, the unchanged Human Movement `download_outputs_from_colab()` helper packages that notebook's stage-prefixed current outputs. Download the ZIP from the Colab Contents pane in VS Code. To continue in a fresh runtime, upload the exact upstream ZIP or ZIPs to `/content/` before running the next notebook. A separate reusable helper restores only the named required current-stage files; it does not search historical outputs or rerun previous notebooks.

Each notebook writes its current results into `eye_tracking_foundations/output/`, normally excluded from Git. New or reconstructed source data under `data/raw/` and `data/processed/` are locally ignored by default, but this does not undo the previously committed GazeBase CSVs. For any further publication, check the release licence, attribution requirements and file scope first.

## Notebook 01: ten-participant cohort selection

Notebook 01 now requests up to **ten complete participants** in Round 1, Session 1, each with FXS, HSS, RAN and TEX. Intended source IDs are Subject_1001–Subject_1010. The committed Notebook 01 run selected all ten participants, 40 task files and 2,778,008 source samples. Re-run the producer and its dependent stages only if the source cohort or methods change. Session 2 remains outside the current seven-notebook analysis.

Note that the repository is public: CSVs previously force-added to Git are publicly accessible despite the existing `.gitignore`. Confirm redistribution rights before adding more original CSVs and avoid adding unrelated source materials.

## Prepared GazeBase folders (Round 1)

The repository contains task-directory placeholders for `Subject_1001` through `Subject_1020`. Session 1 CSVs for subjects 1001–1010 are present; the other source slots and Session 2 folders remain reserved. Their common root is:

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

This layout repeats for subjects 1002–1020. The first ten subjects have original Session 1 CSVs; the other prepared folders contain `.gitkeep` placeholders only.

**Selected Session 1 cohort:** The current Notebook 01 selects `session_id=1` and `max_participants=10` with the four required tasks. It has already recorded all ten source participants. Session 2 folders are reserved for a separately designed cross-session study and are not used by these seven notebooks. Expanding to 11–20 participants would require an explicit cohort update and re-execution of dependent stages.

Preserve the exact source CSV names and place each original CSV inside its matching subject/session/task directory. The current source discovery code searches all subdirectories under `data/raw/gazebase/` and matches filenames; it does not need the CSVs flattened into one directory. The example naming convention is `S_1001_S1_FXS.csv`, `S_1001_S1_HSS.csv`, `S_1001_S1_RAN.csv`, and `S_1001_S1_TEX.csv`. Verify names within the real archive; do not rename original source files to match an example.

**Storage and publication:** The placeholders and the explicitly uploaded 40 CSVs are already part of Git history; `/data/raw/` and `/data/processed/` remain excluded from ordinary local `git add` for new files. GitHub's browser upload and `git add -f` can bypass `.gitignore`. Do not add further original files by default. Extract them into a local checkout or the Colab runtime, or keep a private ZIP in supported cloud storage. For Colab the full location begins with `/content/Eye-Tracking-Neurocognitive-Intelligence/data/raw/gazebase/`. The separate `data/processed/gazebase/` location is available for necessary local reconstruction artefacts, not for stage-owned results, which remain in `eye_tracking_foundations/output/`.
