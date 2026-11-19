# GazeBase Source-Folder Guide

This folder is a **directory scaffold, not a distributed dataset**. The canonical version directory is `GazeBase_v2_0/Round_1/`. Subject directories 1001–1020 each have the following placeholder folders for potential private source CSVs:

- `S1_Fixations/` → FXS
- `S1_Horizontal_Saccades/` → HSS
- `S1_Random_Saccades/` → RAN
- `S1_Reading/` → TEX
- Matching `S2_...` folders are reserved; no current S1 notebook consumes Session 2.

**Upload Session 1 only for now.** First prepare subjects 1001–1010 with all four tasks. Later expand to 1011–1020 if needed. The code must be adjusted from `max_participants=3` before processing more subjects.

Inside each task folder, preserve the actual original CSV basename (for example `S_1001_S1_FXS.csv`), without inventing or renaming fields. The source reader discovers files recursively and selects `round_id=1,session_id=1`.

These task folders contain only `.gitkeep` files in GitHub. They are **not evidence that any subject or task CSV has been acquired**. Never upload raw datasets to this public GitHub repository using the web UI; `.gitignore` does not block web UI uploads. For detailed handling, see [the data guide](../../README.md).
