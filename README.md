# Eye-Tracking & Neurocognitive Intelligence

A comprehensive, reproducible research and learning framework for eye-tracking, gaze analysis, machine learning, deep learning, and multimodal EEG–eye-tracking.

**Current status:** Programme architecture established; the first study scaffold is available. No dataset has been imported, no research notebooks have been implemented or executed, and no scientific results have been reproduced in this repository.

## Research programme

The programme begins with time-stamped eye-movement data, proceeds to task-defined modelling, and develops a separate EEG–eye-tracking research pillar. Image-based gaze estimation and neurocognitive applications are independent extensions, not prerequisites for the core eye-movement workflow.

| Study | Research focus | Current status |
| --- | --- | --- |
| [S1: Eye-Tracking Foundations & Signal Analysis](eye_tracking_foundations/README.md) | Data provenance, gaze quality, eye-movement events, interpretable features and statistics | Scaffold available; implementation not started |
| S2: Eye-Movement Modelling & Advanced AI | Statistical and machine-learning baselines, temporal deep learning and Transformers | Planned; no study directory yet |
| S3: Vision-Based Gaze Estimation | Camera-image-based gaze direction/point estimation | Optional future study; no study directory yet |
| S4: EEG–Eye-Tracking Multimodal Research | Source-audited EEG/gaze alignment, fixation-related analyses and fusion | Planned second major research pillar; no study directory yet |
| S5: Applied Neurocognitive Studies | Separately defined reading, attention and other cognitive research questions | Future applications, subject to dataset and protocol selection |

These are related research directions, not a single completed experimental pipeline. Study names, datasets and experimental designs are refined when each study becomes active.

## Where to start

1. Read the [S1 overview](eye_tracking_foundations/README.md) and its [methods and evidence document](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md).
2. Consult the [data access and provenance guidance](data/README.md) before downloading or processing a public dataset.
3. For now, the `eye_tracking_foundations/code/notebooks/` and `code/src/` locations are reserved for authorised development. There are no runnable notebooks or installed project-specific modules yet.

The proposed first data source is a bounded subset of GazeBase. The exact release, files, licence and data contract require inspection before analysis; EMTeC is a possible complementary example rather than an assumed second input.

## Repository organisation

| Path | Responsibility |
| --- | --- |
| `eye_tracking_foundations/` | First study, with separate overview, detailed scientific document, code, figures and outputs |
| `eye_tracking_foundations/code/notebooks/` | Planned numbered Jupyter notebooks, added as developed |
| `eye_tracking_foundations/code/src/` | Study-specific reusable Python functions, added when needed |
| `eye_tracking_foundations/images/` | Curated scientific illustrations, when produced |
| `eye_tracking_foundations/output/` | Current generated research tables and figures, when produced |
| `data/` | Dataset acquisition guidance; original and processed research data remain local by default |

Additional study folders and genuinely shared utilities will be introduced when their scientific or software responsibilities are established. The scaffold uses `.gitkeep` only for otherwise empty, explicitly approved directories.

## Scientific and reproducibility principles

- Distinguish gaze measurements, derived eye-movement events and behavioural or cognitive outcomes.
- Preserve participant, session, trial, stimulus, timestamp, coordinate-frame and unit information as applicable.
- Assess sampling, data quality, event definitions and dependence between observations before model comparisons.
- Begin with interpretable analyses and appropriate baselines; advanced neural methods are not presumed superior.
- Keep planned methods, implemented code, executed analyses and verified findings clearly separate.
- Preserve current named outputs for downstream stages, without historical approval or hash-gate mechanisms.

## Licensing and data use

Repository code and original documentation are covered by the existing [MIT licence](LICENSE). External datasets, stimuli, third-party code and figures retain their own terms; this repository does not grant redistribution rights for them. Raw recordings, participant-level data, large derivatives and trained artefacts should not be committed merely because they are available locally.
