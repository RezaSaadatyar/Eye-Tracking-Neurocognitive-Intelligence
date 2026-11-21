# Eye-Tracking & Neurocognitive Intelligence

A comprehensive, reproducible research and learning framework for eye-tracking, gaze analysis, machine learning, deep learning, and multimodal EEG–eye-tracking.

**Current status:** S1 has seven executed notebooks, original GazeBase analysis outputs in their saved notebook cells, and source-record consistency checks on ten participants. It is a completed **exploratory methodological workflow**, not independently reproduced or clinically/neurocognitively validated. S2–S5 remain separate planned or optional research directions.

## Research programme

The programme begins with time-stamped eye-movement data, proceeds to task-defined modelling, and develops a separate EEG–eye-tracking research pillar. Image-based gaze estimation and neurocognitive applications are independent extensions, not prerequisites for the core eye-movement workflow.

| Study | Research focus | Current status |
| --- | --- | --- |
| [S1: Eye-Tracking Foundations & Signal Analysis](eye_tracking_foundations/README.md) | Data provenance, gaze quality, event detection, features, participant statistics, sensitivity and evidence synthesis | Seven notebooks executed; bounded exploratory results documented |
| S2: Eye-Movement Modelling & Advanced AI | Statistical and machine-learning baselines, temporal deep learning and Transformers | Planned; no study directory yet |
| S3: Vision-Based Gaze Estimation | Camera-image-based gaze direction/point estimation | Optional future study; no study directory yet |
| S4: EEG–Eye-Tracking Multimodal Research | Source-audited EEG/gaze alignment, fixation-related analyses and fusion | Planned second major research pillar; no study directory yet |
| S5: Applied Neurocognitive Studies | Separately defined reading, attention and other cognitive research questions | Future applications, subject to dataset and protocol selection |

These are related research directions, not a single completed experimental pipeline.

## Where to start

1. Read the [S1 overview](eye_tracking_foundations/README.md) and its [methods and evidence document](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md).
2. Consult the [data access and provenance guidance](data/README.md) before supplying a public GazeBase subset.
3. Inspect the [seven-notebook S1 overview](eye_tracking_foundations/images/study_overview.svg), the [executed S1 notebooks](eye_tracking_foundations/code/notebooks/) and the [S1 execution instructions](GETTING_STARTED.md). The latter explain how to recreate named current outputs when required.

The S1 source scope and sample quality are reported for a defined subset, not for all GazeBase recordings. Dataset attribution and redistribution permissions must be checked independently of the repository's MIT licence. EMTeC remains a potential complementary example, not an input to the executed S1 study.

## Repository organisation

| Path | Responsibility |
| --- | --- |
| `eye_tracking_foundations/` | Executed S1 study: seven notebooks, source modules, a study-wide overview, notebook figures and two scientific documents |
| `python_modules/core/` | Shared notebook infrastructure (pinned Human Movement helpers and separate stage ZIP restoration) |
| `data/` | Dataset acquisition and provenance guidance; raw/processed data remain local by default |
| `tests/` | Focused synthetic-only behaviour checks; not original-dataset results |

Additional study folders and genuinely shared scientific utilities will be introduced when justified.

## Scientific and reproducibility principles

- Distinguish measured gaze, derived events, vendor reference labels and cognitive or behavioural outcomes.
- Preserve source, participant, session, trial and sample identity, timestamps, coordinate frames and units.
- Assess missingness, timing, event definitions and within-participant dependence before model comparisons.
- Use explicit named current outputs; downstream stages read saved files or the matching stage ZIP, without hidden reruns.
- Distinguish implementation, synthetic verification, original-data execution and independently supported findings.

## Licensing and data use

Repository code and original documentation remain under the existing [MIT licence](LICENSE). External datasets, third-party code and figures retain their own attribution and licence requirements; this repository does not authorise unrelated redistribution. A selected set of 40 original GazeBase task CSVs is already tracked in this public repository; dataset redistribution rights and attribution must be assessed separately. Generated notebook output tables and ZIPs remain Git-ignored by default; a separate Git blob guard blocks unapproved files above 50 MB.
