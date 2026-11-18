# Eye-Tracking & Neurocognitive Intelligence

A comprehensive, reproducible research and learning framework for eye-tracking, gaze analysis, machine learning, deep learning, and multimodal EEG–eye-tracking.

**Current status:** S1 has seven authored development notebooks and source modules, with syntax and synthetic data-flow checks completed. **No original GazeBase analysis results or independent scientific reproductions are claimed.** The other study families remain planned or optional.

## Research programme

The programme begins with time-stamped eye-movement data, proceeds to task-defined modelling, and develops a separate EEG–eye-tracking research pillar. Image-based gaze estimation and neurocognitive applications are independent extensions, not prerequisites for the core eye-movement workflow.

| Study | Research focus | Current status |
| --- | --- | --- |
| [S1: Eye-Tracking Foundations & Signal Analysis](eye_tracking_foundations/README.md) | Data provenance, gaze quality, eye-movement events, features, statistics and sensitivity | Seven notebooks authored; awaiting genuine source execution |
| S2: Eye-Movement Modelling & Advanced AI | Statistical and machine-learning baselines, temporal deep learning and Transformers | Planned; no study directory yet |
| S3: Vision-Based Gaze Estimation | Camera-image-based gaze direction/point estimation | Optional future study; no study directory yet |
| S4: EEG–Eye-Tracking Multimodal Research | Source-audited EEG/gaze alignment, fixation-related analyses and fusion | Planned second major research pillar; no study directory yet |
| S5: Applied Neurocognitive Studies | Separately defined reading, attention and other cognitive research questions | Future applications, subject to dataset and protocol selection |

These are related research directions, not a single completed experimental pipeline.

## Where to start

1. Read the [S1 overview](eye_tracking_foundations/README.md) and its [methods and evidence document](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md).
2. Consult the [data access and provenance guidance](data/README.md) before supplying a public GazeBase subset.
3. Read the [S1 execution instructions](GETTING_STARTED.md), prepare the unchanged Colab/shared helpers from the pinned Human Movement source, and execute the [numbered S1 notebooks](eye_tracking_foundations/code/notebooks/) in order.

The initial GazeBase files, licence and sample quality must be checked against the actually selected source. EMTeC remains a potential complementary example, not an assumed second input.

## Repository organisation

| Path | Responsibility |
| --- | --- |
| `eye_tracking_foundations/` | First study: seven numbered notebooks, source modules, workflow figures, current outputs and two scientific documents |
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

Repository code and original documentation remain under the existing [MIT licence](LICENSE). External datasets, third-party code and figures retain their own attribution and licence requirements; this repository does not authorise unrelated redistribution. Original participant-level data and model artefacts are not automatically published.
