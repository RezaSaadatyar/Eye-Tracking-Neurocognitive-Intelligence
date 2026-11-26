# Eye-Tracking & Neurocognitive Intelligence

**Eye-tracking, biomedical signal analysis and neurocognitive AI**

A Python research programme exploring gaze signals, eye-movement modelling, EEG–gaze fusion, image-based gaze estimation and reading-comprehension prediction. Five independent studies use four research datasets and 28 notebooks. Each study provides its own methods, experimental evidence and scientific limitations.

## Research projects

| Study | Dataset and research scope | Explore |
| --- | --- | --- |
| **01. Eye-Tracking Foundations & Signal Analysis** | GazeBase: gap-aware preprocessing, provisional eye-event detection, scanpaths and participant-level statistics | [Overview](eye_tracking_foundations/README.md) · [Methods and evidence](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](eye_tracking_foundations/images/study_overview.svg) |
| **02. Eye-Movement Modelling & Advanced AI** | GazeBase: classical and temporal models for recording-task classification | [Overview](eye_movement_modelling/README.md) · [Methods and evidence](eye_movement_modelling/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](eye_movement_modelling/images/study_overview.svg) |
| **03. EEG–Eye-Tracking Multimodal Research** | ZuCo 2.0: EEG-only, gaze-only and fusion models for reading-task classification | [Overview](eeg_eye_tracking_fusion/README.md) · [Methods and evidence](eeg_eye_tracking_fusion/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](eeg_eye_tracking_fusion/images/study_overview.svg) |
| **04. Vision-Based Gaze Estimation** | MPIIGaze: classical and neural estimation of gaze direction from eye images | [Overview](vision_based_gaze_estimation/README.md) · [Methods and evidence](vision_based_gaze_estimation/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](vision_based_gaze_estimation/images/study_overview.svg) |
| **05. Reading Comprehension & Neurocognitive Modelling** | OneStop: pre-answer gaze and paragraph-text representations for response-correctness prediction | [Overview](reading_comprehension_modelling/README.md) · [Methods and evidence](reading_comprehension_modelling/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](reading_comprehension_modelling/images/study_overview.svg) |

## Key findings

| Study | Analytical cohort | Main evidence |
| --- | --- | --- |
| Foundations | 10 participants; 40 recordings; 2,778,008 samples | 7,015 I-VT fixation, 9,304 I-VT saccade and 5,409 I-DT fixation-only intervals. Method-specific intervals, not independently confirmed eye movements. |
| Eye-movement modelling | 100 participants; 400 recordings; 9,600 windows | Validation-selected Logistic Regression: Test Macro F1 **1.000** (80/80 recordings; 20 held-out participants). Within-GazeBase task classification. |
| EEG–gaze fusion | 18 participants; 630 paired windows | Validation-selected Gaze Random Forest: Validation Macro F1 **0.6244**, Test Macro F1 **0.4997** (160 windows; four people). Fusion did not consistently outperform gaze alone. |
| Vision | 15 participants; 7,500 images; 15 outer LOPO folds | Small CNN: participant-macro mean angular error **7.3048°**. HOG + Ridge: 7.5596°; the paired difference interval includes zero. |
| Reading comprehension | 180 participants; 9,718 trials; 30 articles | Validation-selected TF-IDF fusion (A) and TF-IDF text (B): Test Macro F1 **0.5552** and **0.5035**, respectively. Added gaze is model- and protocol-dependent. |

## Repository structure

```text
Eye-Tracking-Neurocognitive-Intelligence/
├── README.md
├── data/                         # Dataset access and selected GazeBase recordings
├── python_modules/core/          # Shared notebook utilities
├── requirements.txt              # Foundations and eye-movement modelling
├── eye_tracking_foundations/
├── eye_movement_modelling/
├── eeg_eye_tracking_fusion/
├── vision_based_gaze_estimation/
├── reading_comprehension_modelling/
├── tests/                        # Synthetic software/data contracts
└── LICENSE
```

Each study contains a README, scientific methods and evidence, notebooks, source modules, scientific diagrams and derived outputs. Study-specific requirements are provided where needed.

## Explore and reproduce the research

Clone the repository and choose a study from the table above. Its README describes the notebook sequence, required inputs and results. The [dataset guide](data/README.md) identifies original sources and access conditions. The notebooks contain environment setup and analysis code; install the root or study-specific requirements as appropriate. Review the saved results without retraining, or run the relevant notebook sequence in order to regenerate the required inputs and outputs. The ZuCo saved classical pipelines require scikit-learn 1.6.1.

## Scientific scope and future directions

The studies address distinct analytical questions. Repeated windows, images or trials do not increase the independent participant count. Task classification and response correctness are not direct measures of general cognition. The findings are internal to their source datasets; external generalisation, clinical utility and independent original-data reproduction are not established. Further work may examine independent eye-event annotations, cross-session transfer, stimulus-controlled EEG–gaze comparisons, external gaze estimation and text-disjoint reading evaluation.

## References and licensing

Repository code and documentation use the [MIT licence](LICENSE); original data and data-derived materials remain subject to their source terms. See the [dataset guide](data/README.md) and the individual scientific documents for source attribution and restrictions.
