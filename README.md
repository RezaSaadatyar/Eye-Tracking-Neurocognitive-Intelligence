# Eye-Tracking & Neurocognitive Intelligence

**Eye-tracking, gaze signal analysis and neurocognitive AI research**

A Python research programme for source-audited gaze analysis, GazeBase task-recognition modelling, and an authored multimodal EEG–eye-tracking experiment. Eye-Tracking Foundations has seven completed offline notebooks. **Eye-Movement Modelling now has all five notebooks completed on its 100-person main cohort**: 400 source recordings, 9,600 temporal windows and eight candidate models. Logistic Regression was selected on the 16-person Validation set before achieving **80/80 correct on the 20-person main Test** (Macro F1 **1.000**). These are within-GazeBase recording-task results, not cognitive or external-generalisation evidence.

## Research projects

| Project | Current evidence-based status | Start here |
| --- | --- | --- |
| **Eye-Tracking Foundations & Signal Analysis** | **Seven executed notebooks (01–07)** using **10 participants, 40 Round 1 / Session 1 recordings and 2,778,008 source samples**. Source audit, gap-aware preprocessing, provisional I-VT/I-DT events, features, participant statistics, threshold sensitivity and integrated output checks are recorded. Exploratory within-source evidence, not independently validated eye-event accuracy or neurocognitive prediction. | [Study README](eye_tracking_foundations/README.md) · [Scientific methods and evidence](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md) · [Complete workflow diagram](eye_tracking_foundations/images/study_overview.svg) · [How to run](eye_tracking_foundations/README.md#4-running-and-reviewing-the-study) |
| **Eye-Movement Modelling & Advanced AI** | **100-person main study, Notebooks 01–05 complete:** GazeBase Round 1/Session 1, 400 records, 9,600 windows and participant-disjoint **64/16/20** Train/Validation/Test. Eight Validation candidates; **Logistic Regression selected before Test** and achieved main Test **Accuracy 1.000, Macro F1 1.000 (80/80 records from 20 independent people)**. Random Forest also scored 80/80 as a secondary Test comparison. No cross-session, external, cognitive or clinical generalisation demonstrated. | [Study README](eye_movement_modelling/README.md) · [Scientific methods](eye_movement_modelling/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](eye_movement_modelling/images/study_overview.svg) |
| **Vision-Based Gaze Estimation** | **Optional future study:** camera-based gaze direction or point-of-gaze estimation. No completed experiments are claimed. | No study directory yet |
| **EEG–Eye-Tracking Multimodal Research** | **Six-stage expanded 18-person development workflow executed and published:** 36 original NR/TSR runs, 630 paired 12-second EEG/gaze windows and matched features, and **371 Train / 99 Validation / 160 reserved Test** windows. Seven classical/simple-fusion candidates and three GRUs contributed **990 predictions on the same 99 Validation windows from three people**. Stage 06 selected **Gaze Random Forest** using the frozen rule (**pooled Validation Macro F1 0.6244**, Balanced Accuracy **0.6231**, Accuracy **0.6465**), with substantial person-level variation and no observed multimodal gain over the best gaze-only candidate. All **44 compatible stage outputs** are committed, including Stage-06 sensitivity and per-person evidence. **Test remains unscored**, and independent EEG–EyeLink clock drift and cognitive/clinical generalisation are unverified. | [Study README](eeg_eye_tracking_fusion/README.md) · [Scientific methods](eeg_eye_tracking_fusion/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](eeg_eye_tracking_fusion/images/study_overview.svg) · [Executed Notebook 06](eeg_eye_tracking_fusion/code/notebooks/06_robustness_and_final_evaluation.ipynb) |
| **Applied Neurocognitive Research** | **Future studies:** separately defined reading, attention and other research questions, each requiring appropriate source measurements and targets. | No study directory yet |

**Eye-Tracking Foundations** is the first implemented project. Both implemented projects use original GazeBase Round 1, Session 1 source files but answer different research questions and maintain separate derived outputs.

## Selected Eye-Tracking Foundations evidence

The [GazeBase analysis](eye_tracking_foundations/README.md) retains **2,778,008 sample rows across 40 task recordings** from **10 participants** in four tasks: FXS, HSS, RAN and TEX. The executed source-to-evidence sequence generated **7,015 I-VT fixation intervals**, **9,304 I-VT saccade intervals** and **5,409 fixation-only I-DT intervals**. Their total of **21,728** counts method-specific output intervals, **not 21,728 distinct independently confirmed eye movements**. Source identities and valid-gaze fractions were reconciled across the saved producer tables.

The exploratory [participant-level analysis](eye_tracking_foundations/code/notebooks/05_participant_balanced_statistics.ipynb) produced **16 task–feature summaries** using **500 participant bootstrap resamples** (seed 17), with ten independent people contributing to each task. The [I-VT threshold sensitivity analysis](eye_tracking_foundations/code/notebooks/06_detector_sensitivity_and_vendor_reference.ipynb) compared **20, 30 and 40 dva/s** on the same 40 records; jointly evaluable EyeLink comparison samples increased from **1,015,044** to **1,888,902** and **2,357,954**. Conditional agreement cannot be interpreted as a fixed-denominator accuracy comparison or as evidence for choosing the highest threshold.

The [executed integration notebook](eye_tracking_foundations/code/notebooks/07_integrated_evidence_and_reproducibility.ipynb) reconciled the specified source records, participant statistics, original sample counts and threshold summaries. It did **not** reproduce the full pipeline independently. Extremely high calculated gaze speeds, provisional detector definitions, the limited ten-person cohort and vendor-derived rather than expert-adjudicated reference labels constrain scientific interpretation.

For the full protocol, exact numerical findings and limitations, use the [study scientific report](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md) and [seven-notebook diagram](eye_tracking_foundations/images/study_overview.svg).

## Planned research directions

Eye-Movement Modelling has completed the 100-person source-to-Test workflow: fixed participant groups, eight Validation candidates, recorded primary selection and a 20-person main Test. The main Test is now observed and must not guide retrospective model choice. The target remains original recording-task identity, **not cognition**. The **EEG–Eye-Tracking Multimodal Research** study now has published and reconciled results for the expanded **18-person source, alignment, feature, classical, temporal and selection Stages 01–06**: 36 original runs, 630 eligible 12-second windows, 630 paired feature rows and seven classical Validation comparisons. Gaze Random Forest leads the **current classical** candidates with Macro F1 **0.6244** on three held-out people; three GRU models have now also executed (best Gated Fusion GRU Macro F1 **0.6107**); **Stage 06 has now executed and saved the ten-model Gaze Random Forest development selection** (Macro F1 **0.6244**), 30 per-person Validation metrics and three fixed early-fusion masking results. The historical six-person pilot's model ranking is separate. Test remains reserved. Independent clock-drift verification and direct cognitive inference are not established. **Vision-Based Gaze Estimation** remains an independent optional project.

Vision-Based Gaze Estimation and Applied Neurocognitive Research remain future scopes. The EEG–eye-tracking study has six authored notebooks. The **current 18-person execution has completed Stages 01–04**; Stages 05 and 06 remain pending. The earlier six-person pilot completed a separate Validation-only six-notebook development run. **No expanded final Test has been scored.** EMTeC may be considered as a complementary eye-tracking source in a separate approved design; it was **not** used in the completed GazeBase analysis.

## Repository structure

```text
Eye-Tracking-Neurocognitive-Intelligence/
├── README.md
├── GETTING_STARTED.md                  # Source access and execution guide
├── python_modules/core/                # Shared notebook infrastructure
├── data/
│   ├── README.md                       # GazeBase access and provenance
│   ├── raw/gazebase/                    # Selected original source CSVs
│   └── processed/                       # Reserved local preparation area
├── eye_tracking_foundations/
│   ├── README.md                       # Foundations analysis overview
│   ├── RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── code/notebooks/                 # Seven executed Jupyter notebooks
│   ├── code/src/                       # Scientific source and evidence checks
│   ├── images/                         # Overview and notebook scientific SVGs
│   └── output/                         # Derived evidence and numeric outputs
├── eye_movement_modelling/
│   ├── README.md                       # Completed modelling overview
│   ├── RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── code/notebooks/                 # Main Notebooks 01–05 complete
│   ├── code/src/                       # Data, models, training, evaluation
│   ├── images/                         # Overview and notebook SVGs
│   └── output/                         # Model artefacts and final evaluation
├── eeg_eye_tracking_fusion/
│   ├── README.md                       # ZuCo EEG–gaze project introduction
│   ├── RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── requirements.txt
│   ├── code/notebooks/                 # Notebooks 01–06 executed in development
│   ├── code/src/                       # BIDS retrieval, alignment, features, models
│   ├── images/                         # Seven five-band scientific diagrams
│   ├── data/README.md                  # Original ZuCo data source and provenance
│   └── output/                         # 44 current Stage-01–06 expanded evidence files
├── tests/                              # Synthetic scientific software contracts
├── scripts/                            # Git file-size guard
└── .github/workflows/                  # Automated synthetic tests and size policy
```

The optional vision-based and applied research areas do not yet have dedicated project directories. Each project's original data, code and outputs remain scientifically independent.

## Getting started

For Eye-Tracking Foundations, start with the [study README](eye_tracking_foundations/README.md) and [scientific methods](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md). For the completed Eye-Movement Modelling study, use its [README](eye_movement_modelling/README.md), [scientific evidence](eye_movement_modelling/RESEARCH_METHODS_AND_EVIDENCE.md) and [Notebook 05](eye_movement_modelling/code/notebooks/05_robustness_and_final_evaluation.ipynb). For the EEG–eye-tracking research workflow, use its [README](eeg_eye_tracking_fusion/README.md), [principal scientific evidence](eeg_eye_tracking_fusion/RESEARCH_METHODS_AND_EVIDENCE.md) and [executed Notebook 06](eeg_eye_tracking_fusion/code/notebooks/06_robustness_and_final_evaluation.ipynb). All six expanded **development stages** are published, including the ten-model Validation selection; the four reserved Test participants remain unscored.

For original GazeBase acquisition, existing source paths and attribution, consult the [data guide](data/README.md). For VS Code, Colab setup and named output ZIP hand-offs, use [GETTING_STARTED.md](GETTING_STARTED.md). Reviewed study outputs can be explicitly tracked in Git; the repository's independent file-size policy rejects unapproved blobs larger than **50,000,000 bytes**. Large raw/processed data remain locally excluded. ZIP archives made under Colab `/content/` are not automatically published. Re-execution requires the compatible original data and notebook-specific current inputs; a documentation change alone does not require rerunning the experiment.

## Scientific scope, evidence and licensing

- Preserve source recording, participant, task, session and sample identity through analysis.
- Distinguish measured gaze, algorithm-derived eye events, vendor parser annotations and independently observed cognitive outcomes.
- Treat ten people as the independent participants for exploratory uncertainty; repeated task records, samples and events are not additional independent people.
- Distinguish committed execution output, synthetic software tests, internal source-table reconciliation and independent scientific reproduction.
- Do not extend the completed study's findings to clinical diagnosis, generalised cognitive inference or real-time gaze processing.

Original repository code and documentation are covered by the [MIT licence](LICENSE). The original [GazeBase dataset](https://doi.org/10.6084/m9.figshare.12912257) is reported under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) in the [source publication](https://doi.org/10.1038/s41597-021-00959-y). A **40-file GazeBase source subset is already tracked** in this public repository; source credit, licence links and indication of subset selection remain necessary. The dataset is not relicensed under MIT. Reviewed derived outputs may be committed; raw/processed sources remain locally ignored and a separate Git blob check blocks files above **50 MB (50,000,000 bytes)**.
