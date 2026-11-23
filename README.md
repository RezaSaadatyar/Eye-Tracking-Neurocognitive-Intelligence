# Eye-Tracking & Neurocognitive Intelligence

**Eye-tracking, gaze signal analysis and neurocognitive AI research**

A Python research programme covering **three implemented, scientifically independent studies**: GazeBase eye-movement signal analysis, GazeBase recording-task modelling, and ZuCo 2.0 EEG–eye-tracking fusion. **Eye-Tracking Foundations** has seven executed notebooks. **Eye-Movement Modelling** has five completed notebooks on 100 participants; its Validation-selected Logistic Regression classified **80/80 main Test records from 20 people** correctly (Macro F1 **1.000**) within the GazeBase task setting. **EEG–Eye-Tracking Multimodal Research** has six executed notebooks on 18 ZuCo 2.0 participants, with Gaze Random Forest selected on three-person Validation (Macro F1 **0.6244**); its four-person final Test remains unscored. Recording-task classification in either dataset is not direct evidence of cognition or external generalisation.

## Research projects

| Project | Current evidence-based status | Start here |
| --- | --- | --- |
| **Eye-Tracking Foundations & Signal Analysis** | **Seven executed notebooks (01–07)** using **10 participants, 40 Round 1 / Session 1 recordings and 2,778,008 source samples**. Source audit, gap-aware preprocessing, provisional I-VT/I-DT events, features, participant statistics, threshold sensitivity and integrated output checks are recorded. Exploratory within-source evidence, not independently validated eye-event accuracy or neurocognitive prediction. | [Study README](eye_tracking_foundations/README.md) · [Scientific methods and evidence](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md) · [Complete workflow diagram](eye_tracking_foundations/images/study_overview.svg) · [How to run](eye_tracking_foundations/README.md#4-running-and-reviewing-the-study) |
| **Eye-Movement Modelling & Advanced AI** | **100-person main study, Notebooks 01–05 complete:** GazeBase Round 1/Session 1, 400 records, 9,600 windows and participant-disjoint **64/16/20** Train/Validation/Test. Eight Validation candidates; **Logistic Regression selected before Test** and achieved main Test **Accuracy 1.000, Macro F1 1.000 (80/80 records from 20 independent people)**. Random Forest also scored 80/80 as a secondary Test comparison. No cross-session, external, cognitive or clinical generalisation demonstrated. | [Study README](eye_movement_modelling/README.md) · [Scientific methods](eye_movement_modelling/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](eye_movement_modelling/images/study_overview.svg) |
| **EEG–Eye-Tracking Multimodal Research** | **Six-stage expanded 18-person development workflow executed and published:** 36 original NR/TSR runs, 630 paired 12-second EEG/gaze windows and matched features, and **371 Train / 99 Validation / 160 reserved Test** windows. Seven classical/simple-fusion candidates and three GRUs contributed **990 predictions on the same 99 Validation windows from three people**. Stage 06 selected **Gaze Random Forest** using the frozen rule (**pooled Validation Macro F1 0.6244**, Balanced Accuracy **0.6231**, Accuracy **0.6465**), with substantial person-level variation and no observed multimodal gain over the best gaze-only candidate. All **44 compatible stage outputs** are committed, including Stage-06 sensitivity and per-person evidence. **Test remains unscored**, and independent EEG–EyeLink clock drift and cognitive/clinical generalisation are unverified. | [Study README](eeg_eye_tracking_fusion/README.md) · [Scientific methods](eeg_eye_tracking_fusion/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](eeg_eye_tracking_fusion/images/study_overview.svg) · [Executed Notebook 06](eeg_eye_tracking_fusion/code/notebooks/06_robustness_and_final_evaluation.ipynb) |
| **Vision-Based Gaze Estimation** | **Optional future study:** camera-based gaze direction or point-of-gaze estimation. No completed experiments are claimed. | No study directory yet |
| **Applied Neurocognitive Research** | **Future studies:** separately defined reading, attention and other research questions, each requiring appropriate source measurements and targets. | No study directory yet |

**Implemented study boundaries:** Eye-Tracking Foundations and Eye-Movement Modelling both use GazeBase Round 1 / Session 1 but ask different questions and retain separate outputs. EEG–Eye-Tracking Multimodal Research uses **ZuCo 2.0** with its own recordings, participant split, model candidates and evidence. The vision-based and applied-neurocognitive directions are not yet implemented studies.

## Selected Eye-Tracking Foundations evidence

The [GazeBase analysis](eye_tracking_foundations/README.md) retains **2,778,008 sample rows across 40 task recordings** from **10 participants** in four tasks: FXS, HSS, RAN and TEX. The executed source-to-evidence sequence generated **7,015 I-VT fixation intervals**, **9,304 I-VT saccade intervals** and **5,409 fixation-only I-DT intervals**. Their total of **21,728** counts method-specific output intervals, **not 21,728 distinct independently confirmed eye movements**. Source identities and valid-gaze fractions were reconciled across the saved producer tables.

The exploratory [participant-level analysis](eye_tracking_foundations/code/notebooks/05_participant_balanced_statistics.ipynb) produced **16 task–feature summaries** using **500 participant bootstrap resamples** (seed 17), with ten independent people contributing to each task. The [I-VT threshold sensitivity analysis](eye_tracking_foundations/code/notebooks/06_detector_sensitivity_and_vendor_reference.ipynb) compared **20, 30 and 40 dva/s** on the same 40 records; jointly evaluable EyeLink comparison samples increased from **1,015,044** to **1,888,902** and **2,357,954**. Conditional agreement cannot be interpreted as a fixed-denominator accuracy comparison or as evidence for choosing the highest threshold.

The [executed integration notebook](eye_tracking_foundations/code/notebooks/07_integrated_evidence_and_reproducibility.ipynb) reconciled the specified source records, participant statistics, original sample counts and threshold summaries. It did **not** reproduce the full pipeline independently. Extremely high calculated gaze speeds, provisional detector definitions, the limited ten-person cohort and vendor-derived rather than expert-adjudicated reference labels constrain scientific interpretation.

For the full protocol, exact numerical findings and limitations, use the [study scientific report](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md) and [seven-notebook diagram](eye_tracking_foundations/images/study_overview.svg).

## Selected EEG–Eye-Tracking Multimodal Research evidence

The [ZuCo 2.0 study](eeg_eye_tracking_fusion/README.md) includes **18 participants**, **36 original NR/TSR Run-1 recordings** and **630 matched non-overlapping 12-second EEG–gaze windows**. The participant-disjoint partitions contain **11 Train people / 371 windows**, **three Validation people / 99 windows** and **four reserved Test people / 160 windows**. Notebooks **01–06** have produced **44 compatible saved outputs** covering acquisition, temporal pairing, signal features, classical and GRU modelling, and development model selection.

Across **ten predefined candidates** on identical Validation windows, the selected **Gaze Random Forest** achieved pooled **Macro F1 0.6244**, **Balanced Accuracy 0.6231** and **Accuracy 0.6465**. The best tested multimodal candidate, **Gated Fusion GRU**, achieved Macro F1 **0.6107**: fusion did **not** improve over the strongest gaze-only comparator in this Validation cohort. Model scores vary across the three independent Validation participants. The final Test has **not** been scored; independent EEG–EyeLink drift, gaze calibration and cognitive generalisation remain unverified.

For the six-stage workflow, supporting evidence and exact evaluation boundaries, consult the [study overview](eeg_eye_tracking_fusion/images/study_overview.svg), [scientific methods](eeg_eye_tracking_fusion/RESEARCH_METHODS_AND_EVIDENCE.md), [saved model selection](eeg_eye_tracking_fusion/output/06_model_selection.csv) and [executed Notebook 06](eeg_eye_tracking_fusion/code/notebooks/06_robustness_and_final_evaluation.ipynb).

## Planned research directions

**All three established studies have recorded results.** Eye-Tracking Foundations has completed exploratory source-to-evidence analyses; Eye-Movement Modelling has completed its 100-person GazeBase main Test. The selected model for that study was fixed using Validation, so the observed Test must not inform retrospective model selection.

EEG–Eye-Tracking Multimodal Research has completed **Notebooks 01–06 on its expanded 18-person ZuCo 2.0 cohort**, including ten-model Validation selection. Its **four-person, 160-window Test remains reserved**. Any future final Test execution should preserve the current selection and participant roles; independent device-drift checks, gaze-calibration assessment or stimulus-controlled analysis would constitute separately specified scientific extensions.

**Vision-Based Gaze Estimation** and **Applied Neurocognitive Research** remain future research directions without completed experiments. EMTeC is a possible complementary source for a separate approved eye-tracking study; it was **not used** in the current GazeBase or ZuCo 2.0 analyses.

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
│   ├── README.md                       # 18-person ZuCo 2.0 study overview
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
- Use participants, not repeated windows or samples, as independent units when interpreting uncertainty. The current cohorts are 10 people for Foundations, 100 for GazeBase modelling and 18 for ZuCo fusion (three people in its Validation partition).
- Distinguish committed execution output, synthetic software tests, internal source-table reconciliation and independent scientific reproduction.
- Do not extend the completed study's findings to clinical diagnosis, generalised cognitive inference or real-time gaze processing.

Repository code and documentation are covered by the [MIT licence](LICENSE); this does not relicense the original datasets. [GazeBase](https://doi.org/10.6084/m9.figshare.12912257) is reported under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) in its [source publication](https://doi.org/10.1038/s41597-021-00959-y). A **40-file selected GazeBase source subset is tracked** in this repository and requires the original attribution. The EEG–gaze study uses [ZuCo 2.0 (NEMAR nm000279)](https://doi.org/10.82901/nemar.nm000279), whose BIDS metadata report CC BY 4.0; its original EEG/EyeLink recordings are **not** bundled here. Reviewed derived outputs may be committed; large raw/processed files remain excluded and the Git blob policy blocks files larger than **50 MB (50,000,000 bytes)**.
