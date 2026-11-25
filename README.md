# Eye-Tracking & Neurocognitive Intelligence

**Eye-tracking, gaze signal analysis and neurocognitive AI research**

A Python research programme covering **five implemented, scientifically independent studies with recorded experiments**: GazeBase eye-movement signal analysis, GazeBase recording-task modelling, ZuCo 2.0 EEG–eye-tracking fusion, MPIIGaze image-based gaze estimation and OneStop reading-comprehension modelling. **Eye-Tracking Foundations** has seven executed notebooks. **Eye-Movement Modelling** has five completed notebooks on 100 participants; its Validation-selected Logistic Regression classified **80/80 main Test records from 20 people** correctly (Macro F1 **1.000**) within the GazeBase task setting. **EEG–Eye-Tracking Multimodal Research** has six executed notebooks on 18 ZuCo 2.0 participants, with Gaze Random Forest selected on three-person Validation (Macro F1 **0.6244**); its frozen gaze forest achieved **Test Macro F1 0.4997** on four independent people (160 windows). **Vision-Based Gaze Estimation** evaluates 15 held-out MPIIGaze participants using outer LOPO; the Small CNN achieves a participant-macro mean angular error of **7.3048°**. **Reading Comprehension & Neurocognitive Modelling** analyses 9,718 trials from 180 OneStop participants; the Validation-selected candidates achieve Test Macro F1 **0.5560** and **0.5035** in its two protocols. Recording-task classification and response-correctness prediction are not direct measures of general cognition or evidence of external generalisation.

## Research projects

| Project | Current evidence-based status | Start here |
| --- | --- | --- |
| **Eye-Tracking Foundations & Signal Analysis** | **Seven executed notebooks (01–07)** using **10 participants, 40 Round 1 / Session 1 recordings and 2,778,008 source samples**. Source audit, gap-aware preprocessing, provisional I-VT/I-DT events, features, participant statistics, threshold sensitivity and integrated output checks are recorded. Exploratory within-source evidence, not independently validated eye-event accuracy or neurocognitive prediction. | [Study README](eye_tracking_foundations/README.md) · [Scientific methods and evidence](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md) · [Complete workflow diagram](eye_tracking_foundations/images/study_overview.svg) · [How to run](eye_tracking_foundations/README.md#4-running-and-reviewing-the-study) |
| **Eye-Movement Modelling & Advanced AI** | **100-person main study, Notebooks 01–05 complete:** GazeBase Round 1/Session 1, 400 records, 9,600 windows and participant-disjoint **64/16/20** Train/Validation/Test. Eight Validation candidates; **Logistic Regression selected before Test** and achieved main Test **Accuracy 1.000, Macro F1 1.000 (80/80 records from 20 independent people)**. Random Forest also scored 80/80 as a secondary Test comparison. No cross-session, external, cognitive or clinical generalisation demonstrated. | [Study README](eye_movement_modelling/README.md) · [Scientific methods](eye_movement_modelling/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](eye_movement_modelling/images/study_overview.svg) |
| **EEG–Eye-Tracking Multimodal Research** | **18-person ZuCo 2.0 study, Notebooks 01–06 executed with final held-out Test:** 36 NR/TSR runs, 630 paired windows and disjoint **11/3/4** Train/Validation/Test people. Ten candidates produced 990 Validation and 1,600 Test predictions. **Gaze Random Forest was frozen on Validation** (Macro F1 **0.6244**) and achieved **Test Macro F1 0.4997**, Accuracy **0.5250** and Balanced Accuracy **0.5250** on 160 Test windows from four people. The 49 saved outputs include person-level scores, uncertainty and the confusion matrix. Test results did not alter model selection; this is within-ZuCo task evaluation, not external or cognitive validation. | [Study README](eeg_eye_tracking_fusion/README.md) · [Scientific methods](eeg_eye_tracking_fusion/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](eeg_eye_tracking_fusion/images/study_overview.svg) · [Executed Notebook 06](eeg_eye_tracking_fusion/code/notebooks/06_robustness_and_final_evaluation.ipynb) |
| **Vision-Based Gaze Estimation** | **Notebooks 01–05 executed:** MPIIGaze study-defined subset, 7,500 images, 15 participants, 15 outer LOPO folds, 30,000 matched predictions. Participant-macro mean angular errors: Small CNN 7.3048°, HOG + Ridge 7.5596°, ResNet-18 8.2304°, mean direction 9.1601°. The HOG-minus-CNN paired interval includes zero; internal exploratory evaluation, not external validation. | [Study README](vision_based_gaze_estimation/README.md) · [Research methods](vision_based_gaze_estimation/RESEARCH_METHODS_AND_EVIDENCE.md) · [Final workflow](vision_based_gaze_estimation/images/05_cross_person_evaluation_and_robustness.svg) · [Notebook sequence](vision_based_gaze_estimation/code/notebooks/) |
| **Reading Comprehension & Neurocognitive Modelling** | **Five executed notebooks, reconciled internal Test:** OneStop ordinary reading, 9,718 trials from 180 participants and 30 articles. Nineteen candidates per protocol produced 160,018 saved scores. Validation-selected TF-IDF fusion (A) and TF-IDF text (B) achieve Test Macro F1 0.5560 and 0.5035 on 36 and 60 held-out people respectively. Added gaze is model-dependent; external generalisation is not established. | [Study README](reading_comprehension_modelling/README.md) · [Scientific methods](reading_comprehension_modelling/RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow](reading_comprehension_modelling/images/study_overview.svg) |

**Implemented study boundaries:** Eye-Tracking Foundations and Eye-Movement Modelling both use GazeBase Round 1 / Session 1 but ask different questions and retain separate outputs. EEG–Eye-Tracking Multimodal Research uses **ZuCo 2.0** with its own recordings, participant split, model candidates and evidence. The vision-based study has five executed source, image-preparation, classical/neural evaluation and final-analysis notebooks; the OneStop reading-comprehension study has completed its five-stage cohort, feature, modelling and internal held-out evaluation programme with reconciled saved scores.

## Selected Eye-Tracking Foundations evidence

The [GazeBase analysis](eye_tracking_foundations/README.md) retains **2,778,008 sample rows across 40 task recordings** from **10 participants** in four tasks: FXS, HSS, RAN and TEX. The executed source-to-evidence sequence generated **7,015 I-VT fixation intervals**, **9,304 I-VT saccade intervals** and **5,409 fixation-only I-DT intervals**. Their total of **21,728** counts method-specific output intervals, **not 21,728 distinct independently confirmed eye movements**. Source identities and valid-gaze fractions were reconciled across the saved producer tables.

The exploratory [participant-level analysis](eye_tracking_foundations/code/notebooks/05_participant_balanced_statistics.ipynb) produced **16 task–feature summaries** using **500 participant bootstrap resamples** (seed 17), with ten independent people contributing to each task. The [I-VT threshold sensitivity analysis](eye_tracking_foundations/code/notebooks/06_detector_sensitivity_and_vendor_reference.ipynb) compared **20, 30 and 40 dva/s** on the same 40 records; jointly evaluable EyeLink comparison samples increased from **1,015,044** to **1,888,902** and **2,357,954**. Conditional agreement cannot be interpreted as a fixed-denominator accuracy comparison or as evidence for choosing the highest threshold.

The [executed integration notebook](eye_tracking_foundations/code/notebooks/07_integrated_evidence_and_reproducibility.ipynb) reconciled the specified source records, participant statistics, original sample counts and threshold summaries. It did **not** reproduce the full pipeline independently. Extremely high calculated gaze speeds, provisional detector definitions, the limited ten-person cohort and vendor-derived rather than expert-adjudicated reference labels constrain scientific interpretation.

For the full protocol, exact numerical findings and limitations, use the [study scientific report](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md) and [seven-notebook diagram](eye_tracking_foundations/images/study_overview.svg).

## Selected EEG–Eye-Tracking Multimodal Research evidence

The [ZuCo 2.0 study](eeg_eye_tracking_fusion/README.md) includes **18 participants**, **36 original NR/TSR Run-1 recordings** and **630 matched non-overlapping 12-second EEG–gaze windows**. The participant-disjoint partitions contain **11 Train people / 371 windows**, **three Validation people / 99 windows** and **four reserved Test people / 160 windows**. Notebooks **01–06** have produced **49 compatible saved outputs** covering acquisition, alignment, features, classical and GRU modelling, frozen development selection and final Test evidence.

Across **ten predefined candidates** on identical Validation windows, the selected **Gaze Random Forest** achieved pooled **Macro F1 0.6244**, **Balanced Accuracy 0.6231** and **Accuracy 0.6465**. The best tested multimodal candidate, **Gated Fusion GRU**, achieved Macro F1 **0.6107**: fusion did **not** improve over the strongest gaze-only comparator in this Validation cohort. Model scores vary across the three independent Validation participants. The **frozen Gaze Random Forest** achieved **Test Macro F1 0.4997** and **Accuracy 0.5250** across 160 windows and four independent people, lower than its Validation performance. Gated Fusion GRU (**Test Macro F1 0.5157**) and Late Fusion (**0.5204**) are secondary Test comparisons, not new selected winners. The **three-person Validation** and **four-person Test** constrain independent-person inference; NR/TSR instructions or stimuli may differ, and independent EEG–EyeLink clock-drift and gaze-calibration checks were not conducted. These are dataset, measurement and evaluation-scope qualifications, not missing Test executions; their causes are explained in the [study methods](eeg_eye_tracking_fusion/RESEARCH_METHODS_AND_EVIDENCE.md).

For the six-stage workflow and final results, consult the [study overview](eeg_eye_tracking_fusion/images/study_overview.svg), [scientific methods](eeg_eye_tracking_fusion/RESEARCH_METHODS_AND_EVIDENCE.md), [frozen model decision](eeg_eye_tracking_fusion/output/06_model_selection.csv), [final Test metrics](eeg_eye_tracking_fusion/output/06_test_metrics.csv) and [executed Notebook 06](eeg_eye_tracking_fusion/code/notebooks/06_robustness_and_final_evaluation.ipynb).

## Planned research directions

**All five established studies have recorded results.** Eye-Tracking Foundations has completed exploratory source-to-evidence analyses; Eye-Movement Modelling has completed its 100-person GazeBase main Test. The selected model for that study was fixed using Validation, so the observed Test must not inform retrospective model selection.

EEG–Eye-Tracking Multimodal Research has completed **Notebooks 01–06 and the four-person, 160-window held-out Test**, after freezing its development winner on Validation. The primary Test Macro F1 is **0.4997**, with substantial participant-level variation. Independent device-drift checks, gaze-calibration assessment, stimulus-controlled analysis and external generalisation remain separate scientific extensions.

**Vision-Based Gaze Estimation** has completed all five notebook executions: source audit, image preparation, full classical/neural outer LOPO and matched participant-level analysis, with five scientific figures. **Reading Comprehension & Neurocognitive Modelling** implements the first applied neurocognitive study using OneStop; Notebooks 01–05 record the cohort, features, exploratory associations, fixed-candidate modelling and reconciled final Test metrics, conditional participant uncertainty and calibration diagnostics. EMTeC is a possible complementary source for a separate approved eye-tracking study; it was **not used** in the current GazeBase or ZuCo 2.0 analyses.

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
│   ├── code/notebooks/                 # Notebooks 01–06, including held-out Test
│   ├── code/src/                       # BIDS retrieval, alignment, features, models
│   ├── images/                         # Seven five-band scientific diagrams
│   ├── data/README.md                  # Original ZuCo data source and provenance
│   └── output/                         # 49 current Stage-01–06 files, including Test
├── vision_based_gaze_estimation/
│   ├── README.md                       # Executed MPIIGaze study overview
│   ├── RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── requirements.txt                # Imaging study dependencies
│   ├── code/notebooks/                 # Five executed research notebooks
│   ├── code/src/                       # Source, geometry, folds and models
│   ├── images/                         # Six editable scientific diagrams
│   └── output/                         # 31 current derived outputs
├── reading_comprehension_modelling/
│   ├── README.md
│   ├── RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── requirements.txt
│   ├── code/notebooks/                 # Five executed research notebooks
│   ├── code/src/                       # OneStop source, features, models and evaluation
│   ├── code/tests/                     # Synthetic study contracts
│   ├── images/                         # Six methodological SVGs
│   └── output/                         # 31 current derived outputs
├── tests/                              # Synthetic scientific software contracts
├── scripts/                            # Git file-size guard
└── .github/workflows/                  # Synthetic tests, size policy and ZuCo final Test
```

Vision-Based Gaze Estimation has its own [study directory](vision_based_gaze_estimation/) with five executed notebooks and **complete internal outer-LOPO and paired analysis evidence**. The [OneStop study](reading_comprehension_modelling/) has five implemented notebooks, with all five notebooks executed and 31 reconciled saved outputs, including the [final Test metrics](reading_comprehension_modelling/output/05_test_metrics.csv) and [executed final notebook](reading_comprehension_modelling/code/notebooks/05_generalisation_and_final_evaluation.ipynb). Each project's original data, code and outputs remain scientifically independent.

## Getting started

For Eye-Tracking Foundations, start with the [study README](eye_tracking_foundations/README.md) and [scientific methods](eye_tracking_foundations/RESEARCH_METHODS_AND_EVIDENCE.md). For the completed Eye-Movement Modelling study, use its [README](eye_movement_modelling/README.md), [scientific evidence](eye_movement_modelling/RESEARCH_METHODS_AND_EVIDENCE.md) and [Notebook 05](eye_movement_modelling/code/notebooks/05_robustness_and_final_evaluation.ipynb). For EEG–eye-tracking research, start with the [study README](eeg_eye_tracking_fusion/README.md), [scientific evidence](eeg_eye_tracking_fusion/RESEARCH_METHODS_AND_EVIDENCE.md), [executed Notebook 06](eeg_eye_tracking_fusion/code/notebooks/06_robustness_and_final_evaluation.ipynb) and [final Test metrics](eeg_eye_tracking_fusion/output/06_test_metrics.csv). All six notebooks and the frozen held-out Test are published; the saved pipelines require **scikit-learn 1.6.1** for compatible inference. The study README explains how to inspect or rerun Notebook 06 without rebuilding the earlier stages.

For image-based research, start with the [MPIIGaze study README](vision_based_gaze_estimation/README.md) and [scientific evidence](vision_based_gaze_estimation/RESEARCH_METHODS_AND_EVIDENCE.md). For reading-comprehension modelling, start with the [OneStop study README](reading_comprehension_modelling/README.md), [scientific evidence](reading_comprehension_modelling/RESEARCH_METHODS_AND_EVIDENCE.md), [final evaluation notebook](reading_comprehension_modelling/code/notebooks/05_generalisation_and_final_evaluation.ipynb) and [Test metrics](reading_comprehension_modelling/output/05_test_metrics.csv). The OneStop final evaluation runs from the saved CSVs without source acquisition or model fitting.

For original GazeBase acquisition, existing source paths and attribution, consult the [data guide](data/README.md). For VS Code, Colab setup and named output ZIP hand-offs, use [GETTING_STARTED.md](GETTING_STARTED.md). Reviewed study outputs can be explicitly tracked in Git; the repository's independent file-size policy rejects unapproved blobs larger than **50,000,000 bytes**. Large raw/processed data remain locally excluded. ZIP archives made under Colab `/content/` are not automatically published. Re-execution requires the compatible original data and notebook-specific current inputs; a documentation change alone does not require rerunning the experiment.

## Scientific scope, evidence and licensing

- Preserve source recording, participant, task, session and sample identity through analysis.
- Distinguish measured gaze, algorithm-derived eye events, vendor parser annotations and independently observed cognitive outcomes.
- Use participants, not repeated windows or samples, as independent units when interpreting uncertainty. The study cohorts comprise 10 people for Foundations, 100 for GazeBase modelling, 18 for ZuCo fusion, 15 for MPIIGaze and 180 for OneStop; repeated observations do not increase the independent participant count.
- Distinguish committed execution output, synthetic software tests, internal source-table reconciliation and independent scientific reproduction.
- Do not extend the completed study's findings to clinical diagnosis, generalised cognitive inference or real-time gaze processing.

Repository code and documentation are covered by the [MIT licence](LICENSE); this does not relicense the original datasets. [GazeBase](https://doi.org/10.6084/m9.figshare.12912257) is reported under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) in its [source publication](https://doi.org/10.1038/s41597-021-00959-y). A **40-file selected GazeBase source subset is tracked** in this repository and requires the original attribution. The EEG–gaze study uses [ZuCo 2.0 (NEMAR nm000279)](https://doi.org/10.82901/nemar.nm000279), whose BIDS metadata report CC BY 4.0; its original EEG/EyeLink recordings are **not** bundled here. The vision-based study uses [MPIIGaze](https://doi.org/10.18419/darus-3230), subject to [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The reading study uses the [OneStop ordinary-reading component](https://osf.io/zn9sq/); its eye-tracking data are provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), while the underlying OneStopQA text and auxiliary annotations are subject to [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). See each study's scientific document for source attribution and boundaries. Reviewed derived outputs may be committed; large raw/processed files remain excluded and the Git blob policy blocks files larger than **50 MB (50,000,000 bytes)**.
