# Vision-Based Gaze Estimation: Research Methods and Evidence

This document defines the planned scientific methods and evidence requirements for the **newly initialised** vision-based study. It is paired with the concise [study README](README.md). **No MPIIGaze original-data modelling results have been produced or reproduced in this repository.**

## 1. Research Objective and Scope

**Main question:** Can image-based gaze estimation predict camera/normalised-frame 3D gaze direction with low angular error for participants excluded from fitting and model selection?

**Proposed comparisons:** train-derived mean-gaze reference; classical image descriptor + regression; compact CNN; optionally a lightweight deep feature model. A Transformer is not part of the core baseline and requires a separate compute and scientific rationale.

**Excluded from current scope:** eye-event segmentation, EEG fusion, comprehension prediction, screen-calibrated point-of-gaze claims, and personalised calibration benchmarks mixed with calibration-free unseen-person experiments.

## 2. Data Sources, Cohort and Target Definitions

**Candidate dataset:** official [MPIIGaze](https://www.collaborative-ai.org/research/datasets/MPIIGaze/) / [DaRUS archived distribution](https://doi.org/10.18419/darus-3230). Published source descriptions report 15 participants and 213,659 collected full-face images, with Data, Evaluation Subset and Annotation Subset distinguished. These are **source-reported counts, not verified current study records**.

The archive is offered under **CC BY-NC-SA 4.0**. Raw source images and annotations must not be committed to Git by default. The exact release and available files, MATLAB format, normalised/original directory conventions, sample IDs, participant/day IDs, eye-side mapping, label array semantics and archived evaluation lists must be inspected in Notebook 01.

**Known source ambiguity:** official documentation gives differing annotated-subset quantities; neither figure is assumed for the selected usable cohort until a file-level audit resolves definitions.

**Analytical unit:** source-labelled image/eye sample, nested within participant and potentially day. The independent evaluation unit is the participant. The model target will be a labelled **3D gaze vector** in one verified coordinate frame; screen-space targets and 2D points are not silently substituted.

## 3. Data Preparation and Representations

**Notebook 01:** inventory official source files and their provenance, with relative paths, types and sizes; inspect actual MATLAB/image schema before implementing a reusable parser.

**Notebook 02:** link image records to valid 3D labels, preserve participant/day/image/eye identifiers, audit invalid images and vectors, clarify geometry orientation and left/right transformations, and store compact reproducible sample and fold tables. No arbitrary hard-coded image or label dimensions are permitted before source inspection.

The initial [angular-error helper](code/src/gaze_geometry.py) implements a bounded, norm-checked angle between two real 3D vectors and has **synthetic numerical tests only**. It does not assert that MPIIGaze labels are already interpreted correctly.

Image augmentation and any data-derived normalisation must be fitted/applied **within training folds only**. Original image pixels, 3D direction vectors and 2D screen coordinates remain distinct measurement types.

## 4. Experimental Design and Evaluation

**Proposed primary design:** outer participant-held-out Leave-One-Person-Out (LOPO) evaluation using the actual eligible participant list, and separate inner participant-disjoint validation for each outer fold. The source-reported 15 people are not assumed to all survive data eligibility filtering.

Each model must use the same outer test person and compatible source sample identities for a fair baseline/deep comparison. Hyperparameters, checkpoints, augmentation choices and candidate selection are derived **without outer-test labels**. After fitting, each outer fold produces one scored prediction per eligible test sample. Notebook 05 aggregates these already saved predictions and **does not select a new winner because of outer-test performance**.

**Primary metric:** angular error in degrees between normalised predicted and true 3D gaze vectors; report mean, median, participant distribution and support counts. Grouped uncertainty recognises participants, not individual images, as independent units.

**Open design decision:** a full 15-fold neural run is computationally expensive and requires profiling before scheduling. A source-verified small pilot may exercise the pipeline, but must remain distinct from the planned final LOPO protocol.

## 5. Methods, Models and Configurations

**Classical stage (Notebook 03):** within-fold mean-direction baseline and a regularised regression model using HOG or comparable defined image features. Input geometry, feature standardisation and hyperparameter search are Train/inner-Validation only.

**Deep stage (Notebook 04):** compact CNN and optionally ResNet-18 or a comparable light backbone; train on the same image/label representation and outer folds. Report preprocessing, loss function, training schedule, checkpoint selection, number of parameters and inference cost. Architecture choices and training configurations are **proposed, not implemented**.

**Shared stage-specific utility:** `gaze_geometry.py` supplies norm-checked angular error. Dataset-reader and deep-network source modules will be added when actual source files and training designs justify their interfaces, avoiding placeholder modules.

## 6. Results and Comparative Analysis

**No experimental results yet.** This new study contains its planned notebook sequence and a small executable file-inventory/geometry-contract foundation. There are no original-image angular-error values, trained checkpoints, outer-fold prediction tables, rankings or conclusions about any model's advantage.

Once executed, report the observed eligible cohort, each model's per-person predictions, mean/median angular error in degrees, comparison to the strongest valid baseline, failure cases and uncertainty. Do not copy MPIIGaze published benchmark numbers into this study's result columns.

## 7. Robustness, Uncertainty and Additional Analyses

Planned checks include eye-side orientation, image quality/missingness, head pose and illumination **where directly supported by the inspected source**, per-participant variability, configuration stability and inference cost. A source field is not assumed to exist simply because a publication describes a related dataset.

Potential confounding includes repeated days/images for one person, unverified camera normalisation, annotation differences, calibration requirements and invalid 3D label conventions. Neither a synthetic geometry test nor a successful source file inventory validates these scientific assumptions.

## 8. Current Outputs and Reproducibility

The repository now contains five numbered notebook starters, a study-specific angular-error utility, explicit dependency guidance and the two study documents. **No original data or model predictions are committed**. Notebook 01 can create `01_file_inventory.csv` from locally provided source files.

Intended stage-specific files, all **planned until genuinely produced**:

| Producer | Intended files and consumers |
| --- | --- |
| 01 | `01_file_inventory.csv`, source schema notes for Stage 02 |
| 02 | `02_sample_index.csv`, `02_fold_assignments.csv`, image quality plot for 03–05 |
| 03 | `03_baseline_outer_predictions.csv`, baseline fold metrics for 05 |
| 04 | `04_neural_outer_predictions.csv`, training history and fitted weights for 05 |
| 05 | `05_model_metrics.csv`, `05_participant_metrics.csv`, error figure for documentation |

Use the established Colab repository setup and the study-level `output/` directory. Rerunning an implemented stage overwrites its own current named outputs. Do not silently retrain models when a reporting stage opens missing files.

## 9. Discussion, Limitations and Future Work

The planned study extends this programme from already-recorded gaze/EEG signals to **camera-image gaze prediction**, with its own 3D target frame and metrics. The present status is **architecture initialised**, not implemented modelling or evaluated generalisation.

The principal early limitations are uninspected source file conventions, unresolved annotated-subset count definitions, participant-level sample dependence, unknown compute demand of neural LOPO, and dataset-specific licensing. The immediate development step is **Notebook 01**: acquire authorised source files locally, inspect actual schema and confirm the sample/target/fold design. This evidence must precede claims about cohort size or model accuracy.

**Reference:** Zhang, X., Sugano, Y., Fritz, M. and Bulling, A., MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation, IEEE TPAMI. [DOI](https://doi.org/10.1109/TPAMI.2017.2778103).
