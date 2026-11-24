# Vision-Based Gaze Estimation: Research Methods and Evidence

This document describes the recorded Notebook 01 source audit and the implemented, original-data-unexecuted Notebooks 02–04 and specifies the remaining evaluation boundaries for the independent vision-based study. See the [study README](README.md). **No numerical MPIIGaze gaze-estimation results have been produced or independently verified here.**

## 1. Research Objective and Scope

**Main question:** Can image-based gaze estimation predict camera/normalised-frame 3D gaze direction with low angular error for participants excluded from fitting and model selection?

**Implemented candidate families (not executed on the original data):** Train-derived mean-gaze reference; HOG descriptors with Train-scaled Ridge regression; Small CNN; and ResNet-18, optionally initialised with ImageNet weights. A Vision Transformer is outside the current experiment and requires a separate scientific and compute justification.

**Excluded from current scope:** eye-event segmentation, EEG fusion, comprehension prediction, screen-calibrated point-of-gaze claims, and personalised calibration benchmarks mixed with calibration-free unseen-person experiments.

## 2. Data Sources, Cohort and Target Definitions

**Candidate dataset:** official [MPIIGaze](https://www.collaborative-ai.org/research/datasets/MPIIGaze/) / [DaRUS archived distribution](https://doi.org/10.18419/darus-3230). Published source descriptions report 15 participants and 213,659 collected full-face images, with Data, Evaluation Subset and Annotation Subset distinguished. These are **source-reported counts, not verified current study records**.

The archive is offered under **CC BY-NC-SA 4.0**. Raw source images and annotations must not be committed to Git by default. The exact release and available files, MATLAB format, normalised/original directory conventions, sample IDs, participant/day IDs, eye-side mapping, label array semantics and archived evaluation lists must be inspected in Notebook 01.

**Known source ambiguity:** official documentation gives differing annotated-subset quantities; neither figure is assumed for the selected usable cohort until a file-level audit resolves definitions.

**Analytical unit:** source-labelled image/eye sample, nested within participant and potentially day. The independent evaluation unit is the participant. The initial implemented loader selects **left-eye normalised grayscale patches** and interprets the source's candidate 2-angle (radian) or 3-vector gaze format as **3D unit vectors in the normalised coordinate frame**; the specific MAT field and axis convention still require original-source inspection. Screen-space targets and 2D points are never silently substituted.

## 3. Data Preparation and Representations

**Notebook 01 (source audit executed; revised selection awaiting rerun):** automatically reuse the extracted remote Colab source or cached archive; otherwise acquire the publisher archive or official DaRUS file via programmatic access, validate and safely extract ZIP/TAR entries, inventory observed source files, identify participant/day support, inspect representative MATLAB variable headers and publisher evaluation/annotation-list metadata, and save a proposed evaluation protocol. The source metadata reader is [mpiigaze_io.py](code/src/mpiigaze_io.py). These operations do **not** verify image-level gaze targets or constitute a completed evaluation.

**Notebook 02 (implemented, unexecuted on MPIIGaze):** the [source reader](code/src/mpiigaze_data.py) expects normalised MATLAB `data.<eye>.image` and `data.<eye>.gaze` fields, checks image–label counts, rejects unsupported MAT representations and preserves participant/file/row/eye identity. Its preliminary reproducible sample uses **up to 500 left-eye images per person** by deterministic within-person reservoir sampling, not the publisher's official Evaluation Subset. Raw data are automatically recovered in the remote runtime if absent; normalised MAT file paths and sizes must match the current Stage-01 inventory. The image cache remains in ignored `data/processed/mpiigaze/`; the [fold utility](code/src/gaze_evaluation.py) creates participant-based outer LOPO and two-person inner Validation. These source-specific assumptions must be verified against actual MAT files before accepting the resulting sample/target contract.

The initial [angular-error helper](code/src/gaze_geometry.py) implements a bounded, norm-checked angle between two real 3D vectors and has **synthetic numerical tests only**. It does not assert that MPIIGaze labels are already interpreted correctly.

Pixel scaling to [0,1] is a fixed per-image operation; the HOG/Ridge feature standardiser is fitted on **inner-training images only**. The implemented neural pipeline currently uses no stochastic augmentation, so no augmentation claim is made. Original image pixels, 3D direction vectors and 2D screen coordinates remain distinct measurement types. Actual source geometry needs inspection before those fixed transformations are considered scientifically validated.

## 4. Experimental Design and Evaluation

**Implemented proposed design:** outer participant-held-out Leave-One-Person-Out (LOPO), generated from the *actual parsed eligible participant list*, with **two different inner-Validation people** per outer fold (seed 17); the remainder of each fold's people provide inner Training. The source-reported 15 participants are not imposed as an eligible-count assertion. The 500-per-person sampling cap yields a bounded study subset, **not** the published official MPIIGaze evaluation protocol. All generated image identities and Fold roles are retained in producer-owned tables.

Both model notebooks consume the **same Notebook 02 sample index, cached grayscale images, gaze vectors and outer/inner roles**. If a fresh Colab runtime lacks the image cache, exact recorded MAT rows are restored from the official source, checking sample identifiers and gaze targets without reselecting images or changing fold roles. Classical Ridge alpha selection and neural early stopping use **inner Validation only**; outer Test must not guide model choice or methodological edits. **Both Notebooks 03 and 04 have outer-Test scoring switched off by default.** Stage 03 computes inner-Validation results for all available outer-fold configurations; Stage 04 initially trains only **two outer-fold configurations** as a compute-development check. A separately enabled complete outer LOPO run is required before publication of generalisation scores. Notebook 05 remains a starter and must not rank or revise candidates retrospectively using Test scores.

**Primary metric:** angular error in degrees between normalised predicted and true 3D gaze vectors; report mean, median, participant distribution and support counts. Grouped uncertainty recognises participants, not individual images, as independent units.

**Compute boundary:** a full participant-level neural LOPO run can require repeated deep-model fitting and must be reviewed after profiling original-source image dimensions, eligible people and Colab resources. The two-fold development configuration is explicitly **not** a final evaluation. Pretrained ResNet-18 weights may be downloaded independently from torchvision/ImageNet, and their use must be reported.

## 5. Methods, Models and Configurations

**Classical stage (Notebook 03; implemented, not run on source):** each fold fits a **mean training gaze direction** and a **HOG + StandardScaler + Ridge** comparator. Ridge candidates `alpha ∈ {1, 10, 100}` are compared on that fold's inner Validation people. Continuous Ridge predictions are normalised to nonzero unit vectors before 3D angular-error calculation. The default run saves inner metrics and a plot; an explicit full-outer mode also writes matched outer predictions and per-person metrics.

**Deep stage (Notebook 04; implemented, not run on source):** a three-block compact grayscale CNN and ResNet-18 with a grayscale first convolution and optional ImageNet pretraining. Both produce 3D vectors, optimised using **mean cosine-direction loss** with AdamW. The current development configuration uses a maximum of eight epochs, three-epoch early-stopping patience and batch size 128. The checkpoint with the lowest inner-Validation angular error is retained for each model/fold, and parameter counts and epoch histories are saved. PyTorch's CUDA/CPU device is reported. A complete outer-Test pass requires explicit activation and all outer folds; its performance is not used for hyperparameter or model choice.

**Stage-specific utilities:** [`mpiigaze_io.py`](code/src/mpiigaze_io.py) inventories/downloads/extracts and inspects source MAT headers; [`mpiigaze_data.py`](code/src/mpiigaze_data.py) parses selected normalised eye data; [`gaze_geometry.py`](code/src/gaze_geometry.py) implements unit-vector angular error; [`gaze_evaluation.py`](code/src/gaze_evaluation.py) validates Fold and sample identity; [`gaze_models.py`](code/src/gaze_models.py) implements the classical and deep model families. Their original-data compatibility remains to be verified.

## 6. Results and Comparative Analysis

**Recorded source-audit evidence.** Notebook 01 recorded 430,180 archive paths, 15 participant IDs and 30 variable records across 15 representative MATLAB headers. The evaluation/annotation inventory contains 45 participant-list paths and one ReadMe after excluding 46 macOS companions. Revised participant/day MAT counts and the chart require a rerun; the saved candidate counts and existing PNG describe the original audit. Notebooks 02–04 remain unexecuted on original data. Synthetic tests do not establish actual source image/label compatibility or model performance. No real 3D gaze prediction scores, trained original-data checkpoints, outer-fold comparisons or generalisation claims are available.

Once executed, report the observed eligible cohort, each model's per-person predictions, mean/median angular error in degrees, comparison to the strongest valid baseline, failure cases and uncertainty. Do not copy MPIIGaze published benchmark numbers into this study's result columns.

## 7. Robustness, Uncertainty and Additional Analyses

Planned checks include eye-side orientation, image quality/missingness, head pose and illumination **where directly supported by the inspected source**, per-participant variability, configuration stability and inference cost. A source field is not assumed to exist simply because a publication describes a related dataset.

Potential confounding includes repeated days/images for one person, unverified camera normalisation, annotation differences, calibration requirements and invalid 3D label conventions. Neither a synthetic geometry test nor a successful source file inventory validates these scientific assumptions.

## 8. Current Outputs and Reproducibility

The repository contains **one recorded source-audit run (01) and three implemented, original-data-unexecuted notebooks (02–04)** and one Notebook-05 starter; a source adapter, data reader, gaze geometry, participant evaluation and model utility; four editable five-band scientific figures ([01](images/01_mpiigaze_source_audit_and_protocol.svg), [02](images/02_eye_images_geometry_and_quality.svg), [03](images/03_classical_gaze_estimation_baselines.svg), [04](images/04_cnn_and_transfer_learning.svg)); and the two study documents. Synthetic tests verify safe source inventory/extraction, not original-data validity. **No original MPIIGaze data or model predictions are committed.** The source-audit notebook creates stage-owned metadata and a figure only after it has opened original source files.

Intended stage-specific files, all **planned until genuinely produced**:

| Producer | Intended files and consumers |
| --- | --- |
| 01 | `01_file_inventory.csv`, `01_participant_support.csv`, `01_mat_schema.csv`, `01_evaluation_list_inventory.csv`, `01_protocol.csv`, `01_source_inventory.png` (produced when the original archive is available); source verification for Stage 02 |
| 02 | `02_sample_index.csv`, `02_fold_assignments.csv`, `02_source_sample_counts.csv`, `02_cache_location.csv`, `02_image_quality.png`; local ignored `02_normalised_eye_cache.npz` used by 03–04 |
| 03 | Development: `03_inner_validation_metrics.csv`, `03_run_configuration.csv`, `03_baseline_comparison.png`. Full outer LOPO, when explicitly enabled: `03_baseline_outer_predictions.csv`, `03_baseline_fold_metrics.csv` |
| 04 | Development: `04_training_history.csv`, `04_inner_validation_metrics.csv`, `04_model_checkpoints.csv`, `04_run_configuration.csv`, `04_validation_curves.png` and ignored checkpoint weights. Full outer LOPO, when enabled: `04_neural_outer_predictions.csv`, `04_neural_outer_metrics.csv` |
| 05 | `05_model_metrics.csv`, `05_participant_metrics.csv`, error figure for documentation |

Use VS Code connected to the remote Colab runtime, the established repository setup and the study-level `output/` directory. Raw archives, extracted source, image cache and checkpoints remain in ignored runtime directories. Only stage-owned analysis outputs are packaged for manual download from Colab Contents; the user extracts these into the study's local `output/` directory and synchronises the reviewed files through Git. Rerunning an implemented stage overwrites its own current named outputs. Do not silently retrain models when a reporting stage opens missing files.

## 9. Discussion, Limitations and Future Work

The study extends the programme from already-recorded gaze/EEG signals to **camera-image gaze prediction**, with a separate source, 3D normalised gaze frame and angular-error metric. Notebook 01 has recorded original-data audit evidence; Notebooks 02–04 contain implementation but original-data execution is unverified, Notebook 05 is not yet implemented, and no gaze generalisation outcomes have been established.

The principal early limitations are **unverified original MAT source/label conventions**, the unresolved annotated-subset count definitions, participant-level image dependence, a preliminary 500-per-person (not official benchmark) subset, unknown compute demand of full neural LOPO and source licensing. The immediate operation is to **rerun the revised Notebook 01 using the cached source**, then run Notebook 02 and verify its exact image–target mapping, normalised coordinate system and Fold roles before accepting the classical or neural outputs. Software tests cannot replace those checks.

**Reference:** Zhang, X., Sugano, Y., Fritz, M. and Bulling, A., MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation, IEEE TPAMI. [DOI](https://doi.org/10.1109/TPAMI.2017.2778103).
