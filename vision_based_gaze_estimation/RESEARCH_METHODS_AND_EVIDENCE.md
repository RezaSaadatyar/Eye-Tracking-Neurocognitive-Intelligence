# Vision-Based Gaze Estimation: Research Methods and Evidence

This document describes the executed source audit, image preparation and classical/neural development (Notebooks 01–04), and specifies the remaining evaluation boundaries for the independent vision-based study. See the [study README](README.md). **Classical and neural full outer-LOPO predictions and participant metrics are recorded; the final paired and robustness analyses in Notebook 05 remain pending.**

## 1. Research Objective and Scope

**Main question:** Can image-based gaze estimation predict camera/normalised-frame 3D gaze direction with low angular error for participants excluded from fitting and model selection?

**Candidate families (classical and neural development executed):** Train-derived mean-gaze reference; HOG descriptors with Train-scaled Ridge regression; Small CNN; and ResNet-18, optionally initialised with ImageNet weights. A Vision Transformer is outside the current experiment and requires a separate scientific and compute justification.

**Excluded from current scope:** eye-event segmentation, EEG fusion, comprehension prediction, screen-calibrated point-of-gaze claims, and personalised calibration benchmarks mixed with calibration-free unseen-person experiments.

## 2. Data Sources, Cohort and Target Definitions

**Dataset:** official [MPIIGaze](https://www.collaborative-ai.org/research/datasets/MPIIGaze/) / [DaRUS archived distribution](https://doi.org/10.18419/darus-3230). Published source descriptions report 15 participants and 213,659 collected full-face images, with Data, Evaluation Subset and Annotation Subset distinguished. The published image count is author-reported; the current source audit independently records 15 participant IDs.

The archive is offered under **CC BY-NC-SA 4.0**. Raw source images and annotations must not be committed to Git by default. Notebook 01 inspects archive structure, participant/day paths, representative MATLAB headers and evaluation/annotation-list metadata. Sample identities, eye-side mapping and label array semantics are examined in Notebook 02.

**Known source ambiguity:** official documentation gives differing annotated-subset quantities; neither figure is assumed for the selected usable cohort until a file-level audit resolves definitions.

**Analytical unit:** source-labelled image/eye sample, nested within participant and potentially day. The independent evaluation unit is the participant. The initial implemented loader selects **left-eye normalised grayscale patches** and interprets the source's candidate 2-angle (radian) or 3-vector gaze format as **3D unit vectors in the normalised coordinate frame**; the specific MAT field and axis convention still require original-source inspection. Screen-space targets and 2D points are never silently substituted.

## 3. Data Preparation and Representations

**Notebook 01 (source audit executed):** automatically reuse the extracted remote Colab source or cached archive; otherwise acquire the publisher archive or official DaRUS file via programmatic access, validate and safely extract ZIP/TAR entries, inventory observed source files, identify participant/day support, inspect representative MATLAB variable headers and publisher evaluation/annotation-list metadata, and save a proposed evaluation protocol. The source metadata reader is [mpiigaze_io.py](code/src/mpiigaze_io.py). These operations do **not** verify image-level gaze targets or constitute a completed evaluation.

**Notebook 02 (executed on MPIIGaze):** the [source reader](code/src/mpiigaze_data.py) expects normalised MATLAB `data.<eye>.image` and `data.<eye>.gaze` fields, checks image–label counts, rejects unsupported MAT representations and preserves participant/file/row/eye identity. Its preliminary reproducible sample uses **up to 500 left-eye images per person** by deterministic within-person reservoir sampling, not the publisher's official Evaluation Subset. Raw data are automatically recovered in the remote runtime if absent; normalised MAT file paths and sizes must match the current Stage-01 inventory. The image cache remains in ignored `data/processed/mpiigaze/`; the [fold utility](code/src/gaze_evaluation.py) creates participant-based outer LOPO and two-person inner Validation. The executed decoder reconciled all 521 MAT paths and sizes, processed 213,658 left-eye rows and retained 500 images per participant. Row-count checks and normalised target vectors support the computational input contract; independent confirmation of physical gaze-axis conventions remains outstanding.

The initial [angular-error helper](code/src/gaze_geometry.py) implements a bounded, norm-checked angle between two real 3D vectors and has **synthetic numerical tests only**. It does not assert that MPIIGaze labels are already interpreted correctly.

Pixel scaling to [0,1] is a fixed per-image operation; the HOG/Ridge feature standardiser is fitted on **inner-training images only**. The implemented neural pipeline currently uses no stochastic augmentation, so no augmentation claim is made. Original image pixels, 3D direction vectors and 2D screen coordinates remain distinct measurement types. Actual source geometry needs inspection before those fixed transformations are considered scientifically validated.

## 4. Experimental Design and Evaluation

**Executed participant assignment design:** outer participant-held-out Leave-One-Person-Out (LOPO), generated from the *actual parsed eligible participant list*, with **two different inner-Validation people** per outer fold (seed 17); the remainder of each fold's people provide inner Training. The source-reported 15 participants are not imposed as an eligible-count assertion. The 500-per-person sampling cap yields a bounded study subset, **not** the published official MPIIGaze evaluation protocol. All generated image identities and Fold roles are retained in producer-owned tables.

Both model notebooks consume the **same Notebook 02 sample index, cached grayscale images, gaze vectors and outer/inner roles**. If a fresh Colab runtime lacks the image cache, exact recorded MAT rows are restored from the official source, checking sample identifiers and gaze targets without reselecting images or changing fold roles. Classical Ridge alpha selection and neural early stopping use **inner Validation only**; outer Test must not guide model choice or methodological edits. **Notebooks 03 and 04 have completed full outer LOPO using the same final protocol.** Stage 03 computes inner-Validation results for all available outer-fold configurations; Stage 04's recorded development run used two outer-fold configurations; the final configuration covers all available outer folds. Complete classical and neural participant-held-out results are available for the four-model comparison. Notebook 05 implements reconciliation and comparison of complete matched predictions without Test-informed candidate selection.

**Primary metric:** angular error in degrees between normalised predicted and true 3D gaze vectors; report mean, median, participant distribution and support counts. Grouped uncertainty recognises participants, not individual images, as independent units.

**Compute boundary:** a full participant-level neural LOPO run can require repeated deep-model fitting and must be reviewed after profiling original-source image dimensions, eligible people and Colab resources. The two-fold development configuration is explicitly **not** a final evaluation. Pretrained ResNet-18 weights may be downloaded independently from torchvision/ImageNet, and their use must be reported.

## 5. Methods, Models and Configurations

**Classical stage (Notebook 03; full outer LOPO executed):** each fold fits a **mean training gaze direction** and a **HOG + StandardScaler + Ridge** comparator. Ridge candidates `alpha ∈ {1, 10, 100}` are compared on that fold's inner Validation people. Continuous Ridge predictions are normalised to nonzero unit vectors before 3D angular-error calculation. The final run saves inner metrics, a plot, matched outer predictions and per-person metrics.

**Deep stage (Notebook 04; full outer LOPO executed):** a three-block compact grayscale CNN and ResNet-18 with a grayscale first convolution and optional ImageNet pretraining. Both produce 3D vectors, optimised using **mean cosine-direction loss** with AdamW. The fixed development and final configurations use a maximum of eight epochs, three-epoch early-stopping patience and batch size 128. The checkpoint with the lowest inner-Validation angular error is retained for each model/fold, and parameter counts and epoch histories are saved. PyTorch's CUDA/CPU device is reported. The final run evaluated all outer folds; outer-Test performance is not used for hyperparameter or model choice.

**Stage-specific utilities:** [`mpiigaze_io.py`](code/src/mpiigaze_io.py) inventories/downloads/extracts and inspects source MAT headers; [`mpiigaze_data.py`](code/src/mpiigaze_data.py) parses selected normalised eye data; [`gaze_geometry.py`](code/src/gaze_geometry.py) implements unit-vector angular error; [`gaze_evaluation.py`](code/src/gaze_evaluation.py) validates Fold and sample identity; [`gaze_models.py`](code/src/gaze_models.py) implements the classical and deep model families. Their original-data compatibility remains to be verified.

## 6. Results and Comparative Analysis

**Recorded source-audit evidence.** The executed [Notebook 01](code/notebooks/01_mpiigaze_source_audit_and_protocol.ipynb) recorded 430,180 archive file paths, of which 215,386 are operating-system companions and 214,794 are non-companion paths.

| Non-companion source component | File paths |
|---|---:|
| Original | 214,224 |
| Normalized | 521 |
| Evaluation Subset | 31 |
| Annotation Subset | 15 |
| Other | 3 |
| Total | 214,794 |

The [participant support table](output/01_participant_support.csv) identifies all 15 participants (p00–p14). Of 1,042 candidate normalised MAT paths, 521 are companion paths and 521 are participant/day MAT paths, with 7–69 per participant. The [MAT header table](output/01_mat_schema.csv) records 30 variables across 15 representative MATLAB v5 files: a `data` struct and a `filenames` cell array in each file. The [list inventory](output/01_evaluation_list_inventory.csv) includes 15 annotation lists, 30 evaluation lists and one ReadMe. These are path and header counts, not verified image–gaze pairs. See the [source inventory plot](output/01_source_inventory.png).

**Recorded image-preparation evidence (Notebook 02).** The decoder processed 213,658 left-eye source rows and retained 7,500 grayscale images (36 × 60 pixels), 500 per participant. The decoded count differs by one from the published image count of 213,659; its cause is unresolved. The [sample index](output/02_sample_index.csv) contains 7,500 unique IDs spanning 510 selected source MAT files. All saved gaze vectors have unit norms within numerical tolerance, and the image intensity and contrast statistics are finite. Mean intensity ranges from 83.87 to 216.38; pixel SD ranges from 66.67 to 86.94. Participant mean pixel SD ranges from 73.01 to 75.27 (see [quality plot](output/02_image_quality.png)). These are descriptive pixel statistics rather than independent image-quality labels.

The [fold table](output/02_fold_assignments.csv) contains 225 participant-role records: 15 folds with all 15 participants represented exactly once in each fold, 12 Training, two inner Validation and one Test person. Each fold contains 6,000, 1,000 and 500 images respectively. Every participant serves as outer Test once. No cross-role participant overlap was found. The local NPZ is not published; the recorded Notebook 03 run passed the cache loader's order and target-correspondence checks. The source physical axis convention is not independently confirmed.

**Recorded classical development evidence (Notebook 03).** The HOG feature matrix is 7,500 × 360. The saved [inner-Validation metrics](output/03_inner_validation_metrics.csv) contain 60 candidate/fold records and the [configuration table](output/03_run_configuration.csv) contains 30 selected model/fold records. All 15 folds contain two inner Validation participants. The completed final run also scored the outer Test person in every fold.

| Model | Mean fold error (°) | Median (°) | Minimum (°) | Maximum (°) |
|---|---:|---:|---:|---:|
| Mean direction | 9.0481 | 9.1004 | 8.5092 | 9.9054 |
| HOG + Ridge | 7.6961 | 7.7551 | 6.4557 | 8.7646 |

The selected Ridge alpha is 100 in every fold, the upper boundary of the tested grid (1, 10, 100). Mean fold errors for these candidates are 7.7326°, 7.7289° and 7.6961° respectively. The image-based comparator reduced mean fold error by 1.3519° (14.94%) and outperformed the reference in 14 of 15 configurations; outer_03 had errors of 8.7646° versus 8.5672°. The [comparison plot](output/03_baseline_comparison.png) preserves this exception. Mean-direction scores were recalculated from the published sample targets and fold roles, agreeing with the saved values within 0.000005°. HOG/Ridge fits have not been independently repeated because the image cache is local.

These are descriptive inner-Validation results across overlapping participant configurations, not independent fold replicates.

**Classical outer-LOPO results (Notebook 03):**

| Outer-Test metric (°) | Mean direction | HOG + Ridge |
|---|---:|---:|
| Participant-macro mean | 9.1601 | 7.5596 |
| Median participant mean | 8.8903 | 7.5180 |
| Pooled image median | 8.9879 | 7.0152 |
| Pooled RMSE | 10.0070 | 8.6314 |
| Minimum participant mean | 8.2702 | 5.7664 |
| Maximum participant mean | 10.9189 | 9.6914 |

Both models cover all 7,500 samples once, yielding 15,000 matched outer-Test predictions and 30 participant/model metric records. HOG + Ridge reduced participant-macro mean error by 1.6005° (17.47%), with lower error for 14 of 15 people. The exception is p09 (9.6914° versus 8.6698° for mean direction). See the [outer predictions](output/03_baseline_outer_predictions.csv) and [participant metrics](output/03_baseline_fold_metrics.csv). Independent recalculation of angular errors agrees with the saved values within 0.000002°; the reference directions also agree with the Training-derived means. HOG/Ridge fitting was not independently repeated. The internal LOPO estimates are not external validation or evidence of statistical significance.

**Recorded neural full outer-LOPO evidence (Notebook 04).** The source-aligned run completed all 15 folds.

| Outer-Test metric (°) | Small CNN | ResNet-18 |
|---|---:|---:|
| Participant-macro mean | 7.3048 | 8.2304 |
| Median participant mean | 7.2657 | 8.0091 |
| Pooled image median | 6.8695 | 7.7005 |
| Pooled RMSE | 8.2493 | 9.3255 |
| Minimum participant mean | 4.9781 | 6.3302 |
| Maximum participant mean | 9.5559 | 9.6792 |

| Model | Inner Validation (°) | Outer Test (°) |
|---|---:|---:|
| Mean direction | 9.0481 | 9.1601 |
| HOG + Ridge | 7.6961 | 7.5596 |
| Small CNN | 6.7633 | 7.3048 |
| ResNet-18 | 8.1554 | 8.2304 |

The 30 neural fits ran for 238 epochs on CUDA (Tesla T4). Small CNN in outer_07 and ResNet-18 in outer_12 stopped at epoch seven after their best epoch four; the other 28 fits reached epoch eight. All 30 recorded selected epochs match their minimum inner-Validation errors. The networks have 48,387 and 11,171,779 parameters, respectively. Each neural procedure covers all 7,500 indexed samples exactly once. Independent recalculation of angular errors agrees with the saved predictions within 0.000002°. Checkpoint weights remain local and were not independently reloaded. Small CNN has the lowest participant-macro outer-Test mean, but the difference from HOG + Ridge is only 0.2549°. Small CNN, HOG + Ridge and mean direction have the lowest participant error for eight, six and one people respectively. These are descriptive internal results, not evidence of statistical significance or external validity. Parameter counts are not measured runtime or memory. See [training history](output/04_training_history.csv), [selected metrics](output/04_inner_validation_metrics.csv), [configuration](output/04_run_configuration.csv), [outer predictions](output/04_neural_outer_predictions.csv) and [participant metrics](output/04_neural_outer_metrics.csv). Both prediction tables are complete; Notebook 05 analysis remains unexecuted.

The [learning curves](output/04_validation_curves.png) show 15 contributing folds per model through epoch seven and 14 at epoch eight. Changing fold membership at the final epoch limits direct comparison of consecutive means. The mean of individually selected checkpoint errors is distinct from the per-epoch mean. The difference between neural architectures cannot be causally attributed to pretraining, parameter count or learning rate without controlled comparisons.

## 7. Robustness, Uncertainty and Additional Analyses

The implemented final analysis defines participant-macro mean angular error as primary, with pooled mean, median, SD and RMSE as descriptive secondary summaries. It reports all six paired model contrasts using 5,000 participant-summary bootstrap resamples (seed 17), with descriptive 95% percentile intervals conditional on the fitted predictions. Overlapping LOPO training sets and retraining uncertainty are not represented. Image mean-intensity and pixel-SD tertiles are formed from the fixed sample index without using errors; summaries give participant and image counts and participant-macro errors. No physical illumination, head-pose, external validation or latency claims follow. The complete outer predictions are available; these analyses await execution of Notebook 05.

Planned checks include eye-side orientation, image quality/missingness, head pose and illumination **where directly supported by the inspected source**, per-participant variability, configuration stability and inference cost. A source field is not assumed to exist simply because a publication describes a related dataset.

Potential confounding includes repeated days/images for one person, unverified camera normalisation, annotation differences, calibration requirements and invalid 3D label conventions. Neither a synthetic geometry test nor a successful source file inventory validates these scientific assumptions.

## 8. Current Outputs and Reproducibility

The repository contains **four recorded original-data executions (01–04)** and one implemented, unexecuted Notebook 05; a source adapter, data reader, gaze geometry, participant evaluation and model utility; four editable five-band scientific figures ([01](images/01_mpiigaze_source_audit_and_protocol.svg), [02](images/02_eye_images_geometry_and_quality.svg), [03](images/03_classical_gaze_estimation_baselines.svg), [04](images/04_cnn_and_transfer_learning.svg)); and the two study documents. Synthetic tests verify safe source inventory/extraction, not original-data validity. **The raw MPIIGaze archive and full image cache are excluded from Git; derived metadata are published. Both classical and neural outer-Test prediction tables are committed.** The source-audit notebook creates stage-owned metadata and a figure only after it has opened original source files.

Intended stage-specific files, all **planned until genuinely produced**:

| Producer | Intended files and consumers |
| --- | --- |
| 01 | `01_file_inventory.csv`, `01_participant_support.csv`, `01_mat_schema.csv`, `01_evaluation_list_inventory.csv`, `01_protocol.csv`, `01_source_inventory.png` (recorded source-audit outputs); source reconciliation for Stage 02 |
| 02 | `02_sample_index.csv`, `02_fold_assignments.csv`, `02_source_sample_counts.csv`, `02_cache_location.csv`, `02_image_quality.png`; local ignored `02_normalised_eye_cache.npz` used by 03–04 |
| 03 | Development: `03_inner_validation_metrics.csv`, `03_run_configuration.csv`, `03_baseline_comparison.png`. Full outer LOPO, when explicitly enabled: `03_baseline_outer_predictions.csv`, `03_baseline_fold_metrics.csv` |
| 04 | Development: `04_training_history.csv`, `04_inner_validation_metrics.csv`, `04_model_checkpoints.csv`, `04_run_configuration.csv`, `04_validation_curves.png` and ignored checkpoint weights. Full outer LOPO, when enabled: `04_neural_outer_predictions.csv`, `04_neural_outer_metrics.csv` |
| 05 | `05_model_metrics.csv`, `05_participant_metrics.csv`, `05_paired_differences.csv`, `05_image_stratum_metrics.csv`, `05_model_complexity.csv`, three comparison/uncertainty figures |

Use VS Code connected to the remote Colab runtime, the established repository setup and the study-level `output/` directory. Raw archives, extracted source, image cache and checkpoints remain in ignored runtime directories. Only stage-owned analysis outputs are packaged for manual download from Colab Contents; the user extracts these into the study's local `output/` directory and synchronises the reviewed files through Git. Rerunning an implemented stage overwrites its own current named outputs. Do not silently retrain models when a reporting stage opens missing files.

## 9. Discussion, Limitations and Future Work

The study extends the programme from already-recorded gaze/EEG signals to **camera-image gaze prediction**, with a separate source, 3D normalised gaze frame and angular-error metric. Notebooks 01–04 have recorded source-audit, image-preparation and complete classical/neural outer-LOPO evidence; Notebook 05 implements final analysis but is not yet executed. Internal held-out results are available; external generalisation is not established.

The principal early limitations are **unverified original MAT source/label conventions**, the unresolved annotated-subset count definitions, participant-level image dependence, a preliminary 500-per-person (not official benchmark) subset, unmeasured training and inference time and source licensing. The immediate operation is to **execute Notebook 05** using both complete prediction tables and the fixed sample and participant roles. Independent source-axis confirmation remains necessary for physical interpretation. Software tests cannot replace those checks.

**Reference:** Zhang, X., Sugano, Y., Fritz, M. and Bulling, A., MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation, IEEE TPAMI. [DOI](https://doi.org/10.1109/TPAMI.2017.2778103).
