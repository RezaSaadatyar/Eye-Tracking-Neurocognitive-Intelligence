# Vision-Based Gaze Estimation

**Status:** Notebooks **01–04** contain the source-audit, image/target preparation, classical baseline and CNN/transfer-learning code, written in the repository's established Colab and figure/export format. **Notebook 05 is still a starter**. No original MPIIGaze images have been processed in this repository, and no gaze-model training or angular-error results have been verified.

This independent study investigates whether camera-derived eye images can estimate **3D gaze direction for previously unseen participants**, using evidence-based image geometry, classical baselines and compact deep models. Its inputs, coordinate frames and evaluation procedures are **not interchangeable** with the GazeBase recordings or ZuCo EEG–gaze features in other studies.

**Scientific document:** [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

![Five-band MPIIGaze source audit and evaluation protocol schematic](images/01_mpiigaze_source_audit_and_protocol.svg)

*Figure 1. Original-data acquisition, source metadata, participant provenance and proposed evaluation design. The schematic shows the implemented source-audit operations; no MPIIGaze measurements or model results are claimed.*

## 1. Dataset and Experimental Protocol

| Aspect | Proposed protocol and verification status |
| --- | --- |
| Source | [MPIIGaze official dataset](https://www.collaborative-ai.org/research/datasets/MPIIGaze/), [archival distribution](https://doi.org/10.18419/darus-3230); public description: 15 people, 213,659 collected frames; exact selected subset **not yet inspected** |
| Permissions | CC BY-NC-SA 4.0 reported by official providers. Original images/annotations stay outside Git; confirm the selected files and reuse conditions before extraction |
| Predictors | Proposed **normalised grayscale left-eye crops**, initially at most 500 source-matched examples per eligible person. The sampling rule and source coordinate interpretation require original-data verification; head pose is excluded from the initial models |
| Target | Original-source **3D gaze direction** with a documented camera/normalised coordinate frame; 2D screen point is not the initial target |
| Prediction unit | One source-labelled image or eye sample; person/session/day IDs retained for grouping, not an unintended predictor |
| Evaluation unit | Independent **participant**; deterministic outer leave-one-person-out (LOPO) folds with **two distinct inner-Validation people** per fold, subject to a source-confirmed eligible cohort. Notebook 03 and 04 keep outer-Test scoring **disabled by default** |
| Main metric | Mean/median **3D angular error in degrees**; per-person dispersion; 2D pixel error only if an appropriate separate geometry protocol is approved |

The official description distinguishes Data, Evaluation Subset and Annotation Subset, and contains an unresolved inconsistency about annotated-subset counts. **Notebook 01 must establish the actual chosen file structure before any sample counts, image-label joins or training are claimed.**

## 2. Experimental Programme

These are **five complementary notebooks**, not five independent studies. Notebooks **01–04 have implemented code and stage-owned output contracts**, while Notebook **05 has not yet been developed beyond its starter structure**. A successful software test is not evidence that the real dataset has been downloaded, the MAT field mapping independently verified, or a model trained. All four developed notebooks use the same Colab setup, grouped imports, shared plotting utilities and stage-specific ZIP exports.

| Stage | Notebook | Responsibility | Intended current outputs |
| --- | --- | --- | --- |
| 01 | [Source audit and protocol](code/notebooks/01_mpiigaze_source_audit_and_protocol.ipynb) | Implemented original-data access, safe archive extraction, participant/day and MAT schema audit, evaluation-list inspection and proposed protocol | **When executed on actual source:** `01_file_inventory.csv`, `01_participant_support.csv`, `01_mat_schema.csv`, `01_evaluation_list_inventory.csv`, `01_protocol.json` and `01_source_inventory.png` |
| 02 | [Eye images, geometry and quality](code/notebooks/02_eye_images_geometry_and_quality.ipynb) | Implemented normalised-image/3D label decoder, initial 500-per-person source sampling, local-only image cache, participant LOPO/inner folds and pixel-quality plots | `02_sample_index.csv`, `02_fold_assignments.csv`, `02_source_sample_counts.csv`, `02_cache_location.json`, `02_image_quality.png`; ignored local NPZ |
| 03 | [Classical baselines](code/notebooks/03_classical_gaze_estimation_baselines.ipynb) | Implemented Train-only mean direction and HOG + Ridge with nested inner-Validation alpha selection; outer Test **off initially** | `03_inner_validation_metrics.csv`, `03_run_configuration.json`, `03_baseline_comparison.png`; optional explicit `03_baseline_outer_predictions.csv` and `03_baseline_fold_metrics.csv` |
| 04 | [CNN and transfer learning](code/notebooks/04_cnn_and_transfer_learning.ipynb) | Implemented Small CNN and optional ImageNet-pretrained ResNet-18; Train/inner-Validation checkpointing, two-fold development configuration, outer Test **off initially** | `04_training_history.csv`, `04_inner_validation_metrics.csv`, `04_model_checkpoints.csv`, `04_run_configuration.json`, `04_validation_curves.png`; local-only checkpoints, optional explicit Test predictions |
| 05 | [Cross-person evaluation and robustness](code/notebooks/05_cross_person_evaluation_and_robustness.ipynb) | Reconcile outer-fold predictions, report participant-level angular error and uncertainty | `05_model_metrics.csv`, per-person results, error figure |

**Dependency order:** 01 → 02 → (03 and 04 use the **same sample index and fold assignments**) → 05, after Notebook 05 has been implemented. Each outer LOPO person is excluded from fitting and inner model selection. Development mode reports inner-Validation only, and full outer-Test evaluation needs deliberate activation and a frozen protocol. Neither model selection nor retrospective retuning may use the outer Test.

## 3. Main Results

**No original-data results are available yet.** Notebooks 01–04 and their project-specific Python modules are implemented, but none has been verified against the real MPIIGaze archive in this repository. No valid original-image dataset subset, model checkpoints, participant scores or outer-fold Test predictions have been observed or published. Publisher-reported counts are not experimental outcomes of this study.

## 4. Running and Reviewing the Study

### Review

Read the [methods and source requirements](RESEARCH_METHODS_AND_EVIDENCE.md) before running the notebooks. The notebooks use the repository's existing Colab checkout and `setup_colab("vision_based_gaze_estimation")` helper.

### Start with Notebook 01

1. Obtain [MPIIGaze from the official DaRUS archive](https://doi.org/10.18419/darus-3230) under its non-commercial research licence. The publisher also offers a direct multi-GB `MPIIGaze.tar.gz` archive.
2. Either place the **downloaded `MPIIGaze.zip` or `MPIIGaze.tar.gz`** under the repository-root `data/raw/` folder, or place already extracted files under the Git-ignored `data/raw/mpiigaze/` folder. The notebook can safely extract an existing archive.
3. Start [Notebook 01](code/notebooks/01_mpiigaze_source_audit_and_protocol.ipynb) in a new Colab runtime. Its initial mode is **`SOURCE_MODE = "local"`**, so an absent source prints acquisition guidance and skips data-dependent operations without fabricating results. To download the publisher's large archive in Colab, deliberately set **`SOURCE_MODE = "download"`** in Section 02; confirm sufficient disk space first.
4. Rerun Sections 02–09 with the original source files present. They produce observed source inventories, a bounded MATLAB-header sample, the proposed participant-level protocol and a source-file chart, plus an output ZIP. Notebook 01 does not yet construct image-level 3D targets or run models.
5. Run [Notebook 02](code/notebooks/02_eye_images_geometry_and_quality.ipynb) **in the same Colab environment or with the same ignored local cache**, and inspect its actual MAT parsing, image/label identities, 3D gaze directions and participant-fold table before accepting downstream data.
6. Then run [Notebook 03](code/notebooks/03_classical_gaze_estimation_baselines.ipynb) and [Notebook 04](code/notebooks/04_cnn_and_transfer_learning.ipynb) in their default **inner-Validation-only development mode**. Notebook 04 initially trains two outer-fold development configurations, using Colab GPU when available. Pretrained ResNet weights may require a separate internet download. Do not turn on outer-Test scoring for routine debugging.
7. Every stage packages its current `NN_*` files as a ZIP for manual Colab download. The original image cache and neural weights stay under ignored `data/processed/mpiigaze/` and do not travel in the ZIP. Notebook 05 will be implemented only after source-consistent results and an approved final evaluation protocol exist.

The approved architecture uses standard Python and Google Colab. Dataset and model-version requirements are provisional in [requirements.txt](requirements.txt), and will be frozen with actual runs.

## 5. Files and Saved Outputs

| Location | Responsibility |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Notebooks 01–04 implemented but not original-data-executed; Notebook 05 remains a starter |
| [`code/src/`](code/src/) | [Source retrieval/inspection](code/src/mpiigaze_io.py), [normalised-image reader](code/src/mpiigaze_data.py), [3D geometry](code/src/gaze_geometry.py), [grouped evaluation](code/src/gaze_evaluation.py) and [classical/CNN models](code/src/gaze_models.py) |
| [`images/`](images/) | Four editable five-band diagrams for [source audit](images/01_mpiigaze_source_audit_and_protocol.svg), [3D gaze/image preparation](images/02_eye_images_geometry_and_quality.svg), [classical baselines](images/03_classical_gaze_estimation_baselines.svg) and [CNN modelling](images/04_cnn_and_transfer_learning.svg); diagrams describe methods rather than measured results |
| [`requirements.txt`](requirements.txt) | Proposed dependencies, to be pinned for executed experiments |
| `data/raw/mpiigaze/` (repository root) | Original local source data; ignored by Git and **not included** in this repository |
| `output/` (study root) | Stage-owned outputs generated only when an original-data stage is run; no real MPIIGaze evidence files have been verified here |
| [`RESEARCH_METHODS_AND_EVIDENCE.md`](RESEARCH_METHODS_AND_EVIDENCE.md) | Scientific question, data contracts, evaluation plan, source qualification and implementation boundaries |

## 6. Interpretation and Scope

The current contribution includes **implemented source, geometry, baseline and neural development code (Notebooks 01–04)** and four editable scientific diagrams, but **not an original-data-executed or validated gaze-estimation system**. Future image/target files must preserve consistent 3D coordinate conventions and left/right-eye transformations; otherwise even numerically correct angular errors would lack scientific meaning. A 15-person source cannot justify treating repeated images or days as independent people. Results about unseen participants must be derived only from a participant-disjoint protocol.

Face-based gaze estimation, gaze-point calibration, cross-dataset robustness and compact Transformers are possible **future extensions**, not current outcomes. The [source study](https://doi.org/10.1109/TPAMI.2017.2778103) should be cited for the original MPIIGaze dataset and its methods. Dataset access remains subject to CC BY-NC-SA 4.0 terms.
