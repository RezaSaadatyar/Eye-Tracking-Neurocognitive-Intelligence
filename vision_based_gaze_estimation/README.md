# Vision-Based Gaze Estimation

**Status:** Notebook 01 now has a complete executable source-audit workflow and a shared MPIIGaze archive/metadata reader. Notebooks 02–05 remain initial starters. The original source has **not** been executed in this repository; there are no image-based model results or evaluated angular errors.

This independent study investigates whether camera-derived eye images can estimate **3D gaze direction for previously unseen participants**, using evidence-based image geometry, classical baselines and compact deep models. Its inputs, coordinate frames and evaluation procedures are **not interchangeable** with the GazeBase recordings or ZuCo EEG–gaze features in other studies.

**Scientific document:** [Research Methods and Evidence](RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Dataset and Experimental Protocol

| Aspect | Proposed protocol and verification status |
| --- | --- |
| Source | [MPIIGaze official dataset](https://www.collaborative-ai.org/research/datasets/MPIIGaze/), [archival distribution](https://doi.org/10.18419/darus-3230); public description: 15 people, 213,659 collected frames; exact selected subset **not yet inspected** |
| Permissions | CC BY-NC-SA 4.0 reported by official providers. Original images/annotations stay outside Git; confirm the selected files and reuse conditions before extraction |
| Predictors | Source-verified normalised eye crops; optional head-pose information only if the source provides it and the comparison is explicitly defined |
| Target | Original-source **3D gaze direction** with a documented camera/normalised coordinate frame; 2D screen point is not the initial target |
| Prediction unit | One source-labelled image or eye sample; person/session/day IDs retained for grouping, not an unintended predictor |
| Evaluation unit | Independent **participant**; proposed outer leave-one-person-out (LOPO) with inner participant-disjoint model selection, subject to source inspection and compute review |
| Main metric | Mean/median **3D angular error in degrees**; per-person dispersion; 2D pixel error only if an appropriate separate geometry protocol is approved |

The official description distinguishes Data, Evaluation Subset and Annotation Subset, and contains an unresolved inconsistency about annotated-subset counts. **Notebook 01 must establish the actual chosen file structure before any sample counts, image-label joins or training are claimed.**

## 2. Experimental Programme

These are **five complementary notebooks**, not five independent studies. **Notebook 01** now contains source access, archive inspection, participant/day audit, MATLAB-header and evaluation-list checks, a proposed evaluation protocol, figures and output packaging. **Notebooks 02–05** remain structured starters; their scientific modelling and evaluation stages have not been implemented or executed.

| Stage | Notebook | Responsibility | Intended current outputs |
| --- | --- | --- | --- |
| 01 | [Source audit and protocol](code/notebooks/01_mpiigaze_source_audit_and_protocol.ipynb) | Implemented original-data access, safe archive extraction, participant/day and MAT schema audit, evaluation-list inspection and proposed protocol | **When executed on actual source:** `01_file_inventory.csv`, `01_participant_support.csv`, `01_mat_schema.csv`, `01_evaluation_list_inventory.csv`, `01_protocol.json` and `01_source_inventory.png` |
| 02 | [Eye images, geometry and quality](code/notebooks/02_eye_images_geometry_and_quality.ipynb) | Validate image/target frames, sample alignment, eye side and participant-held-out fold construction | `02_sample_index.csv`, `02_fold_assignments.csv`, quality figure |
| 03 | [Classical baselines](code/notebooks/03_classical_gaze_estimation_baselines.ipynb) | Mean-gaze and image-descriptor regression with outer-person independence and inner validation | `03_baseline_outer_predictions.csv`, fold metrics |
| 04 | [CNN and transfer learning](code/notebooks/04_cnn_and_transfer_learning.ipynb) | Compact CNN and justified lightweight network on the same folds | `04_neural_outer_predictions.csv`, training history |
| 05 | [Cross-person evaluation and robustness](code/notebooks/05_cross_person_evaluation_and_robustness.ipynb) | Reconcile outer-fold predictions, report participant-level angular error and uncertainty | `05_model_metrics.csv`, per-person results, error figure |

**Dependency order:** 01 → 02 → (03 and 04 use the same fixed protocol) → 05. Under outer LOPO, each held-out person is evaluated by a model trained without that person. No overall winner may be selected post hoc by inspecting outer-fold Test scores.

## 3. Main Results

**No original-data results are available yet.** Notebook 01 and its archive/MAT source utilities are implemented and covered by synthetic tests, but their execution against the actual MPIIGaze archive has not been verified. The other notebooks have not fitted models or generated outer-fold predictions. Publisher-reported image counts must not be presented as this study's observed results.

## 4. Running and Reviewing the Study

### Review

Read the [methods and source requirements](RESEARCH_METHODS_AND_EVIDENCE.md) before running the notebooks. The notebooks use the repository's existing Colab checkout and `setup_colab("vision_based_gaze_estimation")` helper.

### Start with Notebook 01

1. Obtain [MPIIGaze from the official DaRUS archive](https://doi.org/10.18419/darus-3230) under its non-commercial research licence. The publisher also offers a direct multi-GB `MPIIGaze.tar.gz` archive.
2. Either place the **downloaded `MPIIGaze.zip` or `MPIIGaze.tar.gz`** under the repository-root `data/raw/` folder, or place already extracted files under the Git-ignored `data/raw/mpiigaze/` folder. The notebook can safely extract an existing archive.
3. Start [Notebook 01](code/notebooks/01_mpiigaze_source_audit_and_protocol.ipynb) in a new Colab runtime. Its initial mode is **`SOURCE_MODE = "local"`**, so an absent source prints acquisition guidance and skips data-dependent operations without fabricating results. To download the publisher's large archive in Colab, deliberately set **`SOURCE_MODE = "download"`** in Section 02; confirm sufficient disk space first.
4. Rerun Sections 02–09 with the original source files present. They produce observed source inventories, a bounded MATLAB-header sample, the proposed participant-level protocol and a source-file chart, plus an output ZIP. Notebook 01 does not yet construct image-level 3D targets or run models.
5. Notebook 02 and later modelling stages must wait for the source-specific sample/target geometry and fold protocol to be implemented and validated.

The approved architecture uses standard Python and Google Colab. Dataset and model-version requirements are provisional in [requirements.txt](requirements.txt), and will be frozen with actual runs.

## 5. Files and Saved Outputs

| Location | Responsibility |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Five numbered study notebooks, currently at scaffold/source-contract stage |
| [`code/src/`](code/src/) | 3D angular-error helper and [MPIIGaze source reader](code/src/mpiigaze_io.py), including safe archive extraction, observed file inventory and MAT-header inspection |
| [`requirements.txt`](requirements.txt) | Proposed dependencies, to be pinned for executed experiments |
| `data/raw/mpiigaze/` (repository root) | Original local source data; ignored by Git and **not included** in this repository |
| `output/` (study root) | Stage-owned outputs generated only when an original-data stage is run; no real MPIIGaze evidence files have been verified here |
| [`RESEARCH_METHODS_AND_EVIDENCE.md`](RESEARCH_METHODS_AND_EVIDENCE.md) | Scientific question, data contracts, evaluation plan, source qualification and implementation boundaries |

## 6. Interpretation and Scope

The current contribution includes an **implemented Notebook 01 source-audit workflow** and a five-stage research architecture, **not yet an original-data-executed or evaluated gaze-estimation system**. Future image/target files must preserve consistent 3D coordinate conventions and left/right-eye transformations; otherwise even numerically correct angular errors would lack scientific meaning. A 15-person source cannot justify treating repeated images or days as independent people. Results about unseen participants must be derived only from a participant-disjoint protocol.

Face-based gaze estimation, gaze-point calibration, cross-dataset robustness and compact Transformers are possible **future extensions**, not current outcomes. The [source study](https://doi.org/10.1109/TPAMI.2017.2778103) should be cited for the original MPIIGaze dataset and its methods. Dataset access remains subject to CC BY-NC-SA 4.0 terms.
