# Dataset Access and Provenance

The five studies use four research datasets. Original datasets retain their respective source licences; the repository's MIT licence does not relicense those materials.

| Dataset | Source and licence | Study |
| --- | --- | --- |
| GazeBase v2.0 | [Figshare](https://doi.org/10.6084/m9.figshare.12912257) · [Scientific Data](https://doi.org/10.1038/s41597-021-00959-y) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Foundations and Eye-Movement Modelling |
| ZuCo 2.0 | [NEMAR nm000279](https://doi.org/10.82901/nemar.nm000279) · CC BY 4.0 as reported in BIDS metadata | EEG–Eye-Tracking Fusion |
| MPIIGaze | [DaRUS](https://doi.org/10.18419/darus-3230) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | Vision-Based Gaze Estimation |
| OneStop ordinary reading | [OSF](https://osf.io/zn9sq/) · [dataset documentation](https://lacclab.github.io/OneStop-Eye-Movements/variables) · CC BY 4.0 eye-tracking data; CC BY-SA 4.0 OneStopQA text and auxiliary annotations | Reading Comprehension Modelling |

## GazeBase

The repository includes 40 original Round 1 / Session 1 recordings from subjects 1001–1010, four tasks per participant: FXS, HSS, RAN and TEX. Credit Griffith, Lohr, Abdulin and Komogortsev, cite the original release and acknowledge the subset selection. Preserve original filenames, e.g. `S_1001_S1_FXS.csv`, under `data/raw/gazebase/GazeBase_v2_0/Round_1/`. The source reader discovers the nested files. The [Foundations study](../eye_tracking_foundations/README.md) uses this ten-person subset; the [modelling study](../eye_movement_modelling/README.md) acquires its separate 100-person cohort from the verified original archive. Consult each Notebook 01 for selection and acquisition. The full archive is approximately 6.7 GB.

## ZuCo 2.0

The [EEG–gaze study](../eeg_eye_tracking_fusion/README.md) uses 18 participants and selected NR/TSR Run-1 recordings. Notebook 01 accesses the original BIDS files through NEMAR; the [source guide](../eeg_eye_tracking_fusion/data/README.md) identifies the release. Original BrainVision EEG and EyeLink recordings are not bundled.

## MPIIGaze

The [vision study](../vision_based_gaze_estimation/README.md) uses a study-defined 7,500-image subset of the 15 participants. Notebook 01 obtains and inventories the source archive and Notebook 02 prepares the aligned image and gaze representations. Original images and reconstructed image arrays are required for model fitting. Observe the source non-commercial and share-alike terms.

## OneStop

The [reading-comprehension study](../reading_comprehension_modelling/README.md) uses ordinary first reading: 9,718 eligible trials from 180 participants and 30 articles. Notebook 01 obtains the fixation and interest-area reports from the official ordinary-reading component. Notebook 02 constructs the scalar, lexical and sequence representations required for modelling. See [the source publication](https://doi.org/10.1038/s41597-025-06272-2) and the study methods for the cohort definition and source terms.

## Reuse

Review the study README and scientific document before acquiring data. Use the original release and source identifiers to reproduce the stated cohort. Cite the original authors and comply with the applicable licence when reusing source or derived materials. Each study maintains separate experimental representations and outputs.
