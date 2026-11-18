# Dataset Access and Provenance

The programme uses externally published research datasets. No dataset files have been downloaded into this repository or validated against the proposed study protocols at scaffold creation.

## Initial source candidates

| Source | Proposed role | Source reference | Status |
| --- | --- | --- | --- |
| GazeBase | Primary candidate for S1 gaze-sample quality, timebase and eye-event examples | [Dataset](https://doi.org/10.6084/m9.figshare.12912257) and [publication](https://doi.org/10.1038/s41597-021-00959-y) | Candidate only; exact files and permissions not inspected in this repository |
| EMTeC | Optional S1 complementary raw-to-derived reading and correction example | [Publication](https://doi.org/10.3758/s13428-025-02677-4) | Candidate only; inclusion not approved |
| OneStop | Potential later reading-behaviour application source | See the Research Blueprint | Not selected for S1 |
| ZuCo | Potential later paired EEG–eye-tracking source | See the Research Blueprint | Not selected or downloaded |
| MPIIGaze / Gaze360 | Potential separate image-based gaze study | See the Research Blueprint | Not selected or downloaded |

The source links identify published materials; they do not confirm that a particular archive, licence, event annotation or processed representation is available for immediate use.

## Local organisation and access checks

Use `data/raw/` for legitimately obtained original datasets and `data/processed/` for necessary local reconstruction artefacts if these locations are selected during implementation. These directories are Git-ignored by default; they are not distributed with a fresh clone. Do not place private credentials, identifiable participant recordings or copyrighted stimulus materials in version control.

Before choosing a concrete release, record its source URL, version, licence and permitted uses. Inspect representative source files to verify timestamp units, coordinate system, validity fields, eye identity, participant/session/trial keys and any annotations. Preserve provenance between raw records, vendor-labelled events and newly derived events. Do not infer cohort counts or evaluation labels from publication summaries alone.

S1 may begin with a bounded GazeBase subset only after this source-level inspection. The exact filenames and download instructions will be documented once verified; no unsupported download command or fixed local path is prescribed here.
