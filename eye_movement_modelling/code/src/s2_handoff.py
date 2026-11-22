# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Restore current Eye-Movement Modelling stage outputs in Colab.

A user-uploaded stage ZIP is an explicit input selection and takes precedence
over existing stage files when an archive is explicitly supplied. This helper
does not download source data or modify scientific model settings.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from core.stage_outputs import ensure_stage_inputs


# ======================= 01. Explicit Stage ZIP Handoff ======================
def ensure_s2_stage_inputs(
    output_dir: str | Path, upstream_stage: int,
    required: tuple[str, ...], *, upload_dir: str | Path = "/content",
) -> list[Path]:
    """Prefer an explicitly uploaded current-stage ZIP over tracked old files.

    Restore in a temporary folder, validate all required filenames, then
    replace only files belonging to the supplied stage. Never merge ZIPs
    from different cohorts in one run; downstream identity checks still
    verify that dependent stages share the same protocol and split.
    """
    folder = Path(output_dir).resolve()
    stage = f"{upstream_stage:02d}"
    for name in required:
        if Path(name).name != name or not name.startswith(stage + "_"):
            raise ValueError(f"Invalid stage-{stage} input filename: {name}")
    archive_path = Path(upload_dir) / (
        f"{folder.parent.name}_Notebook_{stage}_Outputs.zip"
    )
    if not archive_path.is_file():
        return ensure_stage_inputs(
            folder, upstream_stage, required, upload_dir=upload_dir,
        )

    folder.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(set(names)) != len(names) or not set(required).issubset(names):
            raise ValueError("Stage ZIP has duplicate or missing required files")
        for member in members:
            name = member.filename
            if (
                member.is_dir() or Path(name).name != name
                or "\\" in name or not name.startswith(stage + "_")
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError(f"Unsafe stage ZIP member: {name}")
        with tempfile.TemporaryDirectory(
            prefix=f".stage_{stage}_", dir=folder,
        ) as temp_root:
            temporary = Path(temp_root)
            for member in members:
                with archive.open(member) as source, (
                    temporary / member.filename
                ).open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            for name in names:
                (temporary / name).replace(folder / name)

    print(f"Restored Notebook {stage} from uploaded ZIP: {archive_path}")
    return [folder / name for name in required]


# ===================== 02. Main-Cohort Compatibility ========================
def verify_main_cohort(
    protocol: dict, split: pd.DataFrame,
) -> dict[str, int]:
    """Verify a compatible participant-disjoint modelling cohort."""
    required = {"participant_id", "record_id", "task", "split"}
    if not required.issubset(split.columns):
        raise ValueError("Source split is missing identity or role columns")
    participants = split["participant_id"].astype(str)
    count = participants.nunique()
    if (
        protocol.get("purpose") != "main"
        or count < 20
        or count != int(protocol["max_participants"])
    ):
        raise ValueError(
            "Expected the current main cohort from Notebook 01, but found "
            f"{count} participants and purpose={protocol.get('purpose')!r}. "
            "Supply matching stage-01 outputs and the corresponding protocol."
        )
    if "selected_source_ids" in protocol:
        recorded = {f"{int(sid) % 1000:03d}"
                    for sid in protocol["selected_source_ids"]}
        if len(recorded) != count or recorded != set(participants):
            raise ValueError(
                "Protocol-selected original participant IDs differ from "
                "the current split. Do not combine different cohort runs."
            )
    if split["record_id"].duplicated().any():
        raise ValueError("Duplicate source record IDs in main split")
    if split.groupby("participant_id")["split"].nunique().max() != 1:
        raise ValueError("Participant leakage across Train/Validation/Test")
    if "study_purpose" in split and not split["study_purpose"].eq("main").all():
        raise ValueError("Main split and saved study purpose disagree")
    if split.groupby("participant_id")["task"].nunique().ne(4).any():
        raise ValueError("Every selected person requires all four tasks")
    if set(split["split"]) != {"train", "validation", "test"}:
        raise ValueError("Main cohort requires all three participant roles")
    counts = {
        role: int(split.loc[split["split"].eq(role), "participant_id"].nunique())
        for role in ("train", "validation", "test")
    }
    if min(counts.values()) == 0:
        raise ValueError("Empty participant role in main cohort")
    return counts
