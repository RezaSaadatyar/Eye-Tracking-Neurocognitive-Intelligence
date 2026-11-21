# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Stream existing gaze sample CSVs by complete record for bounded-RAM analysis."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from gaze_preprocessing import preprocess_record
from gaze_source import require_columns


# ====================== 01. Stream Current Stage Inputs ======================

def validate_notebook_01_inputs(
    manifest: pd.DataFrame, quality: pd.DataFrame,
) -> dict[str, int]:
    """Verify the recorded identities and row counts of Notebook 01 inputs."""
    required = ("record_id", "participant_id", "task")
    require_columns(manifest, required)
    require_columns(quality, (*required, "n_samples"))
    if manifest.empty or quality.empty:
        raise ValueError("Notebook 01 source tables must not be empty.")
    if manifest["record_id"].isna().any() or quality["record_id"].isna().any():
        raise ValueError("Notebook 01 record identifiers must be present.")
    if manifest["record_id"].duplicated().any() or quality["record_id"].duplicated().any():
        raise ValueError("Duplicate Notebook 01 record identifiers.")
    selected = manifest.set_index("record_id")
    audited = quality.set_index("record_id")
    if set(selected.index) != set(audited.index):
        raise ValueError("Notebook 01 manifest and quality record sets differ.")
    for field in ("participant_id", "task", "round_id", "session_id"):
        if field not in selected or field not in audited:
            continue
        a = selected.loc[audited.index, field].astype(str)
        b = audited[field].astype(str)
        if field == "participant_id":
            a, b = a.str.zfill(3), b.str.zfill(3)
        if not a.equals(b):
            raise ValueError(f"Notebook 01 {field} differs across input tables.")
    counts = pd.to_numeric(audited["n_samples"], errors="raise")
    if (
        counts.isna().any()
        or not (counts > 0).all()
        or not (counts == counts.astype("int64")).all()
    ):
        raise ValueError("Notebook 01 source sample counts must be positive integers.")
    return {str(rid): int(count) for rid, count in counts.items()}


def preprocess_stage_file(
    source_csv: str | Path,
    destination_csv: str | Path,
    *,
    smoothing_samples: int = 5,
    max_gap_ms: float = 5.0,
    chunksize: int = 80_000,
    expected_records: set[str] | None = None,
    expected_record_counts: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply unchanged record-level preprocessing without full-cohort RAM copies.

    Input records must be contiguous in the stage-01 CSV (as written by its
    record-concatenating producer). Each record is processed once, preserving
    all sample rows, source fields and method settings. The destination is
    replaced only after successful completion; an incomplete temporary file
    is removed on failure. Returns compact record summaries and one plot trace.
    """
    source = Path(source_csv)
    destination = Path(destination_csv)
    if not source.is_file():
        raise FileNotFoundError(f"Missing Notebook 01 sample table: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files.")
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()

    expected = set(map(str, expected_records)) if expected_records is not None else None
    expected_counts = (
        {str(k): int(v) for k, v in expected_record_counts.items()}
        if expected_record_counts is not None else None
    )
    if expected_counts is not None and expected is not None:
        if set(expected_counts) != expected:
            raise ValueError("Notebook 01 expected count and record IDs differ.")
    closed: set[str] = set()
    active: str | None = None
    pending: list[pd.DataFrame] = []
    summaries: list[dict] = []
    example_trace: pd.DataFrame | None = None
    output_started = False

    def finish_record() -> None:
        nonlocal output_started, example_trace
        if not pending:
            return
        record = pd.concat(pending, ignore_index=True, sort=False)
        require_columns(
            record, ("record_id", "participant_id", "task",
                     "sample_index", "n", "x", "y", "val")
        )
        if record["sample_index"].duplicated().any():
            raise ValueError(f"Duplicate sample indices in record {active}.")
        if expected_counts is not None and len(record) != expected_counts.get(
            str(active), -1
        ):
            raise ValueError(
                f"Notebook 01 source row count mismatch for {active}: "
                f"expected {expected_counts.get(str(active))}, observed {len(record)}."
            )
        prepared = preprocess_record(
            record, smoothing_samples=smoothing_samples,
            max_gap_ms=max_gap_ms,
        )
        prepared.to_csv(
            temporary, mode="a", header=not output_started, index=False
        )
        output_started = True
        if example_trace is None:
            example_trace = prepared.head(3000).copy()
        summaries.append({
            "record_id": str(active),
            "participant_id": str(prepared["participant_id"].iloc[0]).zfill(3),
            "task": str(prepared["task"].iloc[0]),
            "n_samples": int(len(prepared)),
            "n_valid": int(prepared["valid_for_motion"].sum()),
            "n_segments": int(prepared.loc[
                prepared["valid_for_motion"], "segment_id"
            ].nunique()),
            "median_velocity_dva_s": float(prepared["velocity_dva_s"].median()),
            "max_velocity_dva_s": float(prepared["velocity_dva_s"].max()),
        })
        closed.add(str(active))
        pending.clear()

    try:
        for chunk in pd.read_csv(
            source, chunksize=chunksize, low_memory=False,
            dtype={"record_id": str, "participant_id": str},
        ):
            require_columns(chunk, ("record_id", "sample_index", "n", "x", "y", "val"))
            # Split contiguous runs even when an input chunk spans a record end.
            groups = chunk["record_id"].ne(chunk["record_id"].shift()).cumsum()
            for _, run in chunk.groupby(groups, sort=False):
                name = str(run["record_id"].iloc[0])
                if active is None:
                    active = name
                elif name != active:
                    finish_record()
                    if name in closed:
                        raise ValueError(
                            f"Source record {name} is non-contiguous; "
                            "regenerate Notebook 01 current outputs."
                        )
                    active = name
                pending.append(run)
        finish_record()
        if not summaries or example_trace is None:
            raise ValueError("No source records were processed.")
        if expected is not None and closed != expected:
            missing = sorted(expected - closed)
            extra = sorted(closed - expected)
            raise ValueError(
                f"Notebook 01 manifest mismatch; missing={missing}, extra={extra}."
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    result = pd.DataFrame(summaries).sort_values("record_id").reset_index(drop=True)
    return result, example_trace
