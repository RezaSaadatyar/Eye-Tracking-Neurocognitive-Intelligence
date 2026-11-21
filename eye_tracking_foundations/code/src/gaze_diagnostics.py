# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Inspect observed peak gaze velocities without changing preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gaze_source import require_columns


# ================== 01. Identify Peak-Velocity Samples =======================

def inspect_velocity_peaks(
    prepared_csv: str | Path,
    record_summary: pd.DataFrame,
    *,
    top_records: int = 3,
    half_window_samples: int = 60,
    chunksize: int = 80_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an evidence table and local sample windows for extreme speeds.

    Rank *record maxima*, inspect one peak in each selected record, and preserve
    its raw values and neighbouring samples. This is descriptive inspection,
    not outlier exclusion, physiological validation or threshold selection.
    The current Notebook 02 sample CSV is scanned twice in bounded chunks.
    """
    if top_records < 1 or half_window_samples < 1 or chunksize < 1:
        raise ValueError("top_records, half_window_samples and chunksize must be positive")
    require_columns(record_summary, ("record_id", "max_velocity_dva_s"))
    if record_summary["record_id"].duplicated().any():
        raise ValueError("Duplicate record identifiers in Notebook 02 summary")
    ranked = record_summary.dropna(subset=["max_velocity_dva_s"]).sort_values(
        ["max_velocity_dva_s", "record_id"], ascending=[False, True]
    )
    selected = ranked.head(top_records)["record_id"].astype(str).tolist()
    if not selected:
        raise ValueError("No finite velocity maxima to inspect")
    path = Path(prepared_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Notebook 02 prepared sample CSV missing: {path}")
    cols = (
        "record_id", "participant_id", "task", "sample_index", "n",
        "x", "y", "val", "x_smooth", "y_smooth", "valid_for_motion",
        "segment_id", "velocity_dva_s",
    )
    best: dict[str, tuple[float, int]] = {}
    for chunk in pd.read_csv(
        path, usecols=list(cols), chunksize=chunksize,
        dtype={"record_id": str, "participant_id": str},
    ):
        require_columns(chunk, cols)
        for rid, frame in chunk[chunk["record_id"].isin(selected)].groupby(
            "record_id", sort=False
        ):
            finite = frame.loc[np.isfinite(frame["velocity_dva_s"])]
            if finite.empty:
                continue
            row = finite.loc[finite["velocity_dva_s"].idxmax()]
            candidate = (float(row["velocity_dva_s"]), int(row["sample_index"]))
            if rid not in best or candidate[0] > best[rid][0]:
                best[rid] = candidate
    if set(best) != set(selected):
        raise ValueError("A selected record has no finite velocity in prepared CSV")
    for rid in selected:
        stated = float(ranked.set_index("record_id").loc[rid, "max_velocity_dva_s"])
        if not np.isclose(best[rid][0], stated, rtol=1e-6, atol=1e-3):
            raise ValueError(f"Velocity maximum differs from Notebook 02 summary: {rid}")

    windows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path, usecols=list(cols), chunksize=chunksize,
        dtype={"record_id": str, "participant_id": str},
    ):
        for rid in selected:
            peak_index = best[rid][1]
            subset = chunk.loc[
                chunk["record_id"].eq(rid)
                & chunk["sample_index"].between(
                    peak_index - half_window_samples,
                    peak_index + half_window_samples,
                )
            ]
            if not subset.empty:
                windows.append(subset.copy())
    if not windows:
        raise ValueError("No samples found around selected speed peaks")
    samples = pd.concat(windows, ignore_index=True).sort_values(
        ["record_id", "sample_index"], kind="stable"
    ).reset_index(drop=True)
    if samples.duplicated(["record_id", "sample_index"]).any():
        raise ValueError("Overlapping or repeated diagnostic source rows")

    evidence = []
    for rid in selected:
        view = samples.loc[samples["record_id"].eq(rid)].copy()
        view = view.sort_values("sample_index")
        peak_sample = best[rid][1]
        current_rows = view.loc[view["sample_index"].eq(peak_sample)]
        prior_rows = view.loc[view["sample_index"].eq(peak_sample - 1)]
        if len(current_rows) != 1 or len(prior_rows) != 1:
            raise ValueError(f"Missing adjacent samples for peak {rid}")
        current, previous = current_rows.iloc[0], prior_rows.iloc[0]
        dt_ms = float(current["n"] - previous["n"])
        raw_step = float(np.hypot(
            float(current["x"] - previous["x"]),
            float(current["y"] - previous["y"]),
        ))
        smooth_step = float(np.hypot(
            float(current["x_smooth"] - previous["x_smooth"]),
            float(current["y_smooth"] - previous["y_smooth"]),
        ))
        same_run = bool(current["segment_id"] == previous["segment_id"])
        if not (dt_ms > 0 and same_run):
            raise ValueError(f"Peak crosses a gap or invalid run: {rid}")
        rebuilt_speed = 1000.0 * smooth_step / dt_ms
        if not np.isclose(rebuilt_speed, best[rid][0], rtol=1e-6, atol=1e-3):
            raise ValueError(f"Peak velocity cannot be reconstructed: {rid}")
        evidence.append({
            "record_id": rid,
            "participant_id": str(current["participant_id"]).zfill(3),
            "task": str(current["task"]),
            "peak_sample_index": peak_sample,
            "peak_time_ms": float(current["n"]),
            "peak_velocity_dva_s": best[rid][0],
            "previous_val": int(previous["val"]),
            "peak_val": int(current["val"]),
            "previous_valid_for_motion": bool(previous["valid_for_motion"]),
            "peak_valid_for_motion": bool(current["valid_for_motion"]),
            "same_valid_run": same_run,
            "delta_t_ms": dt_ms,
            "raw_displacement_dva": raw_step,
            "smoothed_displacement_dva": smooth_step,
            "reconstructed_velocity_dva_s": rebuilt_speed,
            "window_samples": len(view),
        })
    result = pd.DataFrame(evidence)
    return result, samples
