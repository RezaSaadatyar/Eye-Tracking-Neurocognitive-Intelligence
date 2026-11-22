"""BIDS-derived EEG/gaze alignment and conservative window eligibility.

Co-registered metadata do not establish drift-free independent clocks. These
routines use the BIDS StartTime and sampling frequencies, report coverage and
keep an explicit alignment limitation in the resulting quality table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def eye_sample_times(eye_meta: dict, count: int) -> np.ndarray:
    freq = float(eye_meta["SamplingFrequency"])
    start = float(eye_meta["StartTime"])
    if not np.isfinite(freq) or freq <= 0:
        raise ValueError("Invalid eye sample rate")
    return start + np.arange(count, dtype=float) / freq


def temporal_overlap(eeg_sfreq: float, eeg_samples: int,
                     eye_meta: dict, eye_samples: int) -> tuple[float, float]:
    if eeg_samples <= 0 or eye_samples <= 0 or eeg_sfreq <= 0:
        raise ValueError("Empty or invalid EEG/eye streams")
    et = eye_sample_times(eye_meta, eye_samples)
    lower = max(0., float(et[0]))
    upper = min((eeg_samples - 1) / eeg_sfreq, float(et[-1]))
    if upper <= lower:
        raise ValueError("No timebase overlap in BIDS metadata")
    return lower, upper


def valid_eye_rows(eye: pd.DataFrame) -> np.ndarray:
    required = ("eye1_x_coordinate", "eye1_y_coordinate")
    if not set(required).issubset(eye.columns):
        raise ValueError("Missing named eye coordinates")
    xy = eye[list(required)].apply(pd.to_numeric, errors="coerce")
    return np.isfinite(xy.to_numpy()).all(axis=1)


def window_registry(run: dict, raw, eye: pd.DataFrame,
                    eye_meta: dict, events: pd.DataFrame,
                    window_seconds: float = 12., max_windows: int = 20,
                    min_eye_fraction: float = 0.6) -> tuple[pd.DataFrame, dict]:
    """Find non-overlapping paired windows in the metadata-common timebase."""
    if window_seconds <= 0 or max_windows < 1:
        raise ValueError("Invalid window configuration")
    if not (0 <= min_eye_fraction <= 1):
        raise ValueError("min_eye_fraction must be between 0 and 1")
    start, stop = temporal_overlap(raw.info["sfreq"], raw.n_times,
                                   eye_meta, len(eye))
    eye_t = eye_sample_times(eye_meta, len(eye))
    eye_valid = valid_eye_rows(eye)
    event_t = pd.to_numeric(events["onset"], errors="coerce").to_numpy()
    if (np.isfinite(event_t).sum() == 0 or
            np.nanmin(event_t) < -1 or
            np.nanmax(event_t) > max(stop, raw.times[-1]) + 1):
        raise ValueError("Event onsets are inconsistent with recording time")
    bins = []
    first = np.ceil(start / window_seconds) * window_seconds
    for index in range(max_windows):
        t0 = float(first + index * window_seconds)
        t1 = t0 + window_seconds
        if t1 > stop:
            break
        mask = (eye_t >= t0) & (eye_t < t1)
        frac = float(eye_valid[mask].mean()) if mask.any() else 0.
        bins.append({"window_id": f"{run['participant_id']}_{run['task']}_"
                                   f"r{int(run['run'])}_w{index:03d}",
                     "participant_id": run["participant_id"],
                     "task": run["task"], "run": int(run["run"]),
                     "split": run["split"],
                     "start_seconds": t0, "end_seconds": t1,
                     "eye_valid_fraction": frac,
                     "eligible": frac >= min_eye_fraction})
    report = {"participant_id": run["participant_id"],
              "task": run["task"], "run": int(run["run"]),
              "eeg_sfreq_hz": float(raw.info["sfreq"]),
              "eye_sfreq_hz": float(eye_meta["SamplingFrequency"]),
              "eye_start_seconds": float(eye_meta["StartTime"]),
              "eeg_seconds": float(raw.times[-1]),
              "eye_seconds": float(eye_t[-1] - eye_t[0]),
              "common_start_seconds": start,
              "common_end_seconds": stop,
              "eligible_windows": int(sum(row["eligible"] for row in bins)),
              "total_windows": len(bins),
              "drift_verified": False,
              "alignment_basis": "BIDS StartTime and nominal sample rates"}
    return pd.DataFrame(bins), report


def ensure_disjoint_registry(table: pd.DataFrame) -> None:
    """Check a paired registry without imposing fixed participant counts."""
    if table.empty or table.window_id.duplicated().any():
        raise ValueError("Missing or duplicated window IDs")
    if not table.task.isin(("nr", "tsr")).all():
        raise ValueError("Unexpected task labels")
    by_person = table.groupby("participant_id")["split"].nunique()
    if not by_person.eq(1).all():
        raise ValueError("Participant appears in multiple split roles")
    if table.groupby("participant_id")["task"].nunique().min() != 2:
        raise ValueError("At least one participant lacks a paired task")
    if not table.eligible.any():
        raise ValueError("No eligible aligned windows")
