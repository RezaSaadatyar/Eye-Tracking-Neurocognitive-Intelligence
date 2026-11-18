# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Educational I-VT/I-DT detectors with honest unclassified states."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gaze_source import require_columns

VENDOR_LABELS = {-1: "blink", 1: "fixation", 2: "saccade"}


def _duration_ms(times: np.ndarray) -> float:
    if times.size == 0:
        return float("nan")
    increments = np.diff(times)
    step = float(np.median(increments[increments > 0])) if np.any(increments > 0) else 0.0
    return float(times[-1] - times[0] + step)


def contiguous_runs(labels: np.ndarray):
    if len(labels) == 0:
        return
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        yield int(a), int(b), str(labels[a])


def ivt_labels(
    record: pd.DataFrame, *, threshold_dva_s: float = 30.0,
    min_fixation_ms: float = 80.0, min_saccade_ms: float = 6.0,
) -> np.ndarray:
    """Classify valid velocity samples, reject implausibly short runs."""
    require_columns(record, ("n", "valid_for_motion", "velocity_dva_s", "segment_id"))
    if threshold_dva_s <= 0:
        raise ValueError("threshold_dva_s must be positive")
    speed = record["velocity_dva_s"].to_numpy(float)
    eligible = record["valid_for_motion"].astype(bool).to_numpy() & np.isfinite(speed)
    labels = np.full(len(record), "unknown", dtype="<U12")
    labels[eligible & (speed < threshold_dva_s)] = "fixation"
    labels[eligible & (speed >= threshold_dva_s)] = "saccade"
    time = record["n"].to_numpy(float)
    segments = record["segment_id"].to_numpy()
    # Guard against merging same labels across disconnected valid intervals.
    for segment in np.unique(segments):
        inds = np.flatnonzero(segments == segment)
        for a, b, label in contiguous_runs(labels[inds]):
            if label == "unknown":
                continue
            duration = _duration_ms(time[inds[a:b]])
            required = min_fixation_ms if label == "fixation" else min_saccade_ms
            if duration < required:
                labels[inds[a:b]] = "unknown"
    return labels


def idt_labels(record: pd.DataFrame, *, dispersion_dva: float = 1.0,
               min_fixation_ms: float = 80.0) -> np.ndarray:
    """I-DT fixation-only detector; non-fixations stay unclassified.

    Dispersion is horizontal plus vertical range in degrees of visual angle.
    It is not a saccade or blink classifier.
    """
    require_columns(record, ("n", "x_smooth", "y_smooth", "valid_for_motion", "segment_id"))
    if dispersion_dva <= 0 or min_fixation_ms <= 0:
        raise ValueError("dispersion_dva and min_fixation_ms must be positive")
    labels = np.full(len(record), "unknown", dtype="<U12")
    time = record["n"].to_numpy(float)
    x = record["x_smooth"].to_numpy(float)
    y = record["y_smooth"].to_numpy(float)
    valid = record["valid_for_motion"].astype(bool).to_numpy()
    segments = record["segment_id"].to_numpy()
    for segment in np.unique(segments):
        indices = np.flatnonzero((segments == segment) & valid)
        pos = 0
        while pos < len(indices):
            end = pos + 1
            while end < len(indices) and time[indices[end]] - time[indices[pos]] < min_fixation_ms:
                end += 1
            if end >= len(indices):
                break
            def disp(left, right):
                a = indices[left:right]
                return (float(np.ptp(x[a])) + float(np.ptp(y[a])))
            if disp(pos, end + 1) > dispersion_dva:
                pos += 1
                continue
            while end + 1 < len(indices) and disp(pos, end + 2) <= dispersion_dva:
                end += 1
            labels[indices[pos:end + 1]] = "fixation"
            pos = end + 1
    return labels


def extract_events(record: pd.DataFrame, labels: np.ndarray, method: str) -> pd.DataFrame:
    """Create unambiguous within-record event intervals and mean positions."""
    if len(record) != len(labels):
        raise ValueError("Each sample requires exactly one predicted label")
    out: list[dict] = []
    times = record["n"].to_numpy(float)
    segments = record["segment_id"].to_numpy()
    for segment in np.unique(segments):
        indices = np.flatnonzero(segments == segment)
        for a, b, label in contiguous_runs(labels[indices]):
            if label == "unknown":
                continue
            ids = indices[a:b]
            out.append({
                "record_id": record["record_id"].iloc[0],
                "participant_id": str(record["participant_id"].iloc[0]).zfill(3),
                "task": record["task"].iloc[0],
                "method": method,
                "event_type": label,
                "start_sample": int(record["sample_index"].iloc[ids[0]]),
                "end_sample": int(record["sample_index"].iloc[ids[-1]]),
                "start_ms": float(times[ids[0]]),
                "end_ms": float(times[ids[-1]]),
                "duration_ms": _duration_ms(times[ids]),
                "mean_x_dva": float(record["x_smooth"].iloc[ids].mean()),
                "mean_y_dva": float(record["y_smooth"].iloc[ids].mean()),
                "peak_speed_dva_s": float(record["velocity_dva_s"].iloc[ids].max()),
                "n_samples": int(len(ids)),
            })
    cols=["record_id","participant_id","task","method","event_type","start_sample", "end_sample", "start_ms", "end_ms","duration_ms","mean_x_dva","mean_y_dva","peak_speed_dva_s","n_samples"]
    return pd.DataFrame(out, columns=cols)


def label_agreement(record: pd.DataFrame, predicted: np.ndarray) -> dict:
    """Sample-level agreement with *vendor* labels; no gold-standard claim."""
    require_columns(record, ("lab",))
    vendor = pd.to_numeric(record["lab"], errors="coerce").map(VENDOR_LABELS).to_numpy()
    eligible = pd.notna(vendor) & np.isin(predicted, ["fixation", "saccade"])
    return {
        "eligible_samples": int(eligible.sum()),
        "vendor_coverage_fraction": float(pd.notna(vendor).mean()),
        "predicted_evaluable_fraction": float(eligible.mean()),
        "sample_agreement": float(np.mean(predicted[eligible] == vendor[eligible])) if eligible.any() else np.nan,
    }
