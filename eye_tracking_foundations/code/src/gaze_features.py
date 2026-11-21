# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Summarise I-VT events and visually inspect scanpath-like fixation paths."""
from __future__ import annotations

import numpy as np
import pandas as pd


def record_features(record: pd.DataFrame, event_table: pd.DataFrame) -> dict:
    """Use event-level quantities; preserve missing estimates as missing."""
    events = event_table[event_table.method.eq("IVT")]
    fixation = events[events.event_type.eq("fixation")].sort_values("start_sample")
    saccades = events[events.event_type.eq("saccade")]
    pupil = pd.to_numeric(record.get("dP", pd.Series(np.nan, index=record.index)), errors="coerce")
    good = record["valid_for_motion"].astype(bool)
    pupil_good = pupil[good & pupil.gt(0) & np.isfinite(pupil)]
    points = fixation[["mean_x_dva", "mean_y_dva"]].to_numpy(float)
    scanpath = float(np.hypot(*(np.diff(points, axis=0).T)).sum()) if len(points) >= 2 else np.nan
    return {
        "record_id": record.record_id.iloc[0],
        "participant_id": str(record.participant_id.iloc[0]).zfill(3),
        "task": record.task.iloc[0],
        "valid_fraction": float(good.mean()),
        "n_fixations_ivt": len(fixation),
        "n_saccades_ivt": len(saccades),
        "mean_fixation_ms": float(fixation.duration_ms.mean()),
        "median_fixation_ms": float(fixation.duration_ms.median()),
        "mean_saccade_peak_dva_s": float(saccades.peak_speed_dva_s.mean()),
        "scanpath_length_dva": scanpath,
        "pupil_median_uncalibrated": float(pupil_good.median()) if len(pupil_good) else np.nan,
    }


FEATURE_DESCRIPTIONS = [
    ("valid_fraction", "Proportion of samples eligible for motion analysis", "fraction"),
    ("n_fixations_ivt", "Number of I-VT fixation events", "events"),
    ("n_saccades_ivt", "Number of I-VT saccade events", "events"),
    ("mean_fixation_ms", "Mean detected fixation duration", "ms"),
    ("median_fixation_ms", "Median detected fixation duration", "ms"),
    ("mean_saccade_peak_dva_s", "Mean peak I-VT saccade speed", "dva/s"),
    ("scanpath_length_dva", "Distance between time-ordered detected fixation centroids", "dva"),
    ("pupil_median_uncalibrated", "Median valid uncalibrated pupil area", "arbitrary units"),
]


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_DESCRIPTIONS,columns=["feature", "definition", "unit"])
