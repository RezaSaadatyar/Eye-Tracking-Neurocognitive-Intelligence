# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Gap-respecting gaze smoothing and velocity estimation in dva/s."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gaze_source import require_columns


def preprocess_record(
    record: pd.DataFrame, *, smoothing_samples: int = 5,
    max_gap_ms: float = 5.0,
) -> pd.DataFrame:
    """Never smooth across invalid samples, time reversals or large gaps.

    Raw gaze remains unchanged; the smoothed representation is optional.
    A causal velocity is based on adjacent eligible samples and is undefined
    at each segment start. No interpolation or inferred blink repair occurs.
    """
    require_columns(record, ("record_id", "sample_index", "n", "x", "y", "val"))
    if smoothing_samples < 1 or smoothing_samples % 2 == 0:
        raise ValueError("smoothing_samples must be a positive odd integer")
    if max_gap_ms <= 0:
        raise ValueError("max_gap_ms must be positive")
    frame = record.sort_values("sample_index").copy()
    t = frame["n"].to_numpy(float)
    valid = (frame["val"].eq(0) & np.isfinite(frame["x"]) & np.isfinite(frame["y"])).to_numpy()
    delta = np.diff(t, prepend=np.nan)
    boundary = (~valid) | (~np.roll(valid, 1)) | ~(delta > 0) | (delta > max_gap_ms)
    boundary[0] = True
    frame["valid_for_motion"] = valid
    frame["segment_id"] = np.cumsum(boundary).astype("int64")
    frame["x_smooth"] = np.nan
    frame["y_smooth"] = np.nan
    frame["velocity_dva_s"] = np.nan
    for _, part in frame.loc[frame.valid_for_motion].groupby("segment_id", sort=False):
        idx = part.index
        for axis in ("x", "y"):
            frame.loc[idx, f"{axis}_smooth"] = part[axis].rolling(
                smoothing_samples, center=True, min_periods=1
            ).median().to_numpy()
        n = frame.loc[idx, "n"].to_numpy(float)
        x = frame.loc[idx, "x_smooth"].to_numpy(float)
        y = frame.loc[idx, "y_smooth"].to_numpy(float)
        if len(part) > 1:
            speed = np.hypot(np.diff(x), np.diff(y)) / np.diff(n) * 1000.0
            frame.loc[idx[1:], "velocity_dva_s"] = speed
    return frame


def preprocess_samples(samples: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Apply the same fixed method independently within each record."""
    return pd.concat(
        [preprocess_record(frame, **kwargs) for _, frame in samples.groupby("record_id",sort=True)],
        ignore_index=True, sort=False,
    )
