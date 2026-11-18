# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Summarise timebase, missingness and record-level gaze sample quality."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gaze_source import require_columns


def summarise_quality(samples: pd.DataFrame, *, gap_multiplier: float = 3.0) -> pd.DataFrame:
    """Keep all records, including poor quality; report observed properties."""
    require_columns(samples, ("record_id", "participant_id", "task", "n", "val", "x", "y"))
    results = []
    for record_id, df in samples.groupby("record_id", sort=True):
        df = df.sort_values("sample_index")
        t = df["n"].to_numpy(dtype="float64")
        dt = np.diff(t)
        positive = dt[np.isfinite(dt) & (dt > 0)]
        median_dt = float(np.median(positive)) if positive.size else float("nan")
        gap_limit = gap_multiplier * median_dt if np.isfinite(median_dt) else float("nan")
        valid = df["val"].eq(0) & np.isfinite(df["x"]) & np.isfinite(df["y"])
        pupil = pd.to_numeric(df.get("dP", pd.Series(np.nan, index=df.index)), errors="coerce")
        labs = pd.to_numeric(df.get("lab", pd.Series(np.nan, index=df.index)), errors="coerce")
        results.append({
            "record_id": record_id,
            "participant_id": str(df["participant_id"].iloc[0]).zfill(3),
            "round_id": int(df["round_id"].iloc[0]),
            "session_id": int(df["session_id"].iloc[0]),
            "task": df["task"].iloc[0],
            "n_samples": len(df),
            "duration_s": float((t[-1] - t[0]) / 1000) if len(t) else np.nan,
            "n_valid": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "invalid_fraction": float((~valid).mean()),
            "n_duplicate_or_reversed_steps": int((dt <= 0).sum()),
            "median_interval_ms": median_dt,
            "observed_rate_hz": 1000 / median_dt if median_dt > 0 else np.nan,
            "n_large_gaps": int((dt > gap_limit).sum()) if np.isfinite(gap_limit) else np.nan,
            "pupil_finite_fraction": float(np.isfinite(pupil).mean()),
            "vendor_label_fraction": float(np.isfinite(labs).mean()),
        })
    return pd.DataFrame(results).sort_values("record_id").reset_index(drop=True)
