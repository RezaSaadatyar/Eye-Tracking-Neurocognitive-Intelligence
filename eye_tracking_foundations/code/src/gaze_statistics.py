# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Participant-balanced descriptive summaries and bounded bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def participant_summary(features: pd.DataFrame, columns: tuple[str,...]) -> pd.DataFrame:
    """First aggregate repeated recordings within participant and task."""
    return features.groupby(["task", "participant_id"], as_index=False)[list(columns)].mean()


def summarise_participant_metrics(
    participant_table: pd.DataFrame, columns: tuple[str,...],
    *, draws: int = 500, seed: int = 17,
) -> pd.DataFrame:
    """Percentile bootstrap over people, not eye samples or repeated trials."""
    if draws < 100:
        raise ValueError("Bootstrap draws must be >=100 to yield useful percentiles")
    rng = np.random.default_rng(seed)
    rows=[]
    for task, group in participant_table.groupby("task", sort=True):
        for feature in columns:
            vals = group[feature].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 1:
                means = rng.choice(vals, size=(draws, vals.size), replace=True).mean(axis=1)
                low, high = np.percentile(means, [2.5,97.5])
            else:
                low, high = np.nan, np.nan
            rows.append({
                "task":task,"feature":feature,"n_independent_participants":len(vals),
                "participant_mean":float(vals.mean()) if vals.size else np.nan,
                "participant_sd":float(vals.std(ddof=1)) if vals.size>1 else np.nan,
                "bootstrap_ci_lower":float(low),"bootstrap_ci_upper":float(high),
                "interval_status":"exploratory_cluster_bootstrap" if vals.size>1 else "insufficient_people",
                "bootstrap_draws":draws if vals.size>1 else 0,
            })
    return pd.DataFrame(rows)
