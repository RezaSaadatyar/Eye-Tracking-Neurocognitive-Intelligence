# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Participant-balanced descriptive summaries and bounded bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd



def validate_record_quality_alignment(
    features: pd.DataFrame, quality: pd.DataFrame, *, tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Reconcile source-record keys and valid fractions across Notebooks 01/04.

    This is a consistency check, not a retrospective quality exclusion. A record
    must occur exactly once in each table; source identity, task and participant
    must match; and their recorded valid fractions must agree numerically.
    """
    required = ("record_id", "participant_id", "task", "valid_fraction")
    for name, table in (("Notebook 04 features", features),
                        ("Notebook 01 quality", quality)):
        missing = sorted(set(required) - set(table.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if table.empty or table["record_id"].isna().any():
            raise ValueError(f"{name} has no complete source-record identities")
        if table["record_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate record identifiers")

    left = features[list(required)].copy()
    right = quality[list(required)].copy()
    for table in (left, right):
        table["record_id"] = table["record_id"].astype(str)
        table["participant_id"] = table["participant_id"].astype(str).str.zfill(3)
        table["task"] = table["task"].astype(str)
        table["valid_fraction"] = pd.to_numeric(
            table["valid_fraction"], errors="raise"
        )
    merged = left.merge(
        right, on="record_id", how="outer", validate="one_to_one",
        suffixes=("_notebook04", "_notebook01"), indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[~merged["_merge"].eq("both"), "record_id"].tolist()
        raise ValueError(f"Notebook 01/04 record sets do not agree: {missing[:5]}")
    for field in ("participant_id", "task"):
        if not merged[f"{field}_notebook04"].eq(
            merged[f"{field}_notebook01"]
        ).all():
            raise ValueError(f"Notebook 01/04 {field} identity mismatch")
    a = merged["valid_fraction_notebook04"].to_numpy(dtype=float)
    z = merged["valid_fraction_notebook01"].to_numpy(dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(z).all()):
        raise ValueError("Non-finite valid fractions in Notebook 01/04 inputs")
    if not (np.logical_and(a >= 0, a <= 1).all()
            and np.logical_and(z >= 0, z <= 1).all()):
        raise ValueError("Valid fractions outside [0, 1]")
    if not np.allclose(a, z, rtol=0, atol=tolerance):
        raise ValueError("Notebook 01/04 valid fractions differ for same record")
    merged["absolute_valid_fraction_difference"] = np.abs(a - z)
    return merged.drop(columns="_merge").sort_values(
        "record_id"
    ).reset_index(drop=True)


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
