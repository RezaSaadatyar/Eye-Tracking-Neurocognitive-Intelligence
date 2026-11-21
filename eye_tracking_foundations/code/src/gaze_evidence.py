# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Describe conditional reference support without tuning I-VT thresholds."""
from __future__ import annotations
import numpy as np
import pandas as pd


def threshold_reference_support(
    sensitivity: pd.DataFrame,
    *,
    expected_thresholds: tuple[float, ...] = (20.0, 30.0, 40.0),
) -> pd.DataFrame:
    """Show changing eligible denominators beside conditional agreement."""
    required = (
        "record_id", "threshold_dva_s", "eligible_samples",
        "vendor_coverage_fraction", "predicted_evaluable_fraction",
        "sample_agreement", "predicted_fixation_fraction",
        "predicted_saccade_fraction", "unknown_fraction",
    )
    missing = sorted(set(required) - set(sensitivity.columns))
    if missing:
        raise ValueError(f"Notebook 06 is missing columns: {missing}")
    if sensitivity.empty or sensitivity.record_id.isna().any():
        raise ValueError("Missing Notebook 06 record identifiers")
    if sensitivity.duplicated(["record_id", "threshold_dva_s"]).any():
        raise ValueError("Duplicated record/threshold rows")
    grid = sorted(map(float, sensitivity.threshold_dva_s.unique()))
    if grid != sorted(map(float, expected_thresholds)):
        raise ValueError(f"Unexpected threshold grid: {grid}")
    records = set(sensitivity.record_id.astype(str))
    for threshold, group in sensitivity.groupby("threshold_dva_s"):
        if set(group.record_id.astype(str)) != records:
            raise ValueError(f"Different source records at threshold {threshold}")
    for _, group in sensitivity.groupby("record_id"):
        if not np.allclose(
            group.vendor_coverage_fraction.to_numpy(dtype=float),
            float(group.vendor_coverage_fraction.iloc[0]),
            rtol=0, atol=1e-9,
        ):
            raise ValueError("Vendor coverage changed across thresholds")
    if (sensitivity.eligible_samples < 0).any():
        raise ValueError("Negative vendor eligible count")
    if ((sensitivity.eligible_samples > 0)
            != sensitivity.sample_agreement.notna()).any():
        raise ValueError("Eligible counts and agreement availability differ")
    for field in (
        "vendor_coverage_fraction", "predicted_evaluable_fraction",
        "predicted_fixation_fraction", "predicted_saccade_fraction",
        "unknown_fraction",
    ):
        if not sensitivity[field].between(0, 1).all():
            raise ValueError(f"Invalid fraction: {field}")
    if not sensitivity.sample_agreement.dropna().between(0, 1).all():
        raise ValueError("Conditional agreement outside [0, 1]")
    result = []
    for threshold, group in sensitivity.groupby("threshold_dva_s", sort=True):
        weights = group.eligible_samples.to_numpy(float)
        scores = group.sample_agreement.to_numpy(float)
        valid = (weights > 0) & np.isfinite(scores)
        result.append({
            "threshold_dva_s": float(threshold),
            "n_records": int(len(group)),
            "n_vendor_evaluable_records": int(valid.sum()),
            "total_vendor_evaluable_samples": int(weights[valid].sum()),
            "mean_vendor_coverage_fraction": float(
                group.vendor_coverage_fraction.mean()
            ),
            "mean_predicted_evaluable_fraction": float(
                group.predicted_evaluable_fraction.mean()
            ),
            "mean_predicted_fixation_fraction": float(
                group.predicted_fixation_fraction.mean()
            ),
            "mean_predicted_saccade_fraction": float(
                group.predicted_saccade_fraction.mean()
            ),
            "mean_unknown_fraction": float(group.unknown_fraction.mean()),
            "mean_conditional_vendor_agreement": float(
                group.sample_agreement.mean()
            ),
            "eligible_sample_weighted_agreement": (
                float(np.average(scores[valid], weights=weights[valid]))
                if valid.any() else np.nan
            ),
        })
    return pd.DataFrame(result)


def reconcile_ivt_baseline(
    sensitivity: pd.DataFrame,
    notebook03_vendor: pd.DataFrame,
    *,
    baseline_dva_s: float = 30.0,
    tolerance: float = 1e-7,
) -> pd.DataFrame:
    """Require saved Notebook 03 and Notebook 06 baseline metrics to match."""
    columns = (
        "record_id", "eligible_samples", "vendor_coverage_fraction",
        "predicted_evaluable_fraction", "sample_agreement",
    )
    for table in (sensitivity, notebook03_vendor):
        if not set(columns).issubset(table.columns):
            raise ValueError("Vendor comparison columns are missing")
    baseline = sensitivity.loc[
        np.isclose(sensitivity.threshold_dva_s, baseline_dva_s)
    ]
    if baseline.empty:
        raise ValueError("The 30-dva/s I-VT baseline is missing")
    if baseline.record_id.duplicated().any() or notebook03_vendor.record_id.duplicated().any():
        raise ValueError("Duplicate vendor comparison record IDs")
    if set(baseline.record_id) != set(notebook03_vendor.record_id):
        raise ValueError("Notebook 03/06 vendor record sets differ")
    joined = baseline[list(columns)].merge(
        notebook03_vendor[list(columns)], on="record_id",
        validate="one_to_one", suffixes=("_06", "_03"),
    )
    for key in columns[1:]:
        a = joined[f"{key}_06"].to_numpy(float)
        b = joined[f"{key}_03"].to_numpy(float)
        if not np.isclose(a, b, rtol=0, atol=tolerance, equal_nan=True).all():
            raise ValueError(f"Notebook 03/06 baseline mismatch: {key}")
    return joined.sort_values("record_id").reset_index(drop=True)
