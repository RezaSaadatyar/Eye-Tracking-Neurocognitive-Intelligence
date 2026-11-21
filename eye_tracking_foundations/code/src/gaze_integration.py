# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Reconcile compact Notebook 01–06 evidence without rerunning gaze detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gaze_statistics import validate_record_quality_alignment
from gaze_evidence import threshold_reference_support, reconcile_ivt_baseline


def reconcile_saved_record_evidence(
    quality: pd.DataFrame,
    preprocessing: pd.DataFrame,
    vendor: pd.DataFrame,
    features: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Align upstream records, original sample counts, and the I-VT baseline."""
    alignment = validate_record_quality_alignment(features, quality)
    required = {"record_id", "participant_id", "task", "n_samples"}
    if not required.issubset(preprocessing.columns) or "n_samples" not in quality:
        raise ValueError("Notebook 01/02 sample-count columns are missing")
    if preprocessing.record_id.duplicated().any():
        raise ValueError("Duplicate Notebook 02 record identity")
    original = quality[
        ["record_id", "participant_id", "task", "n_samples"]
    ].copy()
    recent = preprocessing[
        ["record_id", "participant_id", "task", "n_samples"]
    ].copy()
    for table in (original, recent):
        table["record_id"] = table["record_id"].astype(str)
        table["participant_id"] = table["participant_id"].astype(str).str.zfill(3)
        table["task"] = table["task"].astype(str)
    counts = original.merge(
        recent, on="record_id", how="outer", validate="one_to_one",
        indicator=True, suffixes=("_01", "_02"),
    )
    if not counts._merge.eq("both").all():
        raise ValueError("Notebook 01/02 record sets differ")
    for name in ("participant_id", "task", "n_samples"):
        if not counts[f"{name}_01"].eq(counts[f"{name}_02"]).all():
            raise ValueError(f"Notebook 01/02 {name} values differ")
    if set(alignment.record_id) != set(counts.record_id):
        raise ValueError("Notebook 01/02/04 source identities differ")
    if set(vendor.record_id.astype(str)) != set(counts.record_id):
        raise ValueError("Notebook 03 source record identities differ")
    # These checks also enforce complete, consistent record sets per threshold.
    threshold_reference_support(sensitivity)
    if set(sensitivity.record_id.astype(str)) != set(counts.record_id):
        raise ValueError("Notebook 06 source record identities differ")
    baseline = reconcile_ivt_baseline(sensitivity, vendor)
    if len(baseline) != len(counts):
        raise ValueError("Baseline reference support does not cover source")
    merged = alignment.merge(
        counts[["record_id", "n_samples_01"]],
        on="record_id", validate="one_to_one",
    )
    return merged.sort_values("record_id").reset_index(drop=True)


def validate_participant_statistics(
    features: pd.DataFrame,
    participants: pd.DataFrame,
    statistics: pd.DataFrame,
    *,
    tolerance: float = 1e-7,
) -> int:
    """Check independent participant support and means, not bootstrap redrawing."""
    keys = ("task", "participant_id")
    for name, frame in (("features", features), ("participants", participants)):
        if not set(keys).issubset(frame.columns):
            raise ValueError(f"Missing participant/task identity in {name}")
    if participants.duplicated(list(keys)).any():
        raise ValueError("Duplicated participant/task summary identity")
    source_keys = set(map(tuple, features[list(keys)].astype(str).to_numpy()))
    output_keys = set(map(tuple, participants[list(keys)].astype(str).to_numpy()))
    if source_keys != output_keys:
        raise ValueError("Notebook 04/05 participant/task combinations differ")
    fields = ("task", "feature", "n_independent_participants", "participant_mean")
    if not set(fields).issubset(statistics.columns):
        raise ValueError("Missing Notebook 05 statistics fields")
    if statistics.duplicated(["task", "feature"]).any():
        raise ValueError("Duplicate Notebook 05 task-feature summaries")
    for row in statistics.itertuples(index=False):
        feature = row.feature
        if feature not in participants.columns:
            raise ValueError(f"Unknown Notebook 05 feature: {feature}")
        values = participants.loc[
            participants["task"].eq(row.task), feature
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) != int(row.n_independent_participants):
            raise ValueError("Notebook 05 independent participant support differs")
        expected = float(values.mean()) if len(values) else np.nan
        if not np.isclose(
            expected, float(row.participant_mean),
            atol=tolerance, rtol=0, equal_nan=True,
        ):
            raise ValueError("Notebook 05 participant mean differs")
    return int(participants.participant_id.astype(str).nunique())


def validate_threshold_summaries(
    sensitivity: pd.DataFrame,
    summary: pd.DataFrame,
    saved_support: pd.DataFrame,
    *,
    tolerance: float = 1e-7,
) -> pd.DataFrame:
    """Require compact Notebook 06 tables to match detailed source evidence."""
    rebuilt = threshold_reference_support(sensitivity)
    if summary.threshold_dva_s.duplicated().any() or saved_support.threshold_dva_s.duplicated().any():
        raise ValueError("Duplicated threshold summary entries")
    for name, frame, fields in (
        ("summary", summary, (
            "n_records", "mean_predicted_fixation_fraction",
            "mean_unknown_fraction", "mean_conditional_vendor_agreement",
        )),
        ("reference support", saved_support, (
            "n_records", "n_vendor_evaluable_records",
            "total_vendor_evaluable_samples",
            "mean_predicted_evaluable_fraction",
            "mean_conditional_vendor_agreement",
        )),
    ):
        aligned = rebuilt.merge(
            frame, on="threshold_dva_s", how="outer",
            validate="one_to_one", suffixes=("_computed", "_saved"),
            indicator=True,
        )
        if not aligned._merge.eq("both").all():
            raise ValueError(f"Notebook 06 {name} thresholds differ")
        for field in fields:
            a = aligned[f"{field}_computed"].to_numpy(dtype=float)
            b = aligned[f"{field}_saved"].to_numpy(dtype=float)
            if not np.isclose(a, b, atol=tolerance, rtol=0, equal_nan=True).all():
                raise ValueError(f"Notebook 06 {name} differs: {field}")
    return rebuilt
