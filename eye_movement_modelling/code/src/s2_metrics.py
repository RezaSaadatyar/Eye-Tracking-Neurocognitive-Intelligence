# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Comparable, record-level four-task metrics and source identity checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from s2_data import TASKS, CLASS_TO_INDEX


def require_split_consistency(features: pd.DataFrame,
                              partitions: pd.DataFrame) -> pd.DataFrame:
    """One source task record and one split for every source participant."""
    keys = ["record_id", "participant_id", "task"]
    for name, frame in (("features", features), ("partitions", partitions)):
        if not set(keys).issubset(frame.columns):
            raise ValueError(f"{name} lacks source identity columns")
        if frame.record_id.duplicated().any():
            raise ValueError(f"{name} has duplicate source records")
    if "split" not in partitions.columns:
        raise ValueError("Missing frozen participant split")
    merged = features[keys].merge(
        partitions[keys + ["split"]], on="record_id",
        validate="one_to_one", how="outer", suffixes=("_feature", "_split"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Feature and split record identity sets differ")
    for field in ("participant_id", "task"):
        if not merged[f"{field}_feature"].astype(str).eq(
                merged[f"{field}_split"].astype(str)).all():
            raise ValueError(f"Source {field} differs between inputs")
    if merged.groupby("participant_id_split")["split"].nunique().max() != 1:
        raise ValueError("Person appears in multiple partitions")
    return merged.drop(columns="_merge").sort_values("record_id")


def record_probabilities(
    record_id: np.ndarray, y_true: np.ndarray, probabilities: np.ndarray,
) -> pd.DataFrame:
    """Aggregate temporal windows by source record BEFORE evaluation."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(TASKS):
        raise ValueError("Expected four-class probability matrix")
    if len(record_id) != len(y_true) or len(y_true) != len(probabilities):
        raise ValueError("Window prediction arrays do not align")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities")
    if np.min(probabilities) < -1e-6 or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-4,
    ):
        raise ValueError("Invalid class-probability rows")
    rows = []
    for key in sorted(set(record_id.astype(str))):
        mask = record_id.astype(str) == key
        labels = np.unique(y_true[mask])
        if len(labels) != 1:
            raise ValueError(f"Conflicting task labels for source record {key}")
        mean_prob = probabilities[mask].mean(axis=0)
        row = {"record_id": key, "target_index": int(labels[0]),
               "n_windows": int(mask.sum())}
        row.update({f"p_{label}": float(mean_prob[i])
                    for i, label in enumerate(TASKS)})
        row["pred_index"] = int(mean_prob.argmax())
        rows.append(row)
    return pd.DataFrame(rows)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, *,
               model: str, partition: str, n_participants: int) -> dict:
    """Use fixed label indices for macro metrics even if a fold misses a class."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
    )
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if not len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Model comparison needs aligned nonempty records")
    return {
        "model": model, "partition": partition, "n_records": len(y_true),
        "n_participants": int(n_participants),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=list(range(len(TASKS))),
            average="macro", zero_division=0,
        )),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def classical_record_predictions(model, features: pd.DataFrame, *,
                                 model_name: str, partition: str) -> pd.DataFrame:
    """Score records in the exact order supplied; never score using their IDs."""
    probabilities = model.predict_proba(features[list(FEATURE_COLUMN_NAMES)])
    if probabilities.shape[1] != len(TASKS):
        raise ValueError("Expected all four training task classes")
    table = features[["record_id", "participant_id", "task"]].copy()
    table["model"] = model_name
    table["partition"] = partition
    for i, label in enumerate(TASKS):
        table[f"p_{label}"] = probabilities[:, i]
    table["target_index"] = table.task.map(CLASS_TO_INDEX).astype(int)
    table["pred_index"] = probabilities.argmax(axis=1)
    return table


# Central feature list reused by every classical pipeline and final evaluation.
FEATURE_COLUMN_NAMES = (
    "valid_fraction", "x_sd_dva", "y_sd_dva", "step_median_dva",
    "step_p95_dva", "speed_median_dva_s", "speed_p95_dva_s",
    "pupil_coverage", "pupil_median_uncalibrated",
)


def n_subjects(partitions: pd.DataFrame, part: str) -> int:
    return int(partitions.loc[
        partitions.split.eq(part), "participant_id"
    ].astype(str).nunique())


def append_participant_identity(predictions: pd.DataFrame,
                                partitions: pd.DataFrame) -> pd.DataFrame:
    """Check predictions map to exactly their original source subject/task."""
    metadata = partitions[["record_id", "participant_id", "task", "split"]]
    merged = predictions.drop(
        columns=["participant_id", "task"], errors="ignore",
    ).merge(metadata, on="record_id", how="left", validate="many_to_one")
    if merged.split.isna().any():
        raise ValueError("Predicted unknown source record")
    return merged
