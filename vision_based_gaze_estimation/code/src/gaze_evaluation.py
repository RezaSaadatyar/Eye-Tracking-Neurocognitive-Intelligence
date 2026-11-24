# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Participant-disjoint folds, sample alignment and angular-error summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gaze_geometry import angular_error_degrees


# ========================= 01. Outer/Inner Protocol ==========================
def participant_fold_table(participants, *, seed=17, inner_people=2):
    """Record a deterministic nested participant-held-out LOPO design.

    For each outer fold, the held-out person is excluded from every fitting
    and selection decision. Inner Validation uses different people from
    inner Training. Source photographs are never split independently.
    """
    subjects = sorted(set(str(p) for p in participants))
    if len(subjects) < inner_people + 2:
        raise ValueError(
            "Not enough independent participants for the requested inner split")
    rows = []
    for outer, test_person in enumerate(subjects):
        others = [p for p in subjects if p != test_person]
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, outer]))
        validation = set(rng.choice(others, size=inner_people, replace=False))
        for person in subjects:
            role = ("test" if person == test_person else
                    "validation" if person in validation else "train")
            rows.append({
                "fold_id": f"outer_{outer + 1:02d}",
                "outer_test_person": test_person,
                "participant_id": person,
                "role": role,
            })
    result = pd.DataFrame(rows)
    if (result.duplicated(["fold_id", "participant_id"]).any()
            or not result.groupby("fold_id").role.apply(
                lambda v: set(v) == {"train", "validation", "test"}
            ).all()
            or not result.groupby("fold_id").apply(
                lambda g: (g.role.eq("test").sum() == 1
                           and g.role.eq("validation").sum() == inner_people)
            ).all()):
        raise ValueError("Participant role assignments are inconsistent")
    return result


# ========================= 02. Current-File Alignment ========================
def load_image_cache(index_path: Path, cache_path: Path):
    """Load indexed images from a local-only, non-published cache."""
    index = pd.read_csv(index_path, dtype={
        "sample_id": str, "participant_id": str, "source_mat": str})
    if index.empty or index.sample_id.duplicated().any():
        raise ValueError("Missing or repeated eye sample IDs")
    if not Path(cache_path).is_file():
        raise FileNotFoundError(
            f"Notebook 02's local-only image cache is missing: {cache_path}. "
            "Re-run Notebook 02 from original source data; Git does not store images.")
    with np.load(cache_path, allow_pickle=False) as stored:
        ids = stored["sample_id"].astype(str)
        images = stored["images"]
        targets = stored["gaze"]
    if (len(index) != len(ids) or len(index) != len(images)
            or not np.array_equal(index.sample_id.to_numpy(), ids)
            or targets.shape != (len(index), 3)
            or images.ndim != 3 or images.shape[0] != len(index)
            or not np.isfinite(targets).all()):
        raise ValueError("Stage-02 image/label/source identity mismatch")
    index_gaze = index[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(float)
    if not np.allclose(index_gaze, targets, atol=1e-6):
        raise ValueError("Image cache targets differ from the Stage-02 index")
    return index, images, targets


def outer_fold_indices(table, folds, fold_id):
    """Map the per-person frozen fold role table onto sample rows."""
    roles = folds.loc[folds.fold_id.eq(fold_id)]
    if roles.empty or roles.participant_id.duplicated().any():
        raise ValueError(f"Missing or duplicate participant roles: {fold_id}")
    assignment = table.participant_id.map(
        roles.set_index("participant_id").role)
    if assignment.isna().any():
        raise ValueError(f"Unassigned participant samples: {fold_id}")
    i = {role: np.flatnonzero(assignment.eq(role).to_numpy())
         for role in ("train", "validation", "test")}
    persons = [set(table.iloc[values].participant_id) for values in i.values()]
    if not all(len(v) for v in i.values()) or any(
            persons[k] & persons[j] for k in range(3) for j in range(k + 1, 3)):
        raise ValueError("Empty or overlapping participant role in outer fold")
    return i


# ======================= 03. Correct Metric Aggregation ======================
def person_angular_metrics(predictions):
    """Summarise independent-person errors without treating rows as people."""
    needed = {"sample_id", "model", "fold_id", "participant_id",
              "target_x", "target_y", "target_z",
              "pred_x", "pred_y", "pred_z"}
    if needed.difference(predictions.columns):
        raise ValueError(f"Missing prediction columns: {sorted(needed - set(predictions))}")
    if predictions.duplicated(["model", "sample_id"]).any():
        raise ValueError("Repeated model/sample prediction rows")
    if predictions.empty:
        raise ValueError("No out-of-person predictions were recorded")
    observed = predictions.copy()
    truth = observed[["target_x", "target_y", "target_z"]].to_numpy(float)
    guess = observed[["pred_x", "pred_y", "pred_z"]].to_numpy(float)
    observed["angular_error_deg"] = angular_error_degrees(truth, guess)
    summary = (observed.groupby(["model", "fold_id", "participant_id"])
               .angular_error_deg.agg(
                   n_samples="count", mean_angular_error_deg="mean",
                   median_angular_error_deg="median")
               .reset_index())
    return observed, summary
