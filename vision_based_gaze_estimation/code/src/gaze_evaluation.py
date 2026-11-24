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
    counts = (result.groupby(["fold_id", "role"]).size()
              .unstack(fill_value=0))
    required_roles = {"train", "validation", "test"}
    if (result.duplicated(["fold_id", "participant_id"]).any()
            or set(counts.columns) != required_roles
            or not counts["test"].eq(1).all()
            or not counts["validation"].eq(inner_people).all()
            or not counts["train"].gt(0).all()):
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


# ======================= 04. Full Outer Prediction Contracts =================
def validate_lopo_design(index, folds):
    """Validate complete participant-disjoint roles before fitting or reporting."""
    if index.empty or index[["sample_id", "participant_id"]].isna().any().any():
        raise ValueError("Missing sample or participant identity")
    if index.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs")
    people = set(index.participant_id)
    columns = ["fold_id", "outer_test_person", "participant_id", "role"]
    if folds.empty or folds[columns].isna().any().any():
        raise ValueError("Missing fold identity or role")
    if folds.duplicated(["fold_id", "participant_id"]).any():
        raise ValueError("Duplicate participant roles")
    held_out = []
    for fold_id, group in folds.groupby("fold_id"):
        counts = group.role.value_counts().to_dict()
        if (set(group.participant_id) != people
                or counts != {"train": len(people) - 3, "validation": 2, "test": 1}
                or len(people) < 4):
            raise ValueError(f"Invalid participant roles in {fold_id}")
        test_person = group.loc[group.role.eq("test"), "participant_id"].iloc[0]
        if not group.outer_test_person.eq(test_person).all():
            raise ValueError(f"Incorrect outer Test identity in {fold_id}")
        held_out.append(test_person)
    if len(held_out) != len(people) or set(held_out) != people:
        raise ValueError("Each participant must be held out exactly once")


def reconcile_outer_predictions(predictions, index, folds, models):
    """Require complete matched predictions and recompute angular errors."""
    validate_lopo_design(index, folds)
    columns = ["sample_id", "participant_id", "fold_id", "model"]
    if predictions.empty or predictions[columns].isna().any().any():
        raise ValueError("Missing prediction identities")
    if set(predictions.model) != set(models):
        raise ValueError("Unexpected or missing model")
    if predictions.duplicated(["model", "sample_id"]).any():
        raise ValueError("Duplicate model/sample prediction")
    test_roles = folds.loc[folds.role.eq("test"), ["participant_id", "fold_id"]]
    expected = index[["sample_id", "participant_id", "gaze_x", "gaze_y", "gaze_z"]].merge(
        test_roles, on="participant_id", validate="many_to_one")
    expected = expected.set_index("sample_id")
    target_cols = ["target_x", "target_y", "target_z"]
    pred_cols = ["pred_x", "pred_y", "pred_z"]
    for model in models:
        part = predictions.loc[predictions.model.eq(model)].set_index("sample_id")
        if set(part.index) != set(expected.index):
            raise ValueError(f"Missing or unexpected sample coverage: {model}")
        aligned = expected.loc[part.index]
        for col in ("participant_id", "fold_id"):
            if not np.array_equal(part[col].to_numpy(), aligned[col].to_numpy()):
                raise ValueError(f"Incorrect {col}: {model}")
        target = part[target_cols].to_numpy(float)
        guess = part[pred_cols].to_numpy(float)
        if not np.allclose(target, aligned[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(float),
                           rtol=0, atol=1e-6):
            raise ValueError(f"Target mismatch: {model}")
        for vectors in (target, guess):
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1., rtol=0, atol=1e-5):
                raise ValueError(f"Non-unit or non-finite vectors: {model}")
    return person_angular_metrics(predictions)


# ======================= 05. Participant-Level Comparisons ===================
def summarise_outer_evaluation(observed, per_person, models, *, seed=17, draws=5000):
    """Describe participant means and paired differences on the fixed cohort.

    Resample participant error summaries, not individual images. Percentile
    intervals are conditional on saved predictions; overlapping LOPO training
    sets and uncertainty from retraining are not accounted for.
    """
    from itertools import combinations

    paired = per_person.pivot(index="participant_id", columns="model",
                              values="mean_angular_error_deg")[list(models)]
    if paired.empty or paired.isna().any().any() or draws < 1:
        raise ValueError("Incomplete paired participant metrics")
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(paired), size=(draws, len(paired)))
    values = paired.to_numpy(float)
    means = values[resamples].mean(axis=1)
    summary, differences = [], []
    for j, model in enumerate(models):
        errors = observed.loc[observed.model.eq(model), "angular_error_deg"].to_numpy(float)
        low, high = np.quantile(means[:, j], [0.025, 0.975])
        summary.append({
            "model": model, "n_people": len(paired), "n_samples": len(errors),
            "participant_macro_mean_deg": values[:, j].mean(),
            "participant_mean_sd_deg": values[:, j].std(ddof=1),
            "pooled_mean_deg": errors.mean(), "pooled_median_deg": np.median(errors),
            "pooled_sd_deg": errors.std(ddof=1), "pooled_rmse_deg": np.sqrt(np.mean(errors**2)),
            "participant_bootstrap_ci_low_deg": low,
            "participant_bootstrap_ci_high_deg": high,
        })
    for j, k in combinations(range(len(models)), 2):
        delta = values[:, j] - values[:, k]
        low, high = np.quantile(means[:, j] - means[:, k], [0.025, 0.975])
        differences.append({
            "model_a": models[j], "model_b": models[k], "n_people": len(paired),
            "mean_a_minus_b_deg": delta.mean(), "people_a_lower_error": int((delta < 0).sum()),
            "people_equal_error": int((delta == 0).sum()),
            "paired_bootstrap_ci_low_deg": low, "paired_bootstrap_ci_high_deg": high,
        })
    return pd.DataFrame(summary), pd.DataFrame(differences)
