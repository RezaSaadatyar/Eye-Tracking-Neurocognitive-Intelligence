# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Source-identifiable GazeBase task data and participant-disjoint partitions.

Original CSVs are shared with Eye-Tracking Foundations; all generated outputs belong to Eye-Movement Modelling.
No file from the completed Eye-Tracking Foundations output directory is read or overwritten.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ("FXS", "HSS", "RAN", "TEX")
CLASS_TO_INDEX = {task: i for i, task in enumerate(TASKS)}
FEATURES = (
    "valid_fraction", "x_sd_dva", "y_sd_dva", "step_median_dva",
    "step_p95_dva", "speed_median_dva_s", "speed_p95_dva_s",
    "pupil_coverage", "pupil_median_uncalibrated",
)
CHANNELS = ("relative_x_dva", "relative_y_dva",
            "delta_x_dva", "delta_y_dva")


def participant_split(
    inventory: pd.DataFrame, *, seed: int = 17,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.20,
) -> pd.DataFrame:
    """Group recordings by person and reserve independent Test participants.

    For cohorts of at least twenty people, reserve the highest source IDs
    for Test and select Validation reproducibly from the development pool.
    Validation fraction refers to the development pool, not the full cohort.
    """
    required = {"record_id", "participant_id", "task"}
    if not required.issubset(inventory):
        raise ValueError("Inventory lacks source record, person or task identity")
    if inventory.duplicated(["record_id"]).any():
        raise ValueError("Duplicate original recording identity")
    groups = sorted(inventory.participant_id.astype(str).unique())
    n = len(groups)
    if n < 20:
        raise ValueError(
            "At least twenty complete people are needed for "
            "participant-disjoint model evaluation"
        )
    if not (
        0 < test_fraction < 0.5 and 0 < validation_fraction < 0.5
        and np.isfinite(test_fraction) and np.isfinite(validation_fraction)
    ):
        raise ValueError("Split fractions must be finite and between 0 and 0.5")
    rng = np.random.default_rng(seed)
    count_test = max(2, int(np.ceil(test_fraction * n)))
    held_out = set(groups[-count_test:])
    pool = np.array(groups[:-count_test], dtype=str)
    rng.shuffle(pool)
    count_val = max(2, int(np.ceil(validation_fraction * len(pool))))
    validation = set(pool[:count_val])
    training = set(pool[count_val:])
    purpose = "main"
    if not training or not validation or not held_out:
        raise ValueError("Participant partitions must be nonempty")
    membership = {
        **{p: "train" for p in training},
        **{p: "validation" for p in validation},
        **{p: "test" for p in held_out},
    }
    result = inventory[["record_id", "participant_id", "task"]].copy()
    result["participant_id"] = result.participant_id.astype(str)
    result["split"] = result.participant_id.map(membership)
    result["study_purpose"] = purpose
    if result.split.isna().any():
        raise ValueError("Some source participants were not assigned")
    sets = [set(result.loc[result.split.eq(s), "participant_id"])
            for s in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Leakage: one participant belongs to multiple splits")
    return result.sort_values("record_id").reset_index(drop=True)


def _summaries(raw: pd.DataFrame) -> dict[str, float]:
    """Describe the full original recording without using class metadata."""
    x = pd.to_numeric(raw["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(raw["y"], errors="coerce").to_numpy(float)
    n = pd.to_numeric(raw["n"], errors="coerce").to_numpy(float)
    code = pd.to_numeric(raw["val"], errors="coerce").to_numpy(float)
    valid = (code == 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(n)
    dx, dy, dt = np.diff(x), np.diff(y), np.diff(n)
    step_ok = valid[1:] & valid[:-1] & (dt > 0) & (dt <= 5)
    displacement = np.hypot(dx[step_ok], dy[step_ok])
    speed = 1000 * displacement / dt[step_ok]
    pupil = pd.to_numeric(
        raw.get("dP", pd.Series(np.nan, index=raw.index)),
        errors="coerce",
    ).to_numpy(float)
    pupil_ok = valid & np.isfinite(pupil) & (pupil > 0)
    def safe(a, fn):
        return float(fn(a)) if len(a) else np.nan
    return {
        "valid_fraction": float(np.mean(valid)),
        "x_sd_dva": safe(x[valid], np.std),
        "y_sd_dva": safe(y[valid], np.std),
        "step_median_dva": safe(displacement, np.median),
        "step_p95_dva": safe(displacement, lambda a: np.percentile(a, 95)),
        "speed_median_dva_s": safe(speed, np.median),
        "speed_p95_dva_s": safe(speed, lambda a: np.percentile(a, 95)),
        "pupil_coverage": float(pupil_ok.mean()),
        "pupil_median_uncalibrated": safe(pupil[pupil_ok], np.median),
        "n_samples": int(len(raw)),
    }


def _windows(
    raw: pd.DataFrame, *, window_samples: int, max_windows: int,
) -> np.ndarray:
    """Take evenly spaced, non-overlapping windows within valid time segments.

    Invalid samples and >5-ms timing gaps are NEVER bridged. The source gaze
    coordinates are centred within each window; no vendor labels enter inputs.
    """
    if window_samples < 32 or max_windows < 1:
        raise ValueError("Window length >=32 and max_windows >=1 are required")
    x = pd.to_numeric(raw["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(raw["y"], errors="coerce").to_numpy(float)
    n = pd.to_numeric(raw["n"], errors="coerce").to_numpy(float)
    val = pd.to_numeric(raw["val"], errors="coerce").to_numpy(float)
    good = (val == 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(n)
    contiguous = good[:-1] & good[1:] & (np.diff(n) > 0) & (np.diff(n) <= 5)
    starts = np.r_[0, np.flatnonzero(~contiguous) + 1]
    ends = np.r_[np.flatnonzero(~contiguous) + 1, len(raw)]
    candidates: list[int] = []
    for start, end in zip(starts, ends):
        if not good[start] or end - start < window_samples:
            continue
        candidates.extend(range(int(start), int(end) - window_samples + 1,
                                window_samples))
    if not candidates:
        raise ValueError(
            "No complete valid contiguous source windows; inspect quality "
            "before changing scientific window or exclusion definitions."
        )
    if len(candidates) > max_windows:
        chosen = np.linspace(0, len(candidates) - 1,
                             num=max_windows, dtype=int)
        candidates = [candidates[int(i)] for i in chosen]
    windows = np.empty((len(candidates), window_samples, 4), dtype=np.float32)
    for i, start in enumerate(candidates):
        xx = x[start:start + window_samples]
        yy = y[start:start + window_samples]
        xx = xx - np.median(xx)
        yy = yy - np.median(yy)
        windows[i, :, 0] = xx
        windows[i, :, 1] = yy
        windows[i, 1:, 2] = np.diff(xx)
        windows[i, 1:, 3] = np.diff(yy)
        windows[i, 0, 2:] = 0
    if not np.isfinite(windows).all():
        raise ValueError("Non-finite temporal features in valid windows")
    return windows


def build_source_tables(
    inventory: pd.DataFrame, split: pd.DataFrame, *,
    window_samples: int = 256, max_windows_per_record: int = 24,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Stream each original recording once; never import Eye-Tracking Foundations results."""
    from gaze_source import read_record
    if set(inventory.record_id) != set(split.record_id):
        raise ValueError("Inventory/split record IDs differ")
    lookup = split.set_index("record_id")
    rows, arrays, recs, classes = [], [], [], []
    for _, item in inventory.iterrows():
        raw = read_record(item)
        item_windows = _windows(
            raw, window_samples=window_samples,
            max_windows=max_windows_per_record,
        )
        row = {key: item[key] for key in (
            "record_id", "participant_id", "round_id", "session_id", "task"
        )}
        row.update(_summaries(raw))
        row["split"] = str(lookup.loc[item["record_id"], "split"])
        row["n_windows"] = int(len(item_windows))
        rows.append(row)
        arrays.append(item_windows)
        recs.extend([item["record_id"]] * len(item_windows))
        classes.extend([CLASS_TO_INDEX[item["task"]]] * len(item_windows))
    frame = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    features = np.concatenate(arrays, axis=0)
    return frame, features, np.asarray(recs, dtype=str), np.asarray(classes, dtype=np.int64)




def verify_temporal_source_alignment(
    arrays: dict[str, np.ndarray], partitions: pd.DataFrame,
) -> dict[str, int]:
    """Match every temporal window's source Task to the current split table.

    This guards against combining Notebook 01 windows from a previous cohort
    with participant partitions produced by a newer rerun.
    """
    for key in ("x", "y", "record_id"):
        if key not in arrays:
            raise ValueError(f"Source sequence archive missing {key}")
    x, y, ids = arrays["x"], arrays["y"], arrays["record_id"]
    if x.ndim != 3 or x.shape[2] != len(CHANNELS):
        raise ValueError("Unexpected temporal channel dimensions")
    if len(x) != len(y) or len(y) != len(ids):
        raise ValueError("Temporal source identity rows do not align")
    if partitions["record_id"].duplicated().any():
        raise ValueError("Duplicated record IDs in participant registry")
    if not {"record_id", "task", "split", "participant_id"}.issubset(
        partitions.columns
    ):
        raise ValueError("Current participant registry is incomplete")
    lookup = dict(zip(
        partitions.record_id.astype(str),
        partitions.task.map(CLASS_TO_INDEX),
    ))
    source_ids = np.asarray(ids).astype(str)
    if set(source_ids) != set(lookup):
        raise ValueError("Window and split recording sets differ")
    expected = np.array([lookup[key] for key in source_ids], dtype=int)
    if not np.array_equal(expected, np.asarray(y, dtype=int)):
        raise ValueError("Temporal window Task labels differ from source records")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite model input windows")
    return {role: int(np.isin(
        source_ids,
        partitions.loc[partitions.split.eq(role), "record_id"].astype(str),
    ).sum()) for role in ("train", "validation", "test")}


def normalise_train_only(
    windows: np.ndarray, record_ids: np.ndarray, split: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute channel location/scale ONLY from train people's windows."""
    membership = dict(zip(split.record_id, split.split))
    roles = np.asarray([membership[str(x)] for x in record_ids])
    if not (roles == "train").any():
        raise ValueError("No training windows available")
    train = windows[roles == "train"]
    mean = train.mean(axis=(0, 1), dtype=np.float64)
    std = train.std(axis=(0, 1), dtype=np.float64)
    scale = np.maximum(std, 1e-5)
    transformed = ((windows - mean) / scale).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite train-normalised windows")
    return transformed, mean, scale


def save_protocol(path: str | Path, *, participants: int,
                  seed: int, window_samples: int,
                  max_windows_per_record: int, purpose: str,
                  test_fraction: float = 0.20,
                  validation_fraction: float = 0.20,
                  source_first_id: int = 1001,
                  selected_source_ids: tuple[int, ...] | None = None) -> dict:
    """Save the current scientific configuration in a readable CSV row."""
    if selected_source_ids is not None:
        selected_source_ids = tuple(int(sid) for sid in selected_source_ids)
        if (
            len(selected_source_ids) != participants
            or len(set(selected_source_ids)) != participants
            or tuple(sorted(selected_source_ids)) != selected_source_ids
            or selected_source_ids[0] < source_first_id
        ):
            raise ValueError(
                "Selected source IDs must be sorted, distinct and match "
                "the requested participant count"
            )
    settings = {
        "dataset": "GazeBase_v2_0", "round_id": 1, "session_id": 1,
        "task_labels": list(TASKS), "max_participants": participants,
        "split_seed": seed, "window_samples": window_samples,
        "max_windows_per_record": max_windows_per_record,
        "test_fraction": test_fraction,
        "validation_fraction_in_development": validation_fraction,
        "source_first_id": source_first_id,
        "source_last_id": (
            selected_source_ids[-1] if selected_source_ids is not None
            else source_first_id + participants - 1
        ),
        "purpose": purpose,
        "test_role": "new_source_id_reserved_participants",
        "target": "task_identity_not_cognition",
        "temporal_channels": list(CHANNELS),
    }
    if selected_source_ids is not None:
        settings["selected_source_ids"] = list(selected_source_ids)
    csv_path = Path(path)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("The protocol must be saved as a CSV file")
    row = {
        key: "; ".join(map(str, value)) if isinstance(value, list) else value
        for key, value in settings.items()
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    # Remove only the obsolete configuration written by this producer.
    csv_path.with_suffix(".json").unlink(missing_ok=True)
    return settings


def load_protocol(path: str | Path) -> dict:
    """Recover typed settings from a single-row protocol CSV."""
    csv_path = Path(path)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("The expected protocol input is a CSV file")
    table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {
        "dataset", "round_id", "session_id", "task_labels",
        "max_participants", "split_seed", "window_samples",
        "max_windows_per_record", "purpose", "test_role", "target",
        "temporal_channels",
    }
    missing = required.difference(table.columns)
    if len(table) != 1 or missing:
        raise ValueError(
            f"Protocol must have one row and required columns: missing={sorted(missing)}"
        )
    settings = table.iloc[0].to_dict()
    for key in (
        "round_id", "session_id", "max_participants", "split_seed",
        "window_samples", "max_windows_per_record",
    ):
        settings[key] = int(settings[key])
    for key in ("task_labels", "temporal_channels"):
        settings[key] = [part.strip() for part in settings[key].split(";")
                         if part.strip()]
    if "selected_source_ids" in settings:
        settings["selected_source_ids"] = [
            int(part.strip()) for part in
            settings["selected_source_ids"].split(";") if part.strip()
        ]
    for key in ("source_first_id", "source_last_id"):
        if key in settings:
            settings[key] = int(settings[key])
    for key in ("test_fraction", "validation_fraction_in_development"):
        if key in settings:
            settings[key] = float(settings[key])
    return settings


def load_windows(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
