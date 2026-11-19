# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Read bounded GazeBase CSV records with preserved sample provenance."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = "GazeBase"
ARCHIVE = "GazeBase_v2_0.zip"
DOI = "10.6084/m9.figshare.12912257"
FILE_PATTERN = re.compile(
    r"^S_(?P<round>[1-9])(?P<participant>\d{3})_S"
    r"(?P<session>[12])_(?P<task>HSS|VD1|FXS|RAN|TEX|BLG|VD2)\.csv$",
    re.IGNORECASE,
)
REQUIRED = ("n", "x", "y", "val")
TASKS = ("FXS", "HSS", "RAN", "TEX")


def discover_records(
    root: str | Path, *, round_id: int = 1, session_id: int = 1,
    tasks: tuple[str, ...] = TASKS, max_participants: int = 10,
) -> pd.DataFrame:
    """Select complete participant/task bundles without order dependence.

    This selection is descriptive, not a random sample or scientific cohort.
    Participants must have each requested task in the selected round/session.
    """
    folder = Path(root)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"GazeBase CSV directory missing: {folder}. Extract authorised "
            "GazeBase CSV records under data/raw/gazebase/ first."
        )
    if max_participants < 1:
        raise ValueError("max_participants must be positive")
    rows: list[dict] = []
    for path in folder.rglob("*.csv"):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        info = match.groupdict()
        if int(info["round"]) != round_id or int(info["session"]) != session_id:
            continue
        if info["task"].upper() not in tasks:
            continue
        rows.append({
            "path": str(path.resolve()),
            "relative_file": path.relative_to(folder).as_posix(),
            "record_id": path.stem,
            "participant_id": info["participant"],
            "round_id": int(info["round"]),
            "session_id": int(info["session"]),
            "task": info["task"].upper(),
        })
    if not rows:
        raise FileNotFoundError(
            f"No matching GazeBase task CSVs found under {folder}. "
            f"Expected source filenames such as S_1001_S1_FXS.csv; "
            "inspect the extracted GazeBase directory and selected filters."
        )
    inventory = pd.DataFrame(rows).sort_values(
        ["participant_id", "task", "relative_file"]
    )
    duplicates = inventory.duplicated(["participant_id", "task"], keep=False)
    if duplicates.any():
        examples = inventory.loc[
            duplicates, ["participant_id", "task", "relative_file"]
        ].to_dict("records")[:5]
        raise ValueError(
            "Ambiguous multiple source files for the same participant/task "
            f"in the selected round/session: {examples}. Resolve the source "
            "selection rather than discarding records silently."
        )
    counts = inventory.groupby("participant_id")["task"].nunique()
    selected = sorted(counts[counts == len(tasks)].index)[:max_participants]
    if not selected:
        raise ValueError(
            "No participant has all requested tasks in the chosen round/session. "
            "Try inspecting the available source records before changing scope."
        )
    return inventory[inventory.participant_id.isin(selected)].sort_values(
        ["participant_id", "task"]
    ).reset_index(drop=True)


def read_record(row: dict | pd.Series) -> pd.DataFrame:
    """Read one real source file; retain original units and parser labels."""
    raw = pd.read_csv(row["path"], low_memory=False)
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"{row['path']} missing required source columns {sorted(missing)}")
    for field in ["n", "x", "y", "val", "xT", "yT", "dP", "lab"]:
        if field in raw.columns:
            raw[field] = pd.to_numeric(raw[field], errors="coerce")
    if raw["n"].isna().any():
        raise ValueError(f"Invalid/non-numeric timestamp in {row['path']}")
    raw.insert(0, "sample_index", np.arange(len(raw), dtype="int64"))
    raw["sample_valid"] = (
        raw["val"].eq(0) & np.isfinite(raw["x"]) & np.isfinite(raw["y"])
    )
    raw["source"] = SOURCE
    raw["source_archive"] = ARCHIVE
    for key in ("record_id", "participant_id", "round_id", "session_id", "task"):
        raw[key] = row[key]
    raw["source_file"] = row["relative_file"]
    raw["time_s"] = (raw["n"] - raw["n"].iloc[0]) / 1000.0
    # Keep the raw label field; no assumption that every task has labels.
    if "lab" not in raw:
        raw["lab"] = np.nan
    if "dP" not in raw:
        raw["dP"] = np.nan
    return raw


def load_selected_records(inventory: pd.DataFrame) -> pd.DataFrame:
    """Concatenate only selected original recordings, keyed by record/sample."""
    if inventory.empty:
        raise ValueError("No selected source recordings")
    frames = [read_record(row) for _, row in inventory.iterrows()]
    samples = pd.concat(frames, ignore_index=True, sort=False)
    if samples.duplicated(["record_id", "sample_index"]).any():
        raise ValueError("Duplicate record/sample identity")
    return samples


def require_columns(table: pd.DataFrame, columns: list[str] | tuple[str,...]) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"Missing required stage columns: {missing}")


def read_stage(path: str | Path, required: tuple[str, ...]) -> pd.DataFrame:
    """Read an explicit upstream current CSV, never select a historical copy."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expected upstream stage output: {path}")
    frame = pd.read_csv(
        path, low_memory=False, dtype={"participant_id": str, "record_id": str},
    )
    require_columns(frame, required)
    return frame
