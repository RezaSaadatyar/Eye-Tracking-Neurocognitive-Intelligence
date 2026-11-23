# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Access and audit the source distribution of MPIIGaze.

Only the explicit fetch/extract functions perform I/O beyond reading
metadata. Never infer image counts from the number of MATLAB containers.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import os
import re
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile

import pandas as pd

SOURCE_PAGE = "https://www.collaborative-ai.org/research/datasets/MPIIGaze/"
ARCHIVE_URL = "https://collaborative-ai.org/files/datasets/MPIIGaze.tar.gz"
SOURCE_DOI = "https://doi.org/10.18419/darus-3230"
PARTICIPANT = re.compile(r"p\d{2}", flags=re.IGNORECASE)


# ========================== 01. Source File Metadata =========================
def source_component(relative: Path) -> str:
    """Identify the publisher's source family from a relative file path."""
    parts = tuple(p.lower().replace("_", " ").replace("-", " ")
                  for p in relative.parts)
    if "normalized" in parts:
        return "Normalized"
    if "original" in parts:
        return "Original"
    if "calibration" in parts:
        return "Calibration"
    if any("evaluation subset" in p for p in parts):
        return "Evaluation Subset"
    if any("annotation subset" in p for p in parts):
        return "Annotation Subset"
    return "Other"


def source_inventory(root: Path) -> pd.DataFrame:
    """Index observed source files; a MAT container is not one image."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Original MPIIGaze directory is missing: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        person = next((part.lower() for part in rel.parts
                       if PARTICIPANT.fullmatch(part)), None)
        day = next((part for part in rel.parts
                    if re.fullmatch(r"day\d+", part, re.IGNORECASE)), None)
        records.append({
            "relative_path": rel.as_posix(),
            "component": source_component(rel),
            "participant_id": person,
            "recording_day": day,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        })
    return pd.DataFrame(records, columns=[
        "relative_path", "component", "participant_id", "recording_day",
        "suffix", "size_bytes",
    ])


def participant_source_support(inventory: pd.DataFrame) -> pd.DataFrame:
    """Summarise source containers, not verified image/target samples."""
    if inventory.empty:
        return pd.DataFrame(columns=[
            "participant_id", "source_files", "normalised_mat_files",
            "original_files", "calibration_files", "observed_day_folders",
        ])
    work = inventory.dropna(subset=["participant_id"])
    records = []
    for person, group in work.groupby("participant_id", sort=True):
        records.append({
            "participant_id": person,
            "source_files": int(len(group)),
            "normalised_mat_files": int(
                (group.component.eq("Normalized")
                 & group.suffix.eq(".mat")).sum()),
            "original_files": int(group.component.eq("Original").sum()),
            "calibration_files": int(group.component.eq("Calibration").sum()),
            "observed_day_folders": int(
                group.recording_day.dropna().nunique()),
        })
    return pd.DataFrame(records)


# ======================= 02. Inspect MATLAB File Headers =====================
def inspect_mat_header(path: Path) -> list[dict]:
    """Read variable names/shapes without materialising image tensors.

    MATLAB v5 is handled by SciPy; v7.3 HDF5 is handled by h5py.
    An unknown or corrupt file raises a clear error to the caller.
    """
    path = Path(path)
    try:
        from scipy.io import whosmat

        variables = whosmat(str(path))
        return [{"variable": name, "shape": str(tuple(shape)),
                 "storage_type": dtype, "format": "MATLAB v5"}
                for name, shape, dtype in variables]
    except (NotImplementedError, ValueError) as exc:
        import h5py

        if not h5py.is_hdf5(path):
            raise ValueError(f"Unsupported or invalid MATLAB source: {path}") from exc
        with h5py.File(path, "r") as handle:
            return [{"variable": name,
                     "shape": str(tuple(getattr(obj, "shape", ()))),
                     "storage_type": type(obj).__name__,
                     "format": "MATLAB HDF5"}
                    for name, obj in handle.items()]


# ======================= 03. Secure Archive Operations =======================
def _relative_archive_member(name: str) -> Path | None:
    """Validate the member path; permit only the archive's root marker."""
    raw = name.replace("\\", "/")
    if raw in (".", "./"):
        return None
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(
        p in ("", ".", "..") for p in raw.split("/")
        if p and p != "."
    ) or ":" in path.parts[0]:
        raise ValueError(f"Unsafe archive entry: {name}")
    if any(p == ".." for p in path.parts):
        raise ValueError(f"Unsafe archive entry: {name}")
    return Path(*path.parts)


def extract_source_archive(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP or tar.gz into a new source directory safely.

    Do not merge files into an existing directory or follow archive symlinks.
    A temporary extraction directory is atomically renamed on success.
    """
    archive_path, destination = Path(archive_path), Path(destination)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite the existing source: {destination}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing MPIIGaze archive: {archive_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mpiigaze_extract_",
                                 dir=str(destination.parent)))
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                for entry in archive.infolist():
                    relative = _relative_archive_member(entry.filename)
                    if relative is None:
                        continue
                    target = stage / relative
                    mode = (entry.external_attr >> 16)
                    if stat.S_ISLNK(mode):
                        raise ValueError("Symlinks in the source ZIP are not allowed")
                    if entry.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(entry) as reader, target.open("wb") as writer:
                            shutil.copyfileobj(reader, writer, length=2**20)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, mode="r:*") as archive:
                for entry in archive:
                    relative = _relative_archive_member(entry.name)
                    if relative is None:
                        continue
                    target = stage / relative
                    if entry.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif entry.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        stream = archive.extractfile(entry)
                        if stream is None:
                            raise ValueError(f"Missing archive stream: {entry.name}")
                        with stream, target.open("wb") as writer:
                            shutil.copyfileobj(stream, writer, length=2**20)
                    else:
                        raise ValueError(
                            f"Links and special archive entries are unsupported: "
                            f"{entry.name}")
        else:
            raise ValueError(
                "Source must be an MPIIGaze ZIP, .tar or .tar.gz archive")
        if not any(p.is_file() for p in stage.rglob("*")):
            raise ValueError("The source archive contains no files")
        stage.rename(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


# ===================== 04. Explicit Original-Data Download ===================
def download_source_archive(destination: Path, *, url: str = ARCHIVE_URL) -> Path:
    """Stream the official multi-GB source to an ignored local path.

    Call only after explicit user selection of download mode. Never interpret
    a partial download as a complete archive.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Archive already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    if part.exists():
        raise FileExistsError(
            f"Partial archive exists; inspect before restarting: {part}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "MPIIGaze-Academic-Source-Audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            if "text/html" in response.headers.get(
                "Content-Type", "").lower():
                raise ValueError(
                    "Publisher returned HTML, not an original-data archive")
            with part.open("wb") as out:
                shutil.copyfileobj(response, out, length=8 * 2**20)
        if not (zipfile.is_zipfile(part) or tarfile.is_tarfile(part)):
            raise ValueError("Downloaded file is not a valid source archive")
        os.replace(part, destination)
    except Exception:
        # Keep an incomplete transfer for inspection instead of erasing it.
        raise
    return destination
