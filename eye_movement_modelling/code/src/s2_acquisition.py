# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Colab-only acquisition and preliminary checks for original GazeBase CSVs.

Original data are kept separate from tracked notebook source and model outputs.
This module does not assign training partitions, fit models, or inspect Test scores.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# =================== 01. Published Source and Expected Tasks =================
FIGSHARE_FILE = "https://figshare.com/ndownloader/files/27039812"
ARCHIVE_BYTES = 6_710_984_652
ARCHIVE_MD5 = "cb7eb895fb48f8661decf038ab998c9a"
TASKS = ("FXS", "HSS", "RAN", "TEX")
TASK_FOLDERS = {
    "FXS": "S1_Fixations", "HSS": "S1_Horizontal_Saccades",
    "RAN": "S1_Random_Saccades", "TEX": "S1_Reading",
}
FILE_RE = re.compile(r"^S_(1\d{3})_S1_(FXS|HSS|RAN|TEX)\.csv$", re.I)
REQUIRED_COLUMNS = ("n", "x", "y", "val")


def source_csv_path(raw_root: Path, source_id: int, task: str) -> Path:
    """Construct an original-named Round 1 / Session 1 source destination."""
    if source_id < 1000 or source_id > 1999 or task not in TASKS:
        raise ValueError("Expected a Round 1 source ID and approved task label")
    filename = f"S_{source_id}_S1_{task}.csv"
    return (Path(raw_root) / "GazeBase_v2_0" / "Round_1"
            / f"Subject_{source_id}" / TASK_FOLDERS[task] / filename)


def _source_key(member_name: str):
    """Match only original Round 1 / Session 1 source filename identities."""
    match = FILE_RE.fullmatch(Path(member_name).name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2).upper()


# ==================== 02. Resume and Check Official Archive ==================
def _archive_digest(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
    return md5.hexdigest()


def _ensure_archive(archive_path: Path) -> None:
    """Download the official ZIP with a resumable temporary file and check MD5."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file():
        if (archive_path.stat().st_size == ARCHIVE_BYTES
                and _archive_digest(archive_path) == ARCHIVE_MD5):
            print("Reusing the verified GazeBase archive.")
            return
        raise ValueError(
            f"Existing archive failed the published size/MD5 check: {archive_path}"
        )

    partial = archive_path.with_suffix(".zip.part")
    already = partial.stat().st_size if partial.exists() else 0
    if already > ARCHIVE_BYTES:
        raise ValueError("Partial archive is larger than the published source")
    remaining = ARCHIVE_BYTES - already
    free = shutil.disk_usage(archive_path.parent).free
    if free < remaining + 2_000_000_000:
        raise OSError(
            "Insufficient free Colab disk for official GazeBase ZIP and "
            "temporary extraction. Free at least the remaining archive "
            "bytes plus 2 GB before running this cell."
        )

    headers = {"User-Agent": "GazeBase-Colab-Research/1.0"}
    if already:
        headers["Range"] = f"bytes={already}-"
    request = urllib.request.Request(FIGSHARE_FILE, headers=headers)
    print("Downloading official Figshare GazeBase v2.0 archive; "
          "the source ZIP is approximately 6.7 GB.")
    with urllib.request.urlopen(request, timeout=120) as response:
        status = response.status
        content_range = response.headers.get("Content-Range", "")
        is_resume = (
            already > 0 and status == 206
            and content_range.startswith(f"bytes {already}-")
        )
        if already and not is_resume:
            # Do not append a full response to a partial archive.
            already = 0
        mode = "ab" if is_resume else "wb"
        with partial.open(mode) as target:
            transferred = already
            next_notice = ((transferred // 1_000_000_000) + 1) * 1_000_000_000
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                target.write(block)
                transferred += len(block)
                if transferred >= next_notice:
                    print(f"Downloaded {transferred / 1e9:.1f} GB of source ZIP.")
                    next_notice += 1_000_000_000

    if partial.stat().st_size != ARCHIVE_BYTES:
        raise IOError(
            f"Source download incomplete: {partial.stat().st_size:,} / "
            f"{ARCHIVE_BYTES:,} bytes. Rerun the cell to resume."
        )
    if _archive_digest(partial) != ARCHIVE_MD5:
        raise ValueError(
            "Source archive MD5 differs from the published GazeBase v2.0 "
            "checksum; do not extract or use these bytes."
        )
    partial.replace(archive_path)


# ====================== 03. Select Only Intended CSVs ========================
def _copy_archive_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo,
                         destination: Path) -> None:
    """Extract an identified CSV without trusting archive-relative paths."""
    if info.is_dir() or info.file_size <= 0:
        raise ValueError(f"Empty/non-file source member: {info.filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.partial")
    try:
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if temporary.stat().st_size != info.file_size:
            raise IOError(f"Incorrect decompressed CSV length: {info.filename}")
        if destination.exists():
            raise FileExistsError(
                f"Refusing to replace an existing source CSV: {destination}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_selected_csvs(
    archive_path: Path, raw_root: Path, requested_ids: tuple[int, ...],
    *, max_nested_depth: int = 3,
) -> int:
    """Extract only missing FXS/HSS/RAN/TEX files for the requested source IDs.

    Direct CSV members and nested ZIPs are supported; no unrelated task,
    round, session or participant is written. Existing source files are kept.
    """
    if not requested_ids or len(set(requested_ids)) != len(requested_ids):
        raise ValueError("A nonempty set of unique source IDs is required")
    wanted = {
        (source_id, task)
        for source_id in requested_ids for task in TASKS
        if not source_csv_path(raw_root, source_id, task).is_file()
    }
    extracted = 0

    def visit(archive: zipfile.ZipFile, depth: int) -> None:
        nonlocal extracted
        for info in sorted(archive.infolist(), key=lambda x: x.filename):
            if not wanted:
                return
            if info.is_dir():
                continue
            key = _source_key(info.filename)
            if key in wanted and info.filename.lower().endswith(".csv"):
                _copy_archive_member(
                    archive, info, source_csv_path(raw_root, *key),
                )
                wanted.remove(key)
                extracted += 1
                continue
            if not info.filename.lower().endswith(".zip"):
                continue
            if depth >= max_nested_depth:
                continue
            # Ignore clearly unrelated rounds and participants when named.
            round_hint = re.search(r"Round[_ -]?(\d+)", info.filename, re.I)
            if round_hint and int(round_hint.group(1)) != 1:
                continue
            subject_hint = re.search(r"Subject[_ -]?(\d+)", info.filename, re.I)
            if subject_hint and int(subject_hint.group(1)) not in requested_ids:
                continue
            with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as nested:
                with archive.open(info) as stream:
                    shutil.copyfileobj(stream, nested, length=8 * 1024 * 1024)
                nested.seek(0)
                with zipfile.ZipFile(nested) as inner:
                    visit(inner, depth + 1)

    with zipfile.ZipFile(archive_path) as top:
        visit(top, 0)
    if wanted:
        examples = ", ".join(
            f"S_{sid}_S1_{task}.csv" for sid, task in sorted(wanted)[:8]
        )
        raise FileNotFoundError(
            f"The verified archive does not contain all requested CSVs; "
            f"still missing {len(wanted)}: {examples}."
        )
    return extracted


# ================= 04. Source Quality and Persistent Cache ====================
def _inspect_csv(path: Path) -> dict:
    """Inspect all rows without loading the entire original recording in RAM."""
    n_rows = n_valid = bad_time_steps = 0
    previous_time = None
    for frame in pd.read_csv(path, usecols=list(REQUIRED_COLUMNS),
                             chunksize=100_000, low_memory=False):
        if len(frame) == 0:
            continue
        time = pd.to_numeric(frame["n"], errors="coerce").to_numpy(float)
        if not np.isfinite(time).all():
            raise ValueError(f"Non-numeric/non-finite timestamps: {path}")
        x = pd.to_numeric(frame["x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(float)
        code = pd.to_numeric(frame["val"], errors="coerce").to_numpy(float)
        good = (code == 0) & np.isfinite(x) & np.isfinite(y)
        n_rows += len(frame)
        n_valid += int(good.sum())
        if previous_time is not None:
            bad_time_steps += int(time[0] <= previous_time)
        bad_time_steps += int(np.count_nonzero(np.diff(time) <= 0))
        previous_time = time[-1]
    if n_rows == 0 or n_valid == 0:
        raise ValueError(f"No valid samples in requested original CSV: {path}")
    return {
        "n_samples": n_rows,
        "valid_fraction": round(n_valid / n_rows, 8),
        "nonincreasing_time_steps": bad_time_steps,
    }


def _copy_without_overwriting(source: Path, target: Path) -> None:
    """Cache or restore a source file without replacing a different file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing source/cache size conflict: {target}")
        return
    temporary = target.with_suffix(".csv.partial")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_gazebase_preflight(
    *, raw_root: Path, cache_root: Path, download_root: Path,
    output_csv: Path, requested_ids: tuple[int, ...],
) -> pd.DataFrame:
    """Prepare and inspect a bounded Colab/Drive cohort without model fitting.

    Requires an already-mounted Google Drive directory for persistence.
    The original GazeBase ZIP is removed after validated extraction/cache.
    Outputs are a per-source CSV table, not Train/Validation/Test assignments.
    """
    raw_root, cache_root = Path(raw_root), Path(cache_root)
    download_root, output_csv = Path(download_root), Path(output_csv)
    if not cache_root.is_dir():
        raise FileNotFoundError(
            "Drive cache directory is unavailable. Mount Google Drive first."
        )
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("Duplicate source IDs requested")
    for sid in requested_ids:
        for task in TASKS:
            local = source_csv_path(raw_root, sid, task)
            cached = source_csv_path(cache_root, sid, task)
            if not local.exists() and cached.is_file():
                _copy_without_overwriting(cached, local)

    missing = [
        (sid, task) for sid in requested_ids for task in TASKS
        if not source_csv_path(raw_root, sid, task).is_file()
    ]
    if missing:
        archive = download_root / "GazeBase_v2_0.zip"
        _ensure_archive(archive)
        print(f"Recovering {len(missing)} missing original CSVs from the archive.")
        try:
            extract_selected_csvs(archive, raw_root, requested_ids)
        finally:
            # Keep any partially transferred archive for a later resume, but
            # remove the completed temporary ZIP after extraction is attempted.
            if archive.is_file():
                archive.unlink()

    rows = []
    for sid in requested_ids:
        for task in TASKS:
            local = source_csv_path(raw_root, sid, task)
            if not local.is_file():
                raise FileNotFoundError(f"Missing requested source CSV: {local}")
            quality = _inspect_csv(local)
            rows.append({
                "source_id": sid,
                "record_id": f"S_{sid}_S1_{task}",
                "task": task,
                "n_bytes": local.stat().st_size,
                **quality,
            })
            _copy_without_overwriting(
                local, source_csv_path(cache_root, sid, task),
            )
    frame = pd.DataFrame(rows)
    if frame.source_id.nunique() != len(requested_ids):
        raise ValueError("Not all requested independent people are present")
    if frame.groupby("source_id").task.nunique().ne(len(TASKS)).any():
        raise ValueError("A requested person has incomplete task recordings")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(".csv.partial")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(output_csv)
    finally:
        temporary.unlink(missing_ok=True)
    return frame
