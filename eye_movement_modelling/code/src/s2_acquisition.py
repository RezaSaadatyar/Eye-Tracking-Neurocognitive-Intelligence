# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Colab-only acquisition and preliminary checks for original GazeBase CSVs.

Original data are kept separate from tracked notebook source and model outputs.
This module does not assign training partitions, fit models, or inspect Test scores.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ================== 01. Published Source and Expected Tasks ==================
FIGSHARE_FILE = "https://figshare.com/ndownloader/files/27039812"
ARCHIVE_BYTES = 6_710_984_652
ARCHIVE_MD5 = "cb7eb895fb48f8661decf038ab998c9a"
# Try official Figshare endpoints first. This alternate host can bypass a
# zero-byte download response from the website endpoint on some Colab runtimes.
FIGSHARE_DOWNLOAD_URLS = (
    "https://ndownloader.figshare.com/files/27039812",
    "https://api.figshare.com/v2/file/download/27039812",
    FIGSHARE_FILE,
)
# Exact-archive mirror; accepted only if the official published MD5 matches.
# The mirror is a fallback transport, not a change to dataset provenance.
VERIFIED_ARCHIVE_MIRROR = (
    "https://huggingface.co/datasets/Emulated-Inc/"
    "eye-movement-biometrics-across-rounds/resolve/main/GazeBase_v2_0.zip"
)
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
    round_hint = re.search(r"Round[_ -]?(\d+)", member_name, re.I)
    if round_hint and int(round_hint.group(1)) != 1:
        return None
    match = FILE_RE.fullmatch(Path(member_name).name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2).upper()


# =================== 02. Resume and Check Official Archive ===================
def _archive_digest(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
    return md5.hexdigest()


def _download_response(
    *, url: str, partial: Path, already: int,
) -> int:
    """Download one response with strict HTTP/body validation and safe resume.

    A zero-byte/HTML response is rejected before overwriting a valid partial
    archive. If a server ignores Range, restart only after receiving ZIP data.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GazeBaseResearch/1.0)",
        "Accept": "application/octet-stream, application/zip, */*",
    }
    if already:
        headers["Range"] = f"bytes={already}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if status not in (200, 206):
            raise IOError(f"Unexpected HTTP status {status}")
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type or "application/json" in content_type:
            raise IOError(f"Expected ZIP bytes; server returned {content_type}")
        reported_length = response.headers.get("Content-Length", "")
        if reported_length.strip() == "0":
            raise IOError(f"HTTP {status} response has Content-Length: 0")
        content_range = response.headers.get("Content-Range", "")
        resume = (
            already > 0 and status == 206
            and content_range.startswith(f"bytes {already}-")
        )
        if status == 206 and not resume:
            raise IOError(
                f"Unexpected partial-content response: {content_range!r}"
            )
        # Do not truncate a resumable partial until nonempty response data
        # have actually been obtained from the alternative endpoint.
        first = response.read(1024 * 1024)
        if not first:
            raise IOError(f"HTTP {status} response returned zero file bytes")
        if not resume and not first.startswith(b"PK"):
            raise IOError(
                "Response is not a ZIP archive (missing PK signature); "
                f"Content-Type={content_type or 'unspecified'}"
            )
        mode = "ab" if resume else "wb"
        with partial.open(mode) as target:
            target.write(first)
            transferred = (already if resume else 0) + len(first)
            next_notice = (
                (transferred // 1_000_000_000) + 1
            ) * 1_000_000_000
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                target.write(block)
                transferred += len(block)
                if transferred >= next_notice:
                    print(f"Downloaded {transferred / 1e9:.1f} GB.")
                    next_notice += 1_000_000_000
        return transferred


def _ensure_archive(archive_path: Path) -> None:
    """Download GazeBase with validated fallback sources and safe resumption.

    Prefer official Figshare endpoints. If all fail, an exact-archive mirror
    is tried; in every case the published file length and MD5 are mandatory.
    """
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
    if partial.exists() and partial.stat().st_size == 0:
        # This is the precise failure from Figshare observed in Colab;
        # no source bytes exist to resume, so safely discard that empty file.
        partial.unlink()

    already = partial.stat().st_size if partial.exists() else 0
    if already > ARCHIVE_BYTES:
        raise ValueError("Partial archive exceeds the published source size")
    if already == ARCHIVE_BYTES:
        if _archive_digest(partial) != ARCHIVE_MD5:
            raise ValueError(
                "Complete temporary ZIP has an unexpected published MD5. "
                "Remove or inspect the corrupt .zip.part before retrying."
            )
        partial.replace(archive_path)
        return

    remaining = ARCHIVE_BYTES - already
    free = shutil.disk_usage(archive_path.parent).free
    if free < remaining + 2_000_000_000:
        raise OSError(
            "Insufficient Colab disk capacity for the official GazeBase ZIP. "
            "Free at least the remaining archive bytes plus 2 GB."
        )

    endpoints = (*FIGSHARE_DOWNLOAD_URLS, VERIFIED_ARCHIVE_MIRROR)
    errors = []
    for index, url in enumerate(endpoints, start=1):
        previous = partial.stat().st_size if partial.exists() else 0
        label = "official Figshare" if index <= len(FIGSHARE_DOWNLOAD_URLS) else "verified archive mirror"
        print(f"Download source {index}/{len(endpoints)} ({label}): {url}")
        try:
            transferred = _download_response(
                url=url, partial=partial, already=previous,
            )
        except (OSError, ValueError) as exc:
            errors.append(f"{label} {url}: {type(exc).__name__}: {exc}")
            if partial.exists() and partial.stat().st_size == 0:
                partial.unlink()
            print(f"Source {index} did not provide usable ZIP bytes.")
            continue

        if transferred > ARCHIVE_BYTES:
            raise IOError(
                "Download exceeds the published GazeBase ZIP size; "
                f"source={url}, received={transferred:,} bytes."
            )
        if transferred == ARCHIVE_BYTES:
            print("Verifying archive length and official published MD5...")
            if _archive_digest(partial) != ARCHIVE_MD5:
                raise ValueError(
                    "Downloaded archive has the published size but a "
                    "different MD5; do not use it. The partial ZIP is "
                    f"preserved for inspection: {partial}"
                )
            partial.replace(archive_path)
            print("Verified GazeBase archive ready for selective extraction.")
            return
        errors.append(
            f"{label} {url}: incomplete response; "
            f"{transferred:,}/{ARCHIVE_BYTES:,} bytes"
        )
        print(f"Incomplete source response: {transferred:,}/{ARCHIVE_BYTES:,}. "
              "Trying another endpoint with the saved partial ZIP.")

    detail = "\n".join(f"- {line}" for line in errors)
    saved = partial.stat().st_size if partial.exists() else 0
    raise OSError(
        "All GazeBase download endpoints failed or returned insufficient "
        "data from this Colab runtime. The Drive mount is unrelated.\n"
        f"Partial source bytes kept: {saved:,} at {partial}\n"
        "Try a fresh Colab runtime or download the official ZIP from a "
        "browser and place it at the expected archive path.\n"
        f"Endpoint diagnostics:\n{detail}"
    )


# ======================= 03. Select Only Intended CSVs =======================
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


# ================ 04. Archive-Indexed Participant Eligibility ================
def _archive_task_coverage(
    archive_path: Path, *, max_nested_depth: int = 3,
) -> dict[int, set[str]]:
    """Index actual Round-1/Session-1 task filenames without extracting CSVs.

    Both direct CSVs and nested source ZIPs are supported. Nested archives are
    inspected one at a time, with spooling to avoid retaining the dataset in RAM.
    This does not use gaze values, task labels beyond source identity, or Test.
    """
    coverage: dict[int, set[str]] = {}

    def inspect(archive: zipfile.ZipFile, depth: int) -> None:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            key = _source_key(info.filename)
            if key and info.filename.lower().endswith(".csv"):
                if info.file_size > 0:
                    coverage.setdefault(key[0], set()).add(key[1])
                continue
            if not info.filename.lower().endswith(".zip"):
                continue
            if depth >= max_nested_depth:
                continue
            round_hint = re.search(r"Round[_ -]?(\d+)", info.filename, re.I)
            if round_hint and int(round_hint.group(1)) != 1:
                continue
            with tempfile.SpooledTemporaryFile(
                max_size=64 * 1024 * 1024
            ) as nested:
                with archive.open(info) as stream:
                    shutil.copyfileobj(
                        stream, nested, length=8 * 1024 * 1024,
                    )
                nested.seek(0)
                with zipfile.ZipFile(nested) as inner:
                    inspect(inner, depth + 1)

    with zipfile.ZipFile(archive_path) as top:
        inspect(top, 0)
    return coverage


def ensure_gazebase_cohort_preflight(
    *, raw_root: Path, download_root: Path, output_csv: Path,
    participants: int, first_source_id: int = 1001,
) -> pd.DataFrame:
    """Select the first N *actually complete* source people in Round 1.

    The desired count is fixed. Missing source IDs are skipped only based on
    availability of all four original task files, never model results. Print
    skipped/ineligible source IDs and keep the chosen IDs in the audit CSV.
    """
    if not isinstance(participants, int) or not 10 <= participants <= 322:
        raise ValueError("Choose 10–322 participants for this Round-1 study")
    if first_source_id != 1001:
        raise ValueError("The fixed source order begins at 1001")

    raw_root = Path(raw_root)
    download_root = Path(download_root)
    output_csv = Path(output_csv)
    # Reuse complete original CSVs in the current Colab Runtime when enough
    # are present, without attempting another 6.7-GB download.
    complete_local = tuple(
        sid for sid in range(first_source_id, 2000)
        if all(source_csv_path(raw_root, sid, task).is_file()
               for task in TASKS)
    )
    archive = download_root / "GazeBase_v2_0.zip"
    used_archive = False
    if len(complete_local) >= participants:
        selected_ids = complete_local[:participants]
        print(f"Reusing {participants} complete local source people.")
    else:
        _ensure_archive(archive)
        used_archive = True
        coverage = _archive_task_coverage(archive)
        complete_archive = sorted(
            sid for sid, found_tasks in coverage.items()
            if sid >= first_source_id and set(TASKS).issubset(found_tasks)
        )
        if len(complete_archive) < participants:
            raise FileNotFoundError(
                "The verified archive contains only "
                f"{len(complete_archive)} people with all four requested "
                f"Round-1/Session-1 tasks, fewer than {participants}. "
                f"Verified archive retained for inspection: {archive}"
            )
        selected_ids = tuple(complete_archive[:participants])
        skipped = sorted(
            sid for sid in range(first_source_id, selected_ids[-1] + 1)
            if sid not in selected_ids
        )
        print(
            f"Selected {len(selected_ids)} complete source people "
            f"(IDs {selected_ids[0]}–{selected_ids[-1]})."
        )
        print(f"Unavailable/incomplete IDs skipped: {skipped}")
        missing_files = sum(
            not source_csv_path(raw_root, sid, task).is_file()
            for sid in selected_ids for task in TASKS
        )
        print(f"Recovering {missing_files} selected original CSVs.")
        # A failed extraction leaves the verified archive available to retry.
        extract_selected_csvs(archive, raw_root, selected_ids)

    # Audit selected people, reject invalid originals, and write only after
    # all four tasks per participant have passed the existing source checks.
    audit = ensure_gazebase_preflight(
        raw_root=raw_root, download_root=download_root,
        output_csv=output_csv, requested_ids=selected_ids,
    )
    if used_archive:
        archive.unlink(missing_ok=True)
    return audit


# ================== 04. Source Quality and Colab-Local Files =================
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


def ensure_gazebase_preflight(
    *, raw_root: Path, download_root: Path, output_csv: Path,
    requested_ids: tuple[int, ...],
) -> pd.DataFrame:
    """Prepare and inspect a bounded cohort entirely in the Colab runtime.

    Reuse existing original CSVs in the local repository checkout, downloading
    and extracting only missing requested files. No Drive, Test evaluation,
    model training or Git publication is performed.
    """
    raw_root = Path(raw_root)
    download_root = Path(download_root)
    output_csv = Path(output_csv)
    if not requested_ids or len(requested_ids) != len(set(requested_ids)):
        raise ValueError("Provide distinct, nonempty source participant IDs")
    # Preserve source files already present under the local checkout; they
    # remain available to the existing notebooks in this Colab runtime.

    missing = [
        (sid, task) for sid in requested_ids for task in TASKS
        if not source_csv_path(raw_root, sid, task).is_file()
    ]
    used_archive = False
    if missing:
        archive = download_root / "GazeBase_v2_0.zip"
        _ensure_archive(archive)
        used_archive = True
        print(f"Recovering {len(missing)} missing original CSVs from the archive.")
        # On extraction failure retain the verified ZIP for a local retry.
        extract_selected_csvs(archive, raw_root, requested_ids)

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
    if used_archive:
        archive.unlink(missing_ok=True)
    return frame
