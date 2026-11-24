# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Access and audit the source distribution of MPIIGaze.

Reuse the Colab runtime source cache or acquire the official distribution
when it is absent. Never infer image counts from MATLAB container totals.
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
DARUS_ARCHIVE_URL = (
    "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId"
    "?persistentId=doi:10.18419/DARUS-3230/1"
)
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


# ====================== 04. Official Source Acquisition ======================
def download_source_archive(destination: Path, *, url: str = ARCHIVE_URL) -> Path:
    """Stream an official archive to the runtime, never to the user's PC.

    A unique temporary file prevents an interrupted transfer from blocking
    subsequent attempts. Only a recognised archive is promoted on success.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Archive already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(
        prefix=".mpiigaze_download_", suffix=".part",
        dir=destination.parent)
    os.close(fd)
    part = Path(filename)
    request = urllib.request.Request(
        url, headers={"User-Agent": "MPIIGaze-Academic-Source-Audit/1.0",
                      "Accept-Encoding": "identity"})
    try:
        print("Retrieving official MPIIGaze archive:", url, flush=True)
        with urllib.request.urlopen(request, timeout=180) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "application/json" in content_type:
                raise ValueError("Source server returned a page, not an archive")
            total = response.headers.get("Content-Length")
            expected = int(total) if total and total.isdigit() else None
            received = 0
            next_report = 256 * 2**20
            with part.open("wb") as out:
                while True:
                    chunk = response.read(8 * 2**20)
                    if not chunk:
                        break
                    out.write(chunk)
                    received += len(chunk)
                    if received >= next_report:
                        print(f"Downloaded {received / 2**20:,.0f} MiB",
                              flush=True)
                        next_report = received + 256 * 2**20
            if expected is not None and received != expected:
                raise IOError(
                    f"Incomplete archive: {received:,} of {expected:,} bytes")
        if not (zipfile.is_zipfile(part) or tarfile.is_tarfile(part)):
            raise ValueError("Downloaded file is not a valid ZIP/TAR archive")
        os.replace(part, destination)
        print(f"Archive received: {received / 2**30:.2f} GiB", flush=True)
    finally:
        part.unlink(missing_ok=True)
    return destination


# ========================== 05. Runtime Source Cache ==========================
def normalised_mat_files(raw_dir: Path) -> list[Path]:
    """Find the participant/day MAT containers consumed by Notebook 02."""
    root = Path(raw_dir)
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.rglob("*.mat")
        if path.is_file() and not path.is_symlink()
        and source_component(path.relative_to(root)) == "Normalized"
        and PARTICIPANT.fullmatch(path.parent.name)
    )


def ensure_mpiigaze_source(raw_dir: Path) -> Path:
    """Reuse extracted source, extract a cached archive, or fetch automatically.

    Intended for the remote Colab checkout's ignored data/raw directory.
    Official mirrors are attempted in order. The archive and extracted
    source stay in the runtime; no raw data are packaged or published.
    Existing non-empty source folders are never overwritten.
    """
    raw_dir = Path(raw_dir)
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    existing = normalised_mat_files(raw_dir)
    if existing:
        if any(path.stat().st_size == 0 for path in existing):
            raise ValueError(f"Empty normalised MAT container in {raw_dir}")
        print(f"Reusing MPIIGaze runtime cache: {raw_dir}", flush=True)
        print(f"Normalised MAT containers: {len(existing):,}")
        return raw_dir
    if raw_dir.exists():
        if not raw_dir.is_dir() or any(raw_dir.iterdir()):
            raise ValueError(
                f"Existing source path has no recognised normalised MAT files: "
                f"{raw_dir}. Preserve and inspect it before replacing it.")
        raw_dir.rmdir()

    sources = (
        ("MPIIGaze.tar.gz", ARCHIVE_URL),
        ("MPIIGaze.zip", DARUS_ARCHIVE_URL),
    )
    errors = []
    # Prefer any already downloaded archive, regardless of mirror order.
    ordered = sorted(sources,
                     key=lambda item: not (raw_dir.parent / item[0]).is_file())
    for name, url in ordered:
        archive = raw_dir.parent / name
        try:
            if not archive.is_file():
                if shutil.disk_usage(raw_dir.parent).free < 8 * 2**30:
                    raise RuntimeError(
                        "Insufficient runtime disk for archive download and "
                        "extraction (8 GiB free recommended).")
                download_source_archive(archive, url=url)
            else:
                print("Reusing cached source archive:", archive, flush=True)
            print("Extracting original MPIIGaze files in the runtime...",
                  flush=True)
            # Publish the extracted directory only after a source-level check.
            with tempfile.TemporaryDirectory(
                    prefix=".mpiigaze_validate_", dir=raw_dir.parent) as tmp:
                candidate = Path(tmp) / "source"
                extract_source_archive(archive, candidate)
                containers = normalised_mat_files(candidate)
                if not containers or any(p.stat().st_size == 0 for p in containers):
                    raise ValueError(
                        "Archive has no complete-looking participant MAT containers")
                candidate.rename(raw_dir)
            print(f"Source available: {raw_dir}", flush=True)
            print(f"Normalised MAT containers: {len(containers):,}")
            return raw_dir
        except (OSError, ValueError, RuntimeError, EOFError,
                zipfile.BadZipFile, tarfile.TarError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print("Source acquisition attempt failed:", errors[-1], flush=True)
    raise RuntimeError(
        "Could not acquire the original MPIIGaze source from the official "
        "publisher or DaRUS. No source analysis has been performed. "
        "Check Colab network/disk access and retry. Details: "
        + " | ".join(errors))
