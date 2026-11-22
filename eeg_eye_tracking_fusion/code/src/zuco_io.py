"""Minimal, release-pinned ZuCo 2.0 BIDS discovery and file retrieval.

The NEMAR data gateway redirects each requested BIDS path to its content;
GitHub's nemarDatasets repository is used only as a public file index. No
DataLad, cloud drive, archive extraction or dataset-wide download is needed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATASET_ID = "nm000279"
VERSION = "v1.0.0"
API = f"https://api.github.com/repos/nemarDatasets/{DATASET_ID}/contents"
DATA = f"https://data.nemar.org/{DATASET_ID}/{VERSION}"
S3 = f"https://nemar.s3.us-east-2.amazonaws.com/{DATASET_ID}/objects"
TASKS = ("nr", "tsr")
_TRANSIENT_STATUSES = frozenset((429, 500, 502, 503, 504))


def session() -> requests.Session:
    """Return one HTTP session with bounded retries for transient failures."""
    handle = requests.Session()
    handle.headers.update({"User-Agent": "zuco-2-research-notebook/1.0"})
    retry = Retry(
        total=3, connect=3, read=2, status=3, backoff_factor=1.0,
        status_forcelist=_TRANSIENT_STATUSES, allowed_methods={"GET"},
        respect_retry_after_header=True, raise_on_status=False,
    )
    handle.mount("https://", HTTPAdapter(max_retries=retry))
    return handle


def metadata_table(relative_path: str, handle=None) -> pd.DataFrame:
    """Read small BIDS metadata without downloading any signal data."""
    handle = handle or session()
    response = handle.get(f"{DATA}/{relative_path}", timeout=90)
    response.raise_for_status()
    if relative_path.endswith(".tsv"):
        from io import StringIO
        return pd.read_csv(StringIO(response.text), sep="\t")
    raise ValueError(f"Expected a TSV metadata table: {relative_path}")


def list_subject_files(subject: str, handle=None) -> list[str]:
    """List BIDS EEG and eye-data paths for a real participant."""
    if not re.fullmatch(r"sub-[A-Za-z0-9]+", subject):
        raise ValueError("Invalid BIDS subject identifier")
    handle = handle or session()
    response = handle.get(f"{API}/{subject}/eeg", timeout=30)
    response.raise_for_status()
    return [item["path"] for item in response.json()
            if item.get("type") == "file"]


def select_run_paths(paths: list[str], subject: str,
                     runs_per_task: int = 1) -> list[dict]:
    """Select an equal, deterministic number of NR and TSR runs.

    Only complete sets with BrainVision, events and eye-physiology files
    are eligible; metadata are listed independently for transparent audit.
    """
    if runs_per_task < 1:
        raise ValueError("runs_per_task must be positive")
    present = set(paths)
    selected = []
    for task in TASKS:
        candidates = sorted({int(m.group(1)) for p in paths
                             if (m := re.search(
                                 rf"{subject}_task-{task}_run-(\d+)_eeg\.json$", p
                             ))})
        valid = []
        for run in candidates:
            stem = f"{subject}/eeg/{subject}_task-{task}_run-{run}"
            required = {
                "eeg": f"{stem}_eeg.eeg",
                "header": f"{stem}_eeg.vhdr",
                "markers": f"{stem}_eeg.vmrk",
                "eeg_json": f"{stem}_eeg.json",
                "events": f"{stem}_events.tsv",
                "channels": f"{stem}_channels.tsv",
                "eye": f"{stem}_recording-eyetrack_physio.tsv.gz",
                "eye_json": f"{stem}_recording-eyetrack_physio.json",
            }
            if set(required.values()).issubset(present):
                valid.append({"participant_id": subject, "task": task,
                              "run": run, **required})
        if len(valid) < runs_per_task:
            return []
        selected.extend(valid[:runs_per_task])
    return selected


def discover_cohort(max_participants: int = 6,
                    runs_per_task: int = 1,
                    handle=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover complete subject/task/run pairs in source participant order."""
    handle = handle or session()
    population = metadata_table("participants.tsv", handle)
    selected = []
    eligibility = []
    for subject in population["participant_id"].astype(str):
        try:
            paths = list_subject_files(subject, handle)
            runs = select_run_paths(paths, subject, runs_per_task)
            eligible = len(runs) == 2 * runs_per_task
            reason = "complete" if eligible else "missing paired runs"
        except requests.RequestException as error:
            raise RuntimeError(f"Source file listing failed: {subject}") from error
        eligibility.append({"participant_id": subject,
                            "eligible": eligible, "reason": reason})
        if eligible and len({r["participant_id"] for r in selected}) < max_participants:
            selected.extend(runs)
    if not selected:
        raise RuntimeError("No matched NR/TSR runs found in the source")
    return pd.DataFrame(selected), pd.DataFrame(eligibility)


def split_people(ids, seed: int = 17, test_fraction: float = 0.2,
                 validation_fraction: float = 0.2) -> pd.DataFrame:
    """Disjoint person-level roles; never include subject ID as a feature."""
    ids = sorted(set(map(str, ids)))
    if len(ids) < 5:
        raise ValueError("At least five eligible people are needed for roles")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ids))
    n_test = max(1, int(np.ceil(test_fraction * len(ids))))
    n_val = max(1, int(np.ceil(validation_fraction * (len(ids)-n_test))))
    roles = {}
    for subject in shuffled[:n_test]:
        roles[subject] = "test"
    for subject in shuffled[n_test:n_test+n_val]:
        roles[subject] = "validation"
    for subject in shuffled[n_test+n_val:]:
        roles[subject] = "train"
    if "train" not in roles.values():
        raise ValueError("No participants remain for training")
    return pd.DataFrame({"participant_id": ids,
                         "split": [roles[s] for s in ids]})


def _annex_object(relative_path: str, handle) -> tuple[str, int, str]:
    """Resolve a source Git-annex pointer to its public, content-addressed S3 URL."""
    index = handle.get(f"{API}/{relative_path}", timeout=(15, 45))
    index.raise_for_status()
    item = index.json()
    pointer = item.get("target", "")
    if not pointer and item.get("content"):
        pointer = base64.b64decode(item["content"]).decode("utf-8").strip()
    key = pointer.rsplit("/", 1)[-1]
    match = re.fullmatch(
        r"SHA256E-s([0-9]+)--([0-9a-f]{64})(?:\.[a-zA-Z0-9.]+)?", key
    )
    if not pointer.startswith("/annex/objects/") or not match:
        raise ValueError(
            f"Missing verifiable Git-annex source pointer: {relative_path}"
        )
    return f"{S3}/{key}", int(match.group(1)), match.group(2)


def _stream_source(url: str, temporary: Path, handle,
                   expected_size=None, expected_sha256=None) -> None:
    """Download into a temporary file and verify optional annex integrity."""
    digest = hashlib.sha256() if expected_sha256 else None
    size = 0
    with handle.get(url, timeout=(30, 180), stream=True,
                    allow_redirects=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)
                    size += len(chunk)
                    if digest is not None:
                        digest.update(chunk)
    if not size or (expected_size is not None and size != expected_size):
        raise IOError(f"Empty or incomplete source file: {url}")
    if digest is not None and digest.hexdigest() != expected_sha256:
        raise IOError(f"SHA-256 mismatch for source file: {url}")


def download_file(relative_path: str, raw_root: Path, handle=None) -> Path:
    """Retrieve one BIDS source file using NEMAR, then its official S3 copy.

    The source gateway is preferred. Only transient gateway failures trigger
    the content-addressed S3 fallback; successful files remain in the cache.
    """
    handle = handle or session()
    relative_path = str(relative_path).lstrip("/")
    if ".." in Path(relative_path).parts:
        raise ValueError("Unsafe dataset relative path")
    destination = raw_root / relative_path
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        try:
            _stream_source(f"{DATA}/{relative_path}", temporary, handle)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RetryError):
            temporary.unlink(missing_ok=True)
            fallback = True
        except requests.exceptions.HTTPError as error:
            if (error.response is None
                    or error.response.status_code not in _TRANSIENT_STATUSES):
                raise
            temporary.unlink(missing_ok=True)
            fallback = True
        else:
            fallback = False

        if fallback:
            try:
                url, size, digest = _annex_object(relative_path, handle)
            except (requests.RequestException, ValueError) as error:
                raise RuntimeError(
                    f"NEMAR is temporarily unavailable and its public S3 "
                    f"source could not be resolved: {relative_path}"
                ) from error
            print(f"NEMAR gateway unavailable; trying official S3: "
                  f"{Path(relative_path).name}")
            try:
                _stream_source(url, temporary, handle, size, digest)
            except (requests.RequestException, OSError) as error:
                raise RuntimeError(
                    f"Neither NEMAR gateway nor public S3 served the "
                    f"verified source file: {relative_path}. "
                    "Keep downloaded files in /content and try again "
                    "when the source service is available."
                ) from error

        if (destination.suffix in (".eeg", ".gz")
                and temporary.stat().st_size < 512):
            raise IOError(f"Source file is too small: {relative_path}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def fetch_run(row: pd.Series | dict, raw_root: Path,
              handle=None) -> dict[str, Path]:
    """Retrieve just the eight original files belonging to a selected run."""
    keys = ("eeg", "header", "markers", "eeg_json", "events",
            "channels", "eye", "eye_json")
    return {key: download_file(row[key], raw_root, handle) for key in keys}


def read_sidecar(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def open_run(row, raw_root: Path, handle=None):
    """Return an MNE lazy BrainVision raw, eye stream and event metadata."""
    import mne
    files = fetch_run(row, raw_root, handle)
    raw = mne.io.read_raw_brainvision(files["header"], preload=False,
                                     verbose="ERROR")
    eye = pd.read_csv(files["eye"], sep="\t", compression="gzip",
                      na_values=["n/a", "NA", "NaN", ""])
    events = pd.read_csv(files["events"], sep="\t", na_values="n/a")
    eye_meta = read_sidecar(files["eye_json"])
    eeg_meta = read_sidecar(files["eeg_json"])
    return raw, eye, events, eye_meta, eeg_meta
