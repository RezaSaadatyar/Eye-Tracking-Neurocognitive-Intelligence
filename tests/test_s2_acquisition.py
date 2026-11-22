# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline contracts for Colab GazeBase acquisition and source integrity."""
from __future__ import annotations

import io
import hashlib
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eye_movement_modelling" / "code" / "src"))

from s2_acquisition import (
    TASKS, _inspect_csv, ensure_gazebase_preflight,
    ensure_gazebase_cohort_preflight, _archive_task_coverage,
    extract_selected_csvs, source_csv_path,
)


def csv_bytes() -> bytes:
    return b"n,x,y,val\n0,0,1,0\n1,1,2,0\n2,1,2,1\n"


def build_archive(path: Path, ids: tuple[int, ...], nested: bool) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for source_id in ids:
            if nested:
                data = io.BytesIO()
                with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as inner:
                    for task in TASKS:
                        inner.writestr(f"other/S_{source_id}_S1_{task}.csv",
                                       csv_bytes())
                out.writestr(
                    f"GazeBase_v2_0/Round_1/Subject_{source_id}.zip",
                    data.getvalue(),
                )
            else:
                for task in TASKS:
                    out.writestr(
                        f"Round_1/Subject_{source_id}/S_{source_id}_S1_{task}.csv",
                        csv_bytes(),
                    )
        out.writestr("Round_2/S_2001_S1_FXS.csv", csv_bytes())


@pytest.mark.parametrize("nested", [False, True])
def test_bounded_source_extraction_and_no_replacement(tmp_path, nested):
    archive = tmp_path / "source.zip"
    root = tmp_path / "raw"
    build_archive(archive, (1001, 1002, 1003), nested)
    protected = source_csv_path(root, 1001, "FXS")
    protected.parent.mkdir(parents=True)
    protected.write_bytes(csv_bytes())
    assert extract_selected_csvs(archive, root, (1001, 1002)) == 7
    assert protected.read_bytes() == csv_bytes()
    assert len(list(root.rglob("*.csv"))) == 8
    assert not list(root.rglob("*1003*"))
    assert _inspect_csv(protected)["n_samples"] == 3
    assert _inspect_csv(protected)["nonincreasing_time_steps"] == 0
    assert extract_selected_csvs(archive, root, (1001, 1002)) == 0


def test_colab_local_reuse_and_preflight_writes_only_csv(tmp_path):
    raw = tmp_path / "raw"
    for task in TASKS:
        path = source_csv_path(raw, 1001, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(csv_bytes())
    output = tmp_path / "output" / "01_source_acquisition_preflight.csv"
    result = ensure_gazebase_preflight(
        raw_root=raw, download_root=tmp_path / "unused",
        output_csv=output, requested_ids=(1001,),
    )
    assert len(result) == 4
    assert result.source_id.nunique() == 1
    assert set(result.task) == set(TASKS)
    assert output.exists()
    assert len(pd.read_csv(output)) == 4
    assert not (tmp_path / "unused").exists()
    assert len(list(raw.rglob("*.csv"))) == 4


def test_colab_local_preflight_extracts_only_missing_sources(tmp_path, monkeypatch):
    import s2_acquisition as acq

    fixture = tmp_path / "fixture.zip"
    build_archive(fixture, (1001, 1002, 1003), nested=True)
    raw = tmp_path / "raw"
    retained = source_csv_path(raw, 1001, "FXS")
    retained.parent.mkdir(parents=True)
    retained.write_bytes(csv_bytes())

    def use_offline_fixture(dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fixture.read_bytes())

    monkeypatch.setattr(acq, "_ensure_archive", use_offline_fixture)
    output = tmp_path / "output" / "01_source_acquisition_preflight.csv"
    result = ensure_gazebase_preflight(
        raw_root=raw, download_root=tmp_path / "temp_download",
        output_csv=output, requested_ids=(1001, 1002),
    )
    assert len(result) == 8
    assert set(result.source_id) == {1001, 1002}
    assert len(list(raw.rglob("*.csv"))) == 8
    assert retained.read_bytes() == csv_bytes()
    assert not list(raw.rglob("*1003*"))
    assert not (tmp_path / "temp_download" / "GazeBase_v2_0.zip").exists()
    assert len(pd.read_csv(output)) == 8


def test_incomplete_or_invalid_source_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("n,x,y,val\n0,0,1,1\n1,1,2,1\n")
    with pytest.raises(ValueError, match="No valid samples"):
        _inspect_csv(bad)
    with pytest.raises(ValueError, match="source ID"):
        source_csv_path(tmp_path, 900, "FXS")

class _FakeHTTPResponse:
    """Minimal URL response compatible with urllib's context manager."""
    def __init__(self, data, status=200, headers=None):
        self.buffer = io.BytesIO(data)
        self.status = status
        self.headers = headers or {"Content-Type": "application/zip"}
    def read(self, amount=-1):
        return self.buffer.read(amount)
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.buffer.close()


def test_fallback_after_successful_zero_length_http_status(tmp_path, monkeypatch):
    """A misleading 200/0-body response must never be treated as success."""
    import s2_acquisition as acq
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as bundle:
        bundle.writestr("test.csv", csv_bytes())
    payload = data.getvalue()
    monkeypatch.setattr(acq, "ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(acq, "ARCHIVE_MD5", hashlib.md5(payload).hexdigest())
    monkeypatch.setattr(acq, "FIGSHARE_DOWNLOAD_URLS",
                        ("https://example.test/empty", "https://example.test/good"))
    monkeypatch.setattr(acq, "VERIFIED_ARCHIVE_MIRROR",
                        "https://example.test/mirror")
    called = []
    def fake_urlopen(request, timeout):
        called.append(request.full_url)
        if request.full_url.endswith("/empty"):
            return _FakeHTTPResponse(
                b"", headers={"Content-Type": "application/zip",
                              "Content-Length": "0"}
            )
        return _FakeHTTPResponse(payload)
    monkeypatch.setattr(acq.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "source.zip"
    acq._ensure_archive(dest)
    assert dest.read_bytes() == payload
    assert len(called) == 2
    assert not (tmp_path / "source.zip.part").exists()


def test_all_empty_sources_report_actionable_error(tmp_path, monkeypatch):
    import s2_acquisition as acq
    monkeypatch.setattr(acq, "ARCHIVE_BYTES", 100)
    monkeypatch.setattr(acq, "FIGSHARE_DOWNLOAD_URLS",
                        ("https://example.test/official",))
    monkeypatch.setattr(acq, "VERIFIED_ARCHIVE_MIRROR",
                        "https://example.test/mirror")
    def no_bytes(request, timeout):
        return _FakeHTTPResponse(b"", headers={"Content-Type": "application/zip"})
    monkeypatch.setattr(acq.urllib.request, "urlopen", no_bytes)
    with pytest.raises(OSError, match="All GazeBase download endpoints failed"):
        acq._ensure_archive(tmp_path / "source.zip")
    assert not (tmp_path / "source.zip.part").exists()


def test_partial_download_resumes_only_with_matching_content_range(
        tmp_path, monkeypatch):
    import s2_acquisition as acq
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as bundle:
        bundle.writestr("test.csv", csv_bytes())
    payload = data.getvalue()
    partial = tmp_path / "source.zip.part"
    partial.write_bytes(payload[:35])
    monkeypatch.setattr(acq, "ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(acq, "ARCHIVE_MD5", hashlib.md5(payload).hexdigest())
    monkeypatch.setattr(acq, "FIGSHARE_DOWNLOAD_URLS",
                        ("https://example.test/resume",))
    monkeypatch.setattr(acq, "VERIFIED_ARCHIVE_MIRROR",
                        "https://example.test/mirror")
    def fake_urlopen(request, timeout):
        assert request.headers.get("Range") == "bytes=35-"
        return _FakeHTTPResponse(
            payload[35:], status=206,
            headers={
                "Content-Type": "application/zip",
                "Content-Range": f"bytes 35-{len(payload)-1}/{len(payload)}",
            },
        )
    monkeypatch.setattr(acq.urllib.request, "urlopen", fake_urlopen)
    acq._ensure_archive(tmp_path / "source.zip")
    assert (tmp_path / "source.zip").read_bytes() == payload


@pytest.mark.parametrize("nested", [False, True])
def test_cohort_preflight_fills_gaps_without_reducing_target(
    tmp_path, monkeypatch, capsys, nested,
):
    """A verified archive can skip missing IDs and still select ten people."""
    import s2_acquisition as acq

    fixture = tmp_path / "fixture.zip"
    ids = tuple(i for i in range(1001, 1013) if i not in (1003, 1005))
    build_archive(fixture, ids, nested=nested)
    root = tmp_path / "raw"
    # Simulate originals retained from an earlier partial extraction.
    for task in TASKS:
        stored = source_csv_path(root, 1001, task)
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_bytes(csv_bytes())

    def offline_archive(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fixture.read_bytes())

    monkeypatch.setattr(acq, "_ensure_archive", offline_archive)
    out = tmp_path / "output" / "01_source_acquisition_preflight.csv"
    records = ensure_gazebase_cohort_preflight(
        raw_root=root, download_root=tmp_path / "download",
        output_csv=out, participants=10,
    )
    assert len(records) == 40
    assert set(records.source_id) == set(ids)
    assert out.is_file()
    assert len(pd.read_csv(out)) == 40
    assert "1003" in capsys.readouterr().out
    assert not (tmp_path / "download" / "GazeBase_v2_0.zip").exists()
    assert len(list(root.rglob("*.csv"))) == 40


def test_incomplete_cohort_keeps_verified_archive_for_retry(
    tmp_path, monkeypatch,
):
    """Do not delete six gigabytes of verified data on eligibility failures."""
    import s2_acquisition as acq

    fixture = tmp_path / "fixture.zip"
    build_archive(fixture, tuple(range(1001, 1010)), nested=True)
    def offline_archive(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fixture.read_bytes())
    monkeypatch.setattr(acq, "_ensure_archive", offline_archive)
    archive = tmp_path / "download" / "GazeBase_v2_0.zip"
    with pytest.raises(FileNotFoundError, match="only 9 people"):
        ensure_gazebase_cohort_preflight(
            raw_root=tmp_path / "raw", download_root=archive.parent,
            output_csv=tmp_path / "output" / "01_source_acquisition_preflight.csv",
            participants=10,
        )
    assert archive.read_bytes() == fixture.read_bytes()


def test_exact_id_extraction_failure_keeps_zip(tmp_path, monkeypatch):
    """Preserve the verified archive even through the previous exact-ID API."""
    import s2_acquisition as acq

    fixture = tmp_path / "fixture.zip"
    build_archive(fixture, (1001,), nested=False)
    def offline_archive(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fixture.read_bytes())
    monkeypatch.setattr(acq, "_ensure_archive", offline_archive)
    archive = tmp_path / "temp" / "GazeBase_v2_0.zip"
    with pytest.raises(FileNotFoundError, match="still missing"):
        ensure_gazebase_preflight(
            raw_root=tmp_path / "raw", download_root=archive.parent,
            output_csv=tmp_path / "output" / "01_source_acquisition_preflight.csv",
            requested_ids=(1001, 1002),
        )
    assert archive.is_file()


def test_archive_index_excludes_other_round_and_incomplete_tasks(tmp_path):
    """Round-2 source files cannot qualify a missing Round-1 task."""
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for task in TASKS:
            bundle.writestr(
                f"Round_1/Subject_1001/S_1001_S1_{task}.csv", csv_bytes(),
            )
        for task in TASKS[:-1]:
            bundle.writestr(
                f"Round_1/Subject_1002/S_1002_S1_{task}.csv", csv_bytes(),
            )
        bundle.writestr(
            "Round_2/Subject_1002/S_1002_S1_TEX.csv", csv_bytes(),
        )
    indexed = _archive_task_coverage(archive)
    assert indexed[1001] == set(TASKS)
    assert indexed[1002] == set(TASKS[:-1])
