# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline contracts for Colab GazeBase acquisition and source integrity."""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eye_movement_modelling" / "code" / "src"))

from s2_acquisition import (
    TASKS, _inspect_csv, ensure_gazebase_preflight,
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


def test_drive_restore_and_preflight_writes_only_csv(tmp_path):
    raw = tmp_path / "raw"
    cache = tmp_path / "drive"
    cache.mkdir()
    for task in TASKS:
        path = source_csv_path(cache, 1001, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(csv_bytes())
    output = tmp_path / "output" / "01_source_acquisition_preflight.csv"
    result = ensure_gazebase_preflight(
        raw_root=raw, cache_root=cache,
        download_root=tmp_path / "unused",
        output_csv=output, requested_ids=(1001,),
    )
    assert len(result) == 4
    assert result.source_id.nunique() == 1
    assert set(result.task) == set(TASKS)
    assert output.exists()
    assert len(pd.read_csv(output)) == 4
    assert not (tmp_path / "unused").exists()


def test_incomplete_or_invalid_source_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("n,x,y,val\n0,0,1,1\n1,1,2,1\n")
    with pytest.raises(ValueError, match="No valid samples"):
        _inspect_csv(bad)
    with pytest.raises(ValueError, match="source ID"):
        source_csv_path(tmp_path, 900, "FXS")
