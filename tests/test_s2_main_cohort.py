# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Synthetic 100/300-person cohort and current-stage handoff contracts.

No Figshare download, model training, or new Test evaluation is performed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_modules"))
sys.path.insert(0, str(ROOT / "eye_movement_modelling" / "code" / "src"))

from s2_data import participant_split, save_protocol, load_protocol
from s2_handoff import ensure_s2_stage_inputs, verify_main_cohort


TASKS = ("FXS", "HSS", "RAN", "TEX")


def _source_manifest(size: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"record_id": f"S_{1000+i}_S1_{task}",
         "participant_id": f"{i:03d}", "task": task}
        for i in range(1, size+1) for task in TASKS
    ])


@pytest.mark.parametrize(
    ("n", "train", "validation", "test"),
    [(100, 64, 16, 20), (300, 192, 48, 60)],
)
def test_large_cohort_grouping_and_reserved_main_test(
    n, train, validation, test,
):
    inventory = _source_manifest(n)
    split = participant_split(
        inventory, seed=17, test_fraction=0.20,
        validation_fraction=0.20,
    )
    assert len(split) == 4*n
    assert split["record_id"].is_unique
    assert split["study_purpose"].eq("main").all()
    assert split.groupby("participant_id")["split"].nunique().eq(1).all()
    protocol = {"purpose": "main", "max_participants": n}
    counts = verify_main_cohort(protocol, split)
    assert counts == {"train": train, "validation": validation, "test": test}
    actual_test = sorted(split.loc[
        split["split"].eq("test"), "participant_id"
    ].unique())
    assert actual_test == [f"{j:03d}" for j in range(n-test+1, n+1)]
    assert min(map(int, actual_test)) > 10
    pd.testing.assert_frame_equal(
        split, participant_split(
            inventory, seed=17, test_fraction=0.20,
            validation_fraction=0.20,
        ),
    )


def test_stale_pilot_cohort_rejected_without_main_archive():
    split = participant_split(_source_manifest(10))
    with pytest.raises(ValueError, match="current main cohort"):
        verify_main_cohort({"purpose": "pilot", "max_participants": 10}, split)


def test_uploaded_stage_zip_overrides_stale_tracked_pilot_data(tmp_path):
    folder = tmp_path / "checkout" / "eye_movement_modelling" / "output"
    folder.mkdir(parents=True)
    protocol = folder / "01_protocol.csv"
    protocol.write_text("old pilot data", encoding="utf-8")
    current = "dataset,round_id\nGazeBase_v2_0,1\n"
    name = "eye_movement_modelling_Notebook_01_Outputs.zip"
    archive = tmp_path / name
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
        zip_file.writestr("01_protocol.csv", current)
        zip_file.writestr("01_split_assignments.csv", "main cohort splits")
    restored = ensure_s2_stage_inputs(
        folder, 1, ("01_protocol.csv", "01_split_assignments.csv"),
        upload_dir=tmp_path,
    )
    assert len(restored) == 2
    assert protocol.read_text(encoding="utf-8") == current
    assert (folder / "01_split_assignments.csv").read_text(
        encoding="utf-8"
    ) == "main cohort splits"
    assert ensure_s2_stage_inputs(
        folder, 1, ("01_protocol.csv",), upload_dir=tmp_path,
    ) == [protocol]


def test_stage_zip_missing_falls_back_only_to_complete_local_files(tmp_path):
    folder = tmp_path / "checkout" / "eye_movement_modelling" / "output"
    folder.mkdir(parents=True)
    filename = folder / "01_protocol.csv"
    filename.write_text("main", encoding="utf-8")
    assert ensure_s2_stage_inputs(
        folder, 1, ("01_protocol.csv",), upload_dir=tmp_path,
    ) == [filename]
    with pytest.raises(FileNotFoundError, match="Missing stage 01"):
        ensure_s2_stage_inputs(
            folder, 1, ("01_split_assignments.csv",), upload_dir=tmp_path,
        )


def test_new_protocol_roundtrip_retains_main_cohort_configuration(tmp_path):
    path = tmp_path / "01_protocol.csv"
    saved = save_protocol(
        path, participants=300, seed=17, window_samples=256,
        max_windows_per_record=24, purpose="main",
        test_fraction=0.20, validation_fraction=0.20,
        source_first_id=1001,
    )
    assert load_protocol(path) == saved
    assert saved["source_last_id"] == 1300


def test_all_five_current_notebooks_draft_main_not_pilot_results():
    folder = ROOT / "eye_movement_modelling" / "code" / "notebooks"
    files = sorted(folder.glob("0[1-5]_*.ipynb"))
    assert len(files) == 5
    for notebook in files:
        n = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in n["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert not cell["outputs"]
    first = json.loads(files[0].read_text(encoding="utf-8"))
    config = next(
        c for c in first["cells"] if c.get("id") == "s2-c-cohort-config"
    )
    source = "".join(config["source"])
    assert "MAX_PARTICIPANTS = 100" in source
    assert "FIRST_SOURCE_ID = 1001" in source
    assert "MAX_WINDOWS_PER_RECORD = 24" in source
    for notebook in files[1:]:
        source = "\n".join(
            "".join(c["source"]) for c in
            json.loads(notebook.read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"
        )
        assert "ensure_s2_stage_inputs(" in source
        assert "verify_main_cohort(protocol, split)" in source
        assert "drive.mount" not in source
