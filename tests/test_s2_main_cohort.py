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
    with pytest.raises(RuntimeError, match="Incomplete stage 01"):
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


def test_current_notebooks_preserve_real_execution_and_main_input_contract():
    """Keep authentic Stage-01 outputs and check unexecuted model notebook code."""
    folder = ROOT / "eye_movement_modelling" / "code" / "notebooks"
    files = sorted(folder.glob("0[1-5]_*.ipynb"))
    assert len(files) == 5
    for notebook in files:
        n = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in n["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert not [
                item for item in cell.get("outputs", [])
                if item.get("output_type") == "error"
            ]
            if cell["execution_count"] is not None:
                assert isinstance(cell["execution_count"], int)

    first = json.loads(files[0].read_text(encoding="utf-8"))
    by_id = {c.get("id"): c for c in first["cells"]}
    source = "".join(by_id["s2-c-cohort-config"]["source"])
    assert "MAX_PARTICIPANTS = 100" in source
    assert "FIRST_SOURCE_ID = 1001" in source
    assert "MAX_WINDOWS_PER_RECORD = 24" in source
    assert isinstance(by_id["s2-c-017"]["execution_count"], int)
    assert by_id["s2-c-017"]["outputs"]
    assert "ensure_gazebase_cohort_preflight(" in "".join(
        by_id["s2-c-source-acquisition"]["source"]
    )
    assert "selected_people =" in "".join(by_id["s2-c-007"]["source"])

    for notebook in files[1:]:
        source = "\n".join(
            "".join(cell["source"]) for cell in
            json.loads(notebook.read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code"
        )
        assert "ensure_s2_stage_inputs(" in source
        assert "verify_main_cohort(protocol, split)" in source
        assert "drive.mount" not in source


def test_final_main_stage_checks_exact_validation_source_identifiers():
    """A stale model with the same number of people must still be rejected."""
    notebook = (
        ROOT / "eye_movement_modelling" / "code" / "notebooks"
        / "05_robustness_and_final_evaluation.ipynb"
    )
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    recovery = next(c for c in cells if c.get("id") == "s2c7")
    code = "".join(recovery["source"])
    for stage, name in (
        ("02", "class_prior"),
        ("02", "logistic_regression"),
        ("02", "random_forest"),
        ("02", "gradient_boosting"),
        ("03", "cnn"), ("03", "tcn"),
        ("04", "gru"), ("04", "transformer"),
    ):
        assert f'("{name}", "{stage}")' in code
    assert "pred.record_id.duplicated().any()" in code
    assert "set(pred.record_id) != expected_val_ids" in code
    assert "pred.record_id.map(target_lookup)" in code
    assert "do not mix model checkpoints" in code


def test_sparse_selected_ids_roundtrip_and_downstream_reconciliation(tmp_path):
    source = tuple(i for i in range(1001, 1107) if i not in (
        1028, 1036, 1044, 1055, 1070, 1091,
    ))
    assert len(source) == 100
    saved = save_protocol(
        tmp_path / "01_protocol.csv", participants=100, seed=17,
        window_samples=256, max_windows_per_record=24,
        purpose="main", selected_source_ids=source,
    )
    recovered = load_protocol(tmp_path / "01_protocol.csv")
    assert recovered == saved
    assert recovered["source_last_id"] == 1106
    assert len(recovered["selected_source_ids"]) == 100
    split = participant_split(pd.DataFrame([
        {"participant_id": f"{sid % 1000:03d}",
         "record_id": f"S_{sid}_S1_{task}", "task": task}
        for sid in source for task in TASKS
    ]))
    assert verify_main_cohort(recovered, split) == {
        "train": 64, "validation": 16, "test": 20,
    }
    tampered = split.copy()
    tampered["participant_id"] = tampered["participant_id"].replace(
        {"106": "107"}
    )
    with pytest.raises(ValueError, match="Protocol-selected"):
        verify_main_cohort(recovered, tampered)


def test_stage01_figure_matches_observed_cohort_and_stage02_is_prospective():
    """Both editable figures retain measured gaze but avoid copied pilot scores."""
    import xml.etree.ElementTree as ET

    base = ROOT / "eye_movement_modelling" / "images"
    first = ET.parse(base / "01_data_and_experimental_protocol.svg")
    second = ET.parse(base / "02_statistical_and_classical_ml.svg")
    def component(document, identifier):
        return next(el for el in document.iter()
                    if el.attrib.get("id") == identifier)
    assert ET.tostring(component(first, "source-record")) == ET.tostring(
        component(second, "source-record")
    )
    figure1 = " ".join(el.text or "" for el in first.iter())
    figure2 = " ".join(el.text or "" for el in second.iter())
    for expected in ("100 complete people", "64 Train participants",
                     "16 Validation people", "20 reserved Test people"):
        assert expected in figure1
    assert "256 Train recordings" in figure2
    assert "64 records; 16 people" in figure2
    assert "Main Validation pending" in figure2
    assert "8/8" not in figure2 and "Macro F1 1.00" not in figure2
    for document in (first, second):
        for i in range(1, 6):
            node = component(document, f"node-{i}")
            rect = next(el for el in node if el.tag.endswith("rect"))
            assert rect.attrib.get("fill") == "#F7FAFD"
            assert rect.attrib.get("stroke") == "#B8CFDE"
            assert len([el for el in node
                        if el.attrib.get("class") == "nodedesc"]) == 3


def test_stage02_requires_current_validation_before_its_final_zip():
    notebook = (
        ROOT / "eye_movement_modelling" / "code"
        / "notebooks" / "02_statistical_and_classical_ml.ipynb"
    )
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    by_id = {c["id"]: c for c in cells}
    code = "".join(by_id["s2c19"]["source"])
    assert "required_stage02" in code
    assert "saved.record_id.duplicated().any()" in code
    assert "set(saved.record_id) != expected_ids" in code
    assert "expected_labels.sort_index()" in code
    assert "download_outputs_from_colab" in code
    assert all(c["execution_count"] is None and not c["outputs"]
               for c in cells if c["cell_type"] == "code")


def test_stage01_saved_source_tables_agree_with_current_protocol():
    """Check the observed main-cohort registry and derived record identities."""
    from s2_data import load_protocol

    folder = ROOT / "eye_movement_modelling" / "output"
    protocol = load_protocol(folder / "01_protocol.csv")
    audit = pd.read_csv(folder / "01_source_acquisition_preflight.csv")
    inventory = pd.read_csv(
        folder / "01_source_inventory.csv",
        dtype={"record_id": str, "participant_id": str},
    )
    split = pd.read_csv(
        folder / "01_split_assignments.csv",
        dtype={"record_id": str, "participant_id": str},
    )
    features = pd.read_csv(
        folder / "01_record_features.csv",
        dtype={"record_id": str, "participant_id": str},
    )
    eligible = set(protocol["selected_source_ids"])
    assert protocol["purpose"] == "main"
    assert len(eligible) == protocol["max_participants"]
    assert set(audit["source_id"]) == eligible
    assert len(audit) == 4 * len(eligible)
    for dataframe in (audit, inventory, split, features):
        assert dataframe["record_id"].is_unique
    ids = set(audit["record_id"])
    assert all(set(frame["record_id"]) == ids
               for frame in (inventory, split, features))
    assert split.groupby("participant_id")["split"].nunique().eq(1).all()
    assert split.groupby("participant_id")["task"].nunique().eq(4).all()
    assert audit["nonincreasing_time_steps"].eq(0).all()
    assert audit["valid_fraction"].between(0, 1).all()
    assert features["n_windows"].gt(0).all()
    verify_main_cohort(protocol, split)

    # The current 100-person compressed windows are tracked; a later larger
    # cohort may keep its NPZ outside Git under the independent size policy.
    npz = folder / "01_sequence_windows.npz"
    if npz.exists():
        import numpy as np

        with np.load(npz) as archive:
            x = archive["x"]
            y = archive["y"]
            record_ids = archive["record_id"].astype(str)
            assert x.shape == (
                int(features["n_windows"].sum()),
                protocol["window_samples"],
                len(protocol["temporal_channels"]),
            )
            assert len(record_ids) == len(y) == len(x)
            assert set(record_ids) == ids
            assert np.isfinite(x).all()
            expected = dict(
                zip(inventory["record_id"], inventory["task"].map(
                    {task: i for i, task in enumerate(TASKS)}
                ))
            )
            assert np.array_equal(
                y, np.asarray([expected[record] for record in record_ids])
            )
