# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Offline Notebook-stage ZIP handoff and filename compatibility checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_modules"))

from core import output_download
from core.stage_outputs import ensure_stage_inputs


def test_notebook_01_zip_round_trip_uses_named_files(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    out = checkout / "eye_movement_modelling" / "output"
    out.mkdir(parents=True)
    required = (
        "01_source_inventory.csv", "01_split_assignments.csv",
        "01_record_features.csv", "01_sequence_windows.npz",
        "01_sequence_normalisation.csv", "01_protocol.csv",
        "01_source_distribution.png", "01_source_acquisition_preflight.csv",
        "01_source_acquisition_task_summary.csv",
    )
    needed_by_next_notebook = (
        "01_record_features.csv", "01_split_assignments.csv",
        "01_protocol.csv", "01_sequence_windows.npz",
    )
    for name in required:
        (out / name).write_bytes(("sample:" + name).encode("ascii"))
    monkeypatch.setattr(output_download, "REPO_ROOT", checkout)
    output_download.download_outputs_from_colab(
        "eye_movement_modelling/output", "01"
    )
    archive = tmp_path / "eye_movement_modelling_Notebook_01_Outputs.zip"
    assert archive.is_file()
    with ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(required)
    for name in required:
        (out / name).unlink()
    restored = ensure_stage_inputs(
        out, 1, needed_by_next_notebook, upload_dir=tmp_path,
    )
    assert len(restored) == len(needed_by_next_notebook)
    assert all(x.is_file() for x in restored)
    assert all((out / name).is_file() for name in required)


def test_notebook_01_acquisition_inside_standard_section_02():
    path = (ROOT / "eye_movement_modelling" / "code" / "notebooks"
            / "01_data_and_experimental_protocol.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    ids = [c.get("id") for c in cells]
    assert "emm-01a-acquire-code" not in ids
    assert "emm-01a-acquire-md" not in ids
    section = "".join(cells[ids.index("s2-m-006")]["source"])
    assert "02. Configure, Acquire and Select Original GazeBase" in section
    config_idx = ids.index("s2-c-cohort-config")
    acquisition_idx = ids.index("s2-c-source-acquisition")
    assert config_idx == ids.index("s2-m-006") + 1
    assert acquisition_idx == config_idx + 1
    assert ids[acquisition_idx + 1] == "s2-c-007"
    config = "".join(cells[config_idx]["source"])
    acquire = "".join(cells[acquisition_idx]["source"])
    selection = "".join(cells[ids.index("s2-c-007")]["source"])
    assert "MAX_PARTICIPANTS = 100" in config
    assert "FIRST_SOURCE_ID = 1001" in config
    assert "Actual source IDs will be selected" in config
    assert "ensure_gazebase_cohort_preflight(" in acquire
    assert "selected_people =" in selection
    assert "drive.mount" not in acquire
    assert "ZipFile" not in acquire
    end = "".join(cells[ids.index("s2-c-017")]["source"])
    for required in (
        "01_source_inventory.csv", "01_split_assignments.csv",
        "01_record_features.csv", "01_sequence_windows.npz",
        "01_sequence_normalisation.csv", "01_protocol.csv",
        "01_source_distribution.png", "01_source_acquisition_preflight.csv",
        "01_source_acquisition_task_summary.csv",
    ):
        assert required in end
    assert ('download_outputs_from_colab('
            '"eye_movement_modelling/output", "01")') in end
    assert "missing_outputs" in end



def test_temporal_stage_artifact_prefixes_match_final_evaluation():
    notebooks = ROOT / "eye_movement_modelling" / "code" / "notebooks"
    for nbname, stage, names in (
        ("03_cnn_and_tcn_models.ipynb", "03",
         ("cnn", "tcn")),
        ("04_gru_and_transformer_models.ipynb", "04",
         ("gru", "transformer")),
    ):
        nb = json.loads((notebooks / nbname).read_text(encoding="utf-8"))
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        code = "\n".join("".join(c["source"]) for c in cells)
        assert (f'download_outputs_from_colab('
                f'"eye_movement_modelling/output", "{stage}")') in code
        assert f'"{stage}_neural_validation_metrics.csv"' in code
        for name in names:
            assert f'f"{stage}_{name}.pt"' in code
            assert f'f"{stage}_{name}_history.csv"' in code
            assert f'f"{stage}_{name}_validation_predictions.csv"' in code
            assert f'f"{stage}_{name}_evaluation.png"' in code
        assert f'"{stage}_model_comparison.png"' in code
        assert f'"{int(stage)}_' not in code
    evaluation = json.loads(
        (notebooks / "05_robustness_and_final_evaluation.ipynb").read_text(
            encoding="utf-8"
        )
    )
    eval_code = "\n".join(
        "".join(c["source"]) for c in evaluation["cells"]
        if c["cell_type"] == "code"
    )
    for stage in ("03", "04"):
        assert f'ensure_s2_stage_inputs(output_dir, {int(stage)}, (' in eval_code
        assert f'"{stage}_neural_validation_metrics.csv"' in eval_code

def test_temporal_model_complexity_is_saved_with_validation_metrics():
    """Preserve selected epochs and parameter counts for each neural model."""
    notebook = (ROOT / "eye_movement_modelling" / "code" / "notebooks"
                / "03_cnn_and_tcn_models.ipynb")
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for name, code_id in (("cnn", "s2c9"), ("tcn", "s2c11")):
        cell = next(c for c in cells if c.get("id") == code_id)
        source = "".join(cell["source"])
        assert f'model_name = "{name}"' in source
        assert 'result["parameters"] = int(info["parameters"])' in source
        assert 'result["best_epoch"] = int(info["best_epoch"])' in source
        assert cell["execution_count"] is None
        assert not cell["outputs"]
    comparison = next(c for c in cells if c.get("id") == "s2c13")
    assert '"03_neural_validation_metrics.csv"' in "".join(comparison["source"])


def test_study_overview_reflects_latest_source_audit():
    import xml.etree.ElementTree as ET

    path = ROOT / "eye_movement_modelling" / "images" / "study_overview.svg"
    root = ET.parse(path).getroot()
    labels = " ".join((item.text or "") for item in root.iter())
    assert "100 people audited" in labels
    assert "Main models not yet fitted" in labels
    assert "20/30 requires more source CSVs" not in labels


def test_stage03_svg_workflow_matches_stage02_reference():
    """The measured source graphic and workflow card style are preserved."""
    import xml.etree.ElementTree as ET
    folder = ROOT / "eye_movement_modelling" / "images"
    def element(filename, identifier):
        root = ET.parse(folder / filename).getroot()
        return next(x for x in root.iter() if x.get("id") == identifier)
    source2 = element("02_statistical_and_classical_ml.svg", "source-record")
    source3 = element("03_cnn_and_tcn_models.svg", "source-record")
    assert ET.tostring(source2) == ET.tostring(source3)
    for i in range(1, 6):
        node = element("03_cnn_and_tcn_models.svg", f"node-{i}")
        rect = next(x for x in node if x.tag.endswith("rect"))
        assert rect.get("fill") == "#F7FAFD"
        assert rect.get("stroke") == "#B8CFDE"
        assert len([x for x in node if x.get("class") == "nodedesc"]) == 3
    fig = ET.parse(folder / "03_cnn_and_tcn_models.svg")
    text = " ".join((x.text or "") for x in fig.iter())
    assert "Best checkpoint: epoch 8" in text
    assert "Best checkpoint: epoch 5" in text
    assert "implemented and unexecuted" not in text


def test_output_folders_are_no_longer_ignored_but_size_guard_remains():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/eye_movement_modelling/output/*" not in ignore
    assert "/eye_tracking_foundations/output/*" not in ignore
    assert "/data/raw/" in ignore and "/data/processed/" in ignore
    policy = (ROOT / "scripts/check_git_file_sizes.py").read_text(
        encoding="utf-8"
    )
    assert "MAX_BYTES = 50_000_000" in policy


def test_stage04_saved_metrics_match_record_predictions():
    """Recalculate saved scores, distinguishing pilot from main source IDs."""
    import pandas as pd
    import pytest
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    output = ROOT / "eye_movement_modelling" / "output"
    split = pd.read_csv(
        output / "01_split_assignments.csv",
        dtype={"participant_id": str, "record_id": str},
    )
    current = split.loc[split["split"].eq("validation")]
    current_ids = set(current["record_id"])
    current_people = current["participant_id"].nunique()
    purpose = pd.read_csv(output / "01_protocol.csv")["purpose"].iloc[0]
    summary = pd.read_csv(output / "04_neural_validation_metrics.csv")
    assert set(summary["model"]) == {"gru", "transformer"}
    common_record_ids = None

    for model in ("gru", "transformer"):
        scores = summary.loc[summary["model"].eq(model)].iloc[0]
        saved = pd.read_csv(
            output / f"04_{model}_validation_predictions.csv",
            dtype={"record_id": str},
        )
        assert saved["record_id"].is_unique
        assert saved["partition"].eq("validation").all()
        ids = set(saved["record_id"])
        if common_record_ids is None:
            common_record_ids = ids
        else:
            assert ids == common_record_ids
        original_people = saved["record_id"].str.extract(
            r"^S_(\d{4})_S1_"
        )[0]
        assert original_people.notna().all()
        assert int(scores["n_records"]) == len(saved)
        assert int(scores["n_participants"]) == original_people.nunique()

        # After Notebook 01's 100-person rerun, Stage 04 is still from
        # the ten-person pilot. Keep its scientific outputs and check them
        # internally until Stage 04 is trained on the current main cohort.
        if len(saved) == len(current):
            assert ids == current_ids
            assert int(scores["n_participants"]) == current_people
        else:
            assert purpose == "main"
            assert len(saved) < len(current)

        truth, predicted = saved["target_index"], saved["pred_index"]
        assert scores["accuracy"] == pytest.approx(
            accuracy_score(truth, predicted)
        )
        assert scores["macro_f1"] == pytest.approx(
            f1_score(truth, predicted, labels=[0, 1, 2, 3],
                     average="macro", zero_division=0)
        )
        assert scores["balanced_accuracy"] == pytest.approx(
            balanced_accuracy_score(truth, predicted)
        )


def test_stage04_figure_preserves_measured_source_and_card_geometry():
    """Compare with the executed stage-03 editable SVG convention."""
    import xml.etree.ElementTree as ET

    folder = ROOT / "eye_movement_modelling" / "images"
    def part(filename, name):
        root = ET.parse(folder / filename).getroot()
        return next(el for el in root.iter() if el.get("id") == name)

    source3 = part("03_cnn_and_tcn_models.svg", "source-record")
    source4 = part("04_gru_and_transformer_models.svg", "source-record")
    assert ET.tostring(source3) == ET.tostring(source4)
    for i in range(1, 6):
        node = part("04_gru_and_transformer_models.svg", f"node-{i}")
        rect = next(c for c in node if c.tag.endswith("rect"))
        assert rect.get("fill") == "#F7FAFD"
        assert rect.get("stroke") == "#B8CFDE"
        assert len([
            c for c in node if c.get("class") == "nodedesc"
        ]) == 3
    root = ET.parse(folder / "04_gru_and_transformer_models.svg").getroot()
    figure_words = " ".join((c.text or "") for c in root.iter())
    assert "Best checkpoint: epoch 1" in figure_words
    assert "Best checkpoint: epoch 2" in figure_words
    assert "Results unexecuted" not in figure_words


def test_stage04_main_draft_and_export_contract():
    notebook = ROOT / "eye_movement_modelling" / "code" / "notebooks" / (
        "04_gru_and_transformer_models.ipynb"
    )
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    by_id = {c.get("id"): c for c in cells}
    for code_id in ("s2c7", "s2c9", "s2c11", "s2c13", "s2c15"):
        c = by_id[code_id]
        assert c["execution_count"] is None and not c["outputs"]
    for code_id in ("s2c9", "s2c11"):
        source = "".join(by_id[code_id]["source"])
        assert 'result["parameters"] = int(info["parameters"])' in source
        assert 'result["best_epoch"] = int(info["best_epoch"])' in source
    text = "".join(by_id["s2c15"]["source"])
    assert 'download_outputs_from_colab("eye_movement_modelling/output", "04")' in text

