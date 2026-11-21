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
    assert "02. Acquire and Select Original GazeBase" in section
    acquisition_idx = ids.index("s2-c-source-acquisition")
    assert acquisition_idx == ids.index("s2-m-006") + 1
    assert ids[acquisition_idx + 1] == "s2-c-007"
    acquire = "".join(cells[acquisition_idx]["source"])
    selection = "".join(cells[ids.index("s2-c-007")]["source"])
    assert "SOURCE_AUDIT_IDS = tuple(range(1001, 1021))" in acquire
    assert "ensure_gazebase_preflight(" in acquire
    assert "MAX_PARTICIPANTS = 10" in selection
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
        assert f'ensure_stage_inputs(output_dir, {int(stage)}, (' in eval_code
        assert f'"{stage}_neural_validation_metrics.csv"' in eval_code
