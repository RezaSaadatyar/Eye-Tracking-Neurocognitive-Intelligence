# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Synthetic scientific contracts for independent GazeBase modelling.

These tests do not reproduce original-data model training or invent its scores.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eye_movement_modelling" / "code" / "src"))
sys.path.insert(0, str(ROOT / "eye_tracking_foundations" / "code" / "src"))

from gaze_source import discover_records
from s2_data import (
    TASKS, participant_split, build_source_tables, normalise_train_only,
    verify_temporal_source_alignment, save_protocol, load_protocol,
)
from s2_metrics import metric_row, record_probabilities, require_split_consistency


def _manifest(n_people: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"record_id": f"S_{1000+i}_S1_{task}",
         "participant_id": f"{i:03d}", "task": task}
        for i in range(1, n_people + 1) for task in TASKS
    ])


def test_s2_person_groups_and_pilot_partition():
    inv = _manifest(10)
    a = participant_split(inv, seed=17)
    b = participant_split(inv, seed=17)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 40
    assert a.study_purpose.unique().tolist() == ["pilot"]
    assert a.groupby("participant_id").split.nunique().eq(1).all()
    assert set(a.split) == {"train", "validation", "test"}
    assert a.loc[a.split.eq("test"), "participant_id"].nunique() == 2
    assert a.loc[a.split.eq("validation"), "participant_id"].nunique() == 2


def test_s2_main_test_uses_participants_unseen_in_first_ten():
    inv = _manifest(20)
    groups = participant_split(inv, seed=17)
    assert groups.study_purpose.unique().tolist() == ["main"]
    people = set(groups.loc[groups.split.eq("test"), "participant_id"])
    assert all(int(p) > 10 for p in people)
    assert groups.groupby("participant_id").split.nunique().max() == 1
    assert len(groups) == 80


def test_s2_source_feature_and_temporal_contract(tmp_path):
    directory = tmp_path / "GazeBase_v2_0" / "Round_1"
    directory.mkdir(parents=True)
    n = np.arange(320, dtype=int)
    for i in range(1, 11):
        for j, task in enumerate(TASKS):
            path = directory / f"S_{1000+i}_S1_{task}.csv"
            pd.DataFrame({
                "n": n, "x": np.sin(n / 30) + j,
                "y": np.cos(n / 27) + i / 10,
                "val": np.zeros(len(n), dtype=int),
                "dP": np.ones(len(n)) * 100,
                "lab": np.ones(len(n), dtype=int),
            }).to_csv(path, index=False)
    inv = discover_records(directory, max_participants=10)
    group = participant_split(inv)
    frame, windows, record_ids, labels = build_source_tables(
        inv, group, window_samples=256, max_windows_per_record=3,
    )
    assert len(frame) == 40
    assert windows.shape == (40, 256, 4)
    assert all(frame.n_windows == 1)
    assert np.isfinite(windows).all()
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert set(record_ids) == set(inv.record_id)
    require_split_consistency(frame, group)


def test_s2_train_only_normalisation_resists_test_outliers():
    inv = _manifest(10)
    group = participant_split(inv)
    train_ids = group.loc[group.split.eq("train"), "record_id"].values
    test_ids = group.loc[group.split.eq("test"), "record_id"].values
    array = np.zeros((4, 32, 4), dtype=np.float32)
    array[0] = 1
    array[1] = 3
    array[2] = 1e5
    array[3] = -1e5
    source = np.asarray([
        train_ids[0], train_ids[1], test_ids[0], test_ids[1]
    ])
    transformed, mean, scale = normalise_train_only(array, source, group)
    assert np.allclose(mean, 2)
    assert np.allclose(scale, 1)
    assert np.allclose(transformed[:2].mean(axis=(0, 1)), 0)


def test_s2_temporal_windows_aggregate_by_record_not_window():
    rec = np.array(["a", "a", "a", "b"], dtype=str)
    y = np.array([0, 0, 0, 1])
    probs = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.7, 0.2, 0.05, 0.05],
        [0.9, 0.05, 0.03, 0.02],
        [0.05, 0.9, 0.03, 0.02],
    ])
    result = record_probabilities(rec, y, probs)
    assert len(result) == 2
    assert result.set_index("record_id").loc["a", "n_windows"] == 3
    assert result.set_index("record_id").loc["b", "pred_index"] == 1
    with pytest.raises(ValueError, match="Conflicting task"):
        record_probabilities(rec, np.array([0, 1, 0, 1]), probs)


def test_s2_metric_fixed_four_label_macro_f1():
    row = metric_row(
        np.array([0, 1]), np.array([0, 1]),
        model="dummy", partition="validation", n_participants=2,
    )
    assert row["n_records"] == 2
    assert row["macro_f1"] == pytest.approx(0.5)
    assert row["accuracy"] == 1.0


def test_s2_all_authored_python_and_notebook_cells_parse():
    source = ROOT / "eye_movement_modelling" / "code"
    for py in (source / "src").glob("*.py"):
        ast.parse(py.read_text(encoding="utf-8"))
    expected = (
        "01_data_and_experimental_protocol.ipynb",
        "02_statistical_and_classical_ml.ipynb",
        "03_cnn_and_tcn_models.ipynb",
        "04_gru_and_transformer_models.ipynb",
        "05_robustness_and_final_evaluation.ipynb",
    )
    for name in expected:
        p = source / "notebooks" / name
        notebook = json.loads(p.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"][0]["cell_type"] == "markdown"
        # Executed notebooks legitimately retain saved scientific evidence.
        assert all(isinstance(c.get("outputs", []), list)
                   for c in notebook["cells"] if c["cell_type"] == "code")
        for c in notebook["cells"]:
            if c["cell_type"] != "code":
                continue
            source_text = "".join(c["source"])
            filtered = "\n".join(
                line for line in source_text.splitlines()
                if not line.strip().startswith("%")
            )
            ast.parse(filtered)


def test_s2_pipeline_versions_and_dependencies_declared():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "scikit-learn" in text and "torch" in text and "joblib" in text


def test_s2_temporal_source_alignment_detects_stale_windows():
    registry = participant_split(_manifest(10))
    arrays = {
        "x": np.zeros((len(registry), 32, 4), dtype=np.float32),
        "record_id": registry.record_id.to_numpy(dtype=str),
        "y": registry.task.map({
            label: i for i, label in enumerate(TASKS)
        }).to_numpy(dtype=np.int64),
    }
    counts = verify_temporal_source_alignment(arrays, registry)
    assert sum(counts.values()) == len(registry)
    bad = dict(arrays)
    bad["y"] = arrays["y"].copy()
    bad["y"][0] = (bad["y"][0] + 1) % 4
    with pytest.raises(ValueError, match="Task labels differ"):
        verify_temporal_source_alignment(bad, registry)
    missing = {key: value[:-1] for key, value in arrays.items()}
    with pytest.raises(ValueError, match="recording sets differ"):
        verify_temporal_source_alignment(missing, registry)


def test_s2_classical_models_fit_without_test_leakage():
    from s2_classical import classical_candidates
    from s2_metrics import (
        FEATURE_COLUMN_NAMES, classical_record_predictions,
    )

    registry = participant_split(_manifest(10))
    rows = []
    for idx, src in registry.iterrows():
        class_id = TASKS.index(src.task)
        row = {
            "record_id": src.record_id, "participant_id": src.participant_id,
            "task": src.task, "split": src.split,
        }
        for j, feature in enumerate(FEATURE_COLUMN_NAMES):
            row[feature] = float(
                class_id * 0.25 + (int(src.participant_id) % 3) * 0.02
                + 0.001 * j
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    train = frame.loc[frame.split.eq("train")]
    valid = frame.loc[frame.split.eq("validation")]
    for name, model in classical_candidates().items():
        model.fit(
            train[list(FEATURE_COLUMN_NAMES)],
            train.task.map({x: i for i, x in enumerate(TASKS)}),
        )
        pred = classical_record_predictions(
            model, valid, model_name=name, partition="validation",
        )
        assert len(pred) == len(valid)
        assert np.allclose(pred[[f"p_{x}" for x in TASKS]].sum(axis=1), 1)


def test_s2_temporal_architectures_forward_if_torch_available():
    torch = pytest.importorskip("torch")
    from s2_models import make_model, parameter_count

    test = torch.zeros(2, 256, 4)
    for name in ("cnn", "tcn", "gru", "transformer"):
        model = make_model(name)
        output = model(test)
        assert output.shape == (2, 4)
        assert torch.isfinite(output).all()
        assert parameter_count(model) > 0

def test_protocol_csv_round_trip(tmp_path):
    path = tmp_path / "01_protocol.csv"
    saved = save_protocol(
        path, participants=10, seed=17,
        window_samples=256, max_windows_per_record=24, purpose="pilot",
    )
    loaded = load_protocol(path)
    assert loaded == saved
    assert path.exists()
    assert not path.with_suffix(".json").exists()
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "temporal_channels" in header and "split_seed" in header
