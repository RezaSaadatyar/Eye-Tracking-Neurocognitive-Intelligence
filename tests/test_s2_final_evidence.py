# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Reconcile committed S2 pilot results without repeating Test inference."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eye_movement_modelling" / "output"
NOTEBOOK = (
    ROOT / "eye_movement_modelling" / "code" / "notebooks"
    / "05_robustness_and_final_evaluation.ipynb"
)
FIGURES = ROOT / "eye_movement_modelling" / "images"
TASKS = {"FXS": 0, "HSS": 1, "RAN": 2, "TEX": 3}
PRIORITY = (
    "class_prior", "logistic_regression", "random_forest",
    "gradient_boosting", "cnn", "tcn", "gru", "transformer",
)


def _row_metrics(truth: pd.Series, predicted: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(
            truth, predicted, labels=list(TASKS.values()),
            average="macro", zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
    }


def test_pilot_validation_selection_uses_only_saved_validation_scores():
    """Rebuild the fixed ranking without consulting stage-05 Test scores."""
    metrics = pd.concat(
        [
            pd.read_csv(OUTPUT / "02_validation_metrics.csv"),
            pd.read_csv(OUTPUT / "03_neural_validation_metrics.csv"),
            pd.read_csv(OUTPUT / "04_neural_validation_metrics.csv"),
        ],
        ignore_index=True,
    )
    assert metrics["model"].is_unique
    assert set(metrics["model"]) == set(PRIORITY)
    assert metrics["partition"].eq("validation").all()
    priority = {name: i for i, name in enumerate(PRIORITY)}
    metrics["priority"] = metrics["model"].map(priority)
    winner = metrics.sort_values(
        ["macro_f1", "balanced_accuracy", "priority"],
        ascending=[False, False, True],
    ).iloc[0]["model"]
    decision = pd.read_csv(OUTPUT / "05_model_selection.csv").iloc[0]
    assert winner == decision["selected_model"]
    assert decision["selection_rule"] == (
        "validation_macro_f1_then_balanced_accuracy_then_order"
    )
    assert decision["test_used_in_selection"] == False
    assert decision["purpose"] == "pilot"


def test_saved_pilot_test_metrics_and_groupwise_support_reconcile():
    """Recalculate observed Test scores using record-level saved predictions."""
    split = pd.read_csv(
        OUTPUT / "01_split_assignments.csv",
        dtype={"record_id": str, "participant_id": str},
    )
    expected = split.loc[split["split"].eq("test")]
    expected_ids = set(expected["record_id"])
    expected_people = set(expected["participant_id"])
    assert set(split.loc[split["split"].eq("train"), "participant_id"]).isdisjoint(
        expected_people
    )
    assert set(split.loc[split["split"].eq("validation"), "participant_id"]).isdisjoint(
        expected_people
    )

    predictions = pd.read_csv(
        OUTPUT / "05_test_predictions.csv",
        dtype={"record_id": str, "participant_id": str},
    )
    report = pd.read_csv(OUTPUT / "05_test_metrics.csv")
    groupwise = pd.read_csv(
        OUTPUT / "05_groupwise_test_metrics.csv",
        dtype={"participant_id": str},
    )
    decision = pd.read_csv(OUTPUT / "05_model_selection.csv").iloc[0]
    assert set(predictions["model"]) == set(PRIORITY)
    assert set(report["model"]) == set(PRIORITY)
    assert report["model"].is_unique
    assert not predictions.duplicated(["model", "record_id"]).any()

    expected_identity = expected.set_index("record_id")
    for name, frame in predictions.groupby("model"):
        assert set(frame["record_id"]) == expected_ids
        assert frame["partition"].eq("test").all()
        assert set(frame["participant_id"]) == expected_people
        for item in frame.itertuples(index=False):
            source = expected_identity.loc[item.record_id]
            assert item.participant_id == source.participant_id
            assert item.task == source.task
            assert item.target_index == TASKS[item.task]
        probabilities = frame[[f"p_{label}" for label in TASKS]].to_numpy()
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)
        assert np.array_equal(
            probabilities.argmax(axis=1), frame["pred_index"].to_numpy()
        )

        scores = report.loc[report["model"].eq(name)].iloc[0]
        assert scores["partition"] == "test"
        assert scores["study_purpose"] == "pilot"
        assert scores["n_records"] == len(frame)
        assert scores["n_participants"] == len(expected_people)
        assert bool(scores["selected_primary"]) == (
            name == decision["selected_model"]
        )
        for metric, value in _row_metrics(
            frame["target_index"], frame["pred_index"],
        ).items():
            assert scores[metric] == pytest.approx(value)

        for person, subframe in frame.groupby("participant_id"):
            person_result = groupwise.loc[
                groupwise["model"].eq(name)
                & groupwise["participant_id"].eq(person)
            ]
            assert len(person_result) == 1
            row = person_result.iloc[0]
            assert row["n_records"] == len(subframe)
            assert row["n_participants"] == 1
            for metric, value in _row_metrics(
                subframe["target_index"], subframe["pred_index"],
            ).items():
                assert row[metric] == pytest.approx(value)


def test_recorded_pupil_masking_is_validation_only_and_descriptive():
    robustness = pd.read_csv(OUTPUT / "05_robustness_validation.csv")
    decision = pd.read_csv(OUTPUT / "05_model_selection.csv").iloc[0]
    assert set(robustness["model"]) == {decision["selected_model"]}
    assert set(robustness["condition"]) == {
        "original_validation", "pupil_features_masked",
    }
    expected_macro_f1 = pd.read_csv(
        OUTPUT / "02_validation_metrics.csv"
    ).set_index("model").loc[decision["selected_model"], "macro_f1"]
    original = robustness.loc[
        robustness["condition"].eq("original_validation")
    ].iloc[0]
    assert original["macro_f1"] == pytest.approx(expected_macro_f1)
    assert robustness["n_records"].nunique() == 1
    # The altered-data predictions were not saved; this test checks the
    # reported table, not a fresh perturbation or a causal attribution.


def test_final_notebook_preserves_test_outputs_and_single_zip_export():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    by_id = {c.get("id"): c for c in cells}
    assert [c.get("id") for c in cells].index("s2c9") < [
        c.get("id") for c in cells
    ].index("s2c11")
    for ident in ("s2c7", "s2c9", "s2c11", "s2c13", "s2c15", "s2c17"):
        assert by_id[ident]["outputs"]
    final = by_id["s2c19"]
    code = "".join(final["source"])
    assert 'download_outputs_from_colab("eye_movement_modelling/output", "05")' in code
    assert "print(" not in code
    assert final["execution_count"] is None
    assert not final["outputs"]


def test_final_figure_matches_source_and_workflow_reference():
    def svg_component(filename: str, identifier: str):
        root = ET.parse(FIGURES / filename).getroot()
        return next(node for node in root.iter()
                    if node.attrib.get("id") == identifier)

    source4 = svg_component("04_gru_and_transformer_models.svg", "source-record")
    source5 = svg_component("05_robustness_and_final_evaluation.svg", "source-record")
    assert ET.tostring(source4) == ET.tostring(source5)
    for index in range(1, 6):
        node = svg_component(
            "05_robustness_and_final_evaluation.svg", f"node-{index}",
        )
        rect = next(child for child in node if child.tag.endswith("rect"))
        assert rect.attrib["fill"] == "#F7FAFD"
        assert rect.attrib["stroke"] == "#B8CFDE"
        assert len([
            child for child in node
            if child.attrib.get("class") == "nodedesc"
        ]) == 3
    root = ET.parse(FIGURES / "05_robustness_and_final_evaluation.svg")
    words = " ".join(element.text or "" for element in root.iter())
    assert "Primary: 7/8 correct" in words
    assert "Test Macro F1: 0.867" in words
    assert "remain unobserved" not in words
