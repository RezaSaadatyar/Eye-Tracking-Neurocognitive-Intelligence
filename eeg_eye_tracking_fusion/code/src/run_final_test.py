"""Execute Notebook 06's frozen final Test branch outside Colab.

The headless runner substitutes environment setup cells in memory, executes
the scientific cells and verifies the Test outputs. It preserves the original
notebook setup while saving the resulting scientific outputs.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "eeg_eye_tracking_fusion"
OUTPUT = STUDY / "output"
NOTEBOOK = STUDY / "code/notebooks/06_robustness_and_final_evaluation.ipynb"
TEST_FILES = (
    "06_test_metrics.csv", "06_test_predictions.csv",
    "06_person_metrics.csv", "06_cluster_bootstrap.csv",
    "06_primary_test_confusion.png",
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def frozen_decision_check():
    # Read the completed development decision before accessing Test labels.
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    decision = pd.read_csv(OUTPUT / "06_model_selection.csv")
    require(len(decision) == 1, "Expected exactly one frozen development decision")
    row = decision.iloc[0]
    require(row.selected_model == "gaze_forest"
            and row.run_mode in ("development", "final")
            and not bool(row.test_used_in_selection),
            "Development selection was changed or used Test")
    require(int(row.source_people) == 18
            and int(row.validation_n_people) == 3
            and np.isclose(row.validation_macro_f1, 0.624390243902439,
                           atol=1e-9),
            "Frozen 18-person Validation result does not match the recorded decision")
    fitted = joblib.load(OUTPUT / "04_gaze_forest.joblib")
    forest = fitted.named_steps.get("randomforestclassifier")
    require(isinstance(forest, RandomForestClassifier)
            and forest.n_estimators == 150
            and forest.min_samples_leaf == 3
            and forest.class_weight == "balanced"
            and forest.random_state == 17,
            "Frozen trained Gaze Random Forest configuration does not match")
    print("FROZEN DEVELOPMENT MODEL: gaze_forest | Validation Macro F1 0.6243902439")


def portable_setup():
    return """from pathlib import Path
import json
import random
import sys
root = Path.cwd()
sys.path.insert(0, str(root / 'python_modules'))
sys.path.insert(0, str(root / 'eeg_eye_tracking_fusion/code/src'))
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display
from core.plot_style import plot_style
from features import verify_temporal_feature_alignment
from models import (
    MODEL_ORDER, CLASSICAL_ORDER, NEURAL_ORDER, CLASS_TO_INDEX,
    metrics, matrix, probabilities, record_predictions,
    torch_network, predict_neural, feature_columns,
    participant_prediction_metrics,
)
SEED = 17
random.seed(SEED)
np.random.seed(SEED)
work = root / 'eeg_eye_tracking_fusion'
output_dir = work / 'output'
plot_style(font_family='DejaVu Sans', figure_dpi=100)
"""


def validate_results():
    import sys
    sys.path.insert(0, str(STUDY / "code/src"))
    from models import MODEL_ORDER, metrics

    for filename in TEST_FILES:
        file = OUTPUT / filename
        require(file.is_file() and file.stat().st_size > 0,
                f"Missing Test output: {filename}")
    decision = pd.read_csv(OUTPUT / "06_model_selection.csv").iloc[0]
    require(decision.selected_model == "gaze_forest"
            and decision.run_mode == "final"
            and not bool(decision.test_used_in_selection),
            "Final evaluation changed the frozen model decision")
    table = pd.read_csv(OUTPUT / "03_features.csv", dtype={"participant_id": str})
    test = table.loc[table.split.eq("test")]
    other = table.loc[table.split.ne("test")]
    require(test.participant_id.nunique() == 4
            and not (set(test.participant_id) & set(other.participant_id))
            and test.window_id.is_unique,
            "Test participants/windows overlap with development")
    scores = pd.read_csv(OUTPUT / "06_test_metrics.csv")
    predictions = pd.read_csv(OUTPUT / "06_test_predictions.csv",
                              dtype={"participant_id": str})
    person = pd.read_csv(OUTPUT / "06_person_metrics.csv")
    bootstrap = pd.read_csv(OUTPUT / "06_cluster_bootstrap.csv")
    require(len(scores) == len(MODEL_ORDER) and
            set(scores.model) == set(MODEL_ORDER),
            "Test model summary is incomplete")
    require(len(predictions) == len(test) * len(MODEL_ORDER)
            and not predictions.duplicated(["model", "window_id"]).any()
            and predictions.split.eq("test").all(),
            "Test predictions are missing, duplicated or contaminated")
    require(len(person) == len(MODEL_ORDER) * test.participant_id.nunique(),
            "Participant-level Test evidence is incomplete")
    require(len(bootstrap) == 500, "Selected-model bootstrap did not complete")
    labels = test.task.map({"nr": 0, "tsr": 1}).to_numpy(dtype=int)
    for name in MODEL_ORDER:
        subset = predictions.loc[predictions.model.eq(name)].set_index("window_id")
        require(set(subset.index) == set(test.window_id),
                f"Unmatched Test windows for {name}")
        aligned = subset.loc[test.window_id]
        probs = aligned[["p_nr", "p_tsr"]].to_numpy(dtype=float)
        guess = aligned.pred_index.to_numpy(dtype=int)
        require(np.array_equal(aligned.target_index.to_numpy(dtype=int), labels)
                and np.array_equal(aligned.participant_id.to_numpy(),
                                   test.participant_id.to_numpy())
                and np.isfinite(probs).all()
                and (probs >= 0).all() and (probs <= 1).all()
                and np.allclose(probs.sum(axis=1), 1, atol=1e-5)
                and np.array_equal(guess, probs.argmax(axis=1)),
                f"Test labels/probabilities are inconsistent for {name}")
        recorded = scores.loc[scores.model.eq(name)].iloc[0]
        require(int(recorded.n_windows) == len(test)
                and int(recorded.n_people) == test.participant_id.nunique()
                and bool(recorded.selected_primary) == (name == "gaze_forest"),
                f"Model/Test support mismatch for {name}")
        for metric, value in metrics(labels, guess).items():
            require(np.isclose(value, recorded[metric], atol=1e-10),
                    f"Test {metric} could not be reproduced for {name}")
    primary = scores.loc[scores.model.eq("gaze_forest")].iloc[0]
    subset = person.loc[person.model.eq("gaze_forest")]
    require(subset.participant_id.nunique() == 4,
            "Missing primary participant-level scores")
    lower, upper = np.quantile(bootstrap.macro_f1, [0.025, 0.975])
    report = {
        "selected_model": "gaze_forest", "n_windows": int(primary.n_windows),
        "n_people": int(primary.n_people), "accuracy": float(primary.accuracy),
        "macro_f1": float(primary.macro_f1),
        "balanced_accuracy": float(primary.balanced_accuracy),
        "person_macro_f1": {
            row.participant_id: float(row.macro_f1)
            for row in subset.itertuples(index=False)
        },
        "bootstrap_macro_f1_95_percentile": [float(lower), float(upper)],
    }
    print("VERIFIED_FINAL_TEST_PRIMARY " + json.dumps(report, sort_keys=True))
    print("SECONDARY_MODEL_RESULTS")
    print(scores.sort_values("macro_f1", ascending=False).to_string(index=False))


def main():
    frozen_decision_check()
    original = nbformat.read(NOTEBOOK, as_version=4)
    require(original.cells[8].source.count("RUN_FINAL_TEST = True") == 1
            and "selected_name = str(ranking.iloc[0].model)" in original.cells[8].source,
            "Notebook 06 has not explicitly enabled its frozen Test branch")
    run = deepcopy(original)
    run.cells[2].source = "from pathlib import Path\nroot = Path.cwd()\n"
    run.cells[4].source = portable_setup()
    run.cells[18].source = "print('Final Test outputs saved in eeg_eye_tracking_fusion/output')"
    client = NotebookClient(
        run, timeout=1200, kernel_name="python3", allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute()
    validate_results()
    # Record actual inference outputs, preserving the Colab-specific source
    # setup, download cell, notebook cell identities and scientific code.
    for index in (6, 8, 10, 12, 14, 16):
        original.cells[index].outputs = executed.cells[index].outputs
        original.cells[index].execution_count = executed.cells[index].execution_count
    nbformat.write(original, NOTEBOOK)
    print("Notebook 06 final Test code executed; scientific outputs verified.")


if __name__ == "__main__":
    main()
