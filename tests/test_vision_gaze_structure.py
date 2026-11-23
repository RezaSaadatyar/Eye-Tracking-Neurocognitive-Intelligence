"""Small geometry and structure contracts; synthetic vectors only."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "vision_based_gaze_estimation"
sys.path.insert(0, str(STUDY / "code" / "src"))
from gaze_geometry import angular_error_degrees


def test_gaze_angular_error_known_directions():
    reference = np.array([[0., 0., 1.], [1., 0., 0.], [1., 1., 0.]])
    prediction = np.array([[0., 0., 1.], [0., 1., 0.], [-1., -1., 0.]])
    assert np.allclose(angular_error_degrees(reference, prediction),
                       [0., 90., 180.])


def test_gaze_angular_error_rejects_invalid_vectors():
    for invalid in (np.zeros((1, 3)), np.array([[np.nan, 0., 1.]])):
        with pytest.raises(ValueError):
            angular_error_degrees(np.array([[0., 0., 1.]]), invalid)
    with pytest.raises(ValueError):
        angular_error_degrees([[0., 0., 1.]], [[0., 1.]])


def test_vision_study_notebook_scaffold_structure():
    names = [
        "01_mpiigaze_source_audit_and_protocol",
        "02_eye_images_geometry_and_quality",
        "03_classical_gaze_estimation_baselines",
        "04_cnn_and_transfer_learning",
        "05_cross_person_evaluation_and_robustness",
    ]
    for name in names:
        path = STUDY / "code" / "notebooks" / (name + ".ipynb")
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert nb["cells"][0]["cell_type"] == "markdown"
        assert all(cell["execution_count"] is None and not cell["outputs"]
                   for cell in nb["cells"] if cell["cell_type"] == "code")
    study_readme = (STUDY / "README.md").read_text(encoding="utf-8")
    methods = (STUDY / "RESEARCH_METHODS_AND_EVIDENCE.md").read_text(
        encoding="utf-8")
    assert sum(line.startswith("## ") for line in study_readme.splitlines()) == 6
    assert sum(line.startswith("## ") for line in methods.splitlines()) == 9
