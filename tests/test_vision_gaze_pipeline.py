"""Source-boundary and synthetic-only contracts for the MPIIGaze study.

No original participant files are used; this suite checks the code paths,
record grouping, vector geometry and metadata alignment before Colab runs.
"""

import ast
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "vision_based_gaze_estimation"
sys.path.insert(0, str(STUDY / "code" / "src"))

from mpiigaze_data import read_normalised_eye_mat, build_eye_sample_cache
from gaze_evaluation import (
    participant_fold_table, outer_fold_indices, load_image_cache,
    person_angular_metrics,
)
from gaze_models import normalise_directions, fit_ridge_and_predict


def _write_synthetic_day(path, *, people_seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.stack([
        np.full((36, 60), 30 + people_seed, dtype=np.uint8),
        np.full((36, 60), 150 + people_seed, dtype=np.uint8),
        np.full((36, 60), 210 + people_seed, dtype=np.uint8),
    ])
    gazes = np.array([[0., 0.], [0.1, 0.3], [-0.2, -0.3]])
    savemat(path, {"data": {"left": {"image": images, "gaze": gazes}}})


def test_source_mat_images_and_angles_have_matching_rows(tmp_path):
    path = tmp_path / "day01.mat"
    _write_synthetic_day(path)
    images, gaze = read_normalised_eye_mat(path)
    assert images.shape == (3, 36, 60)
    assert images.dtype == np.uint8
    assert gaze.shape == (3, 3)
    assert np.isfinite(gaze).all()
    assert np.allclose(np.linalg.norm(gaze, axis=1), 1., atol=1e-6)
    assert np.array_equal(images[:, 0, 0], [30, 150, 210])
    with pytest.raises(ValueError, match="eye"):
        read_normalised_eye_mat(path, eye="centre")


def test_source_sample_reservoir_is_repeatable_without_label_selection(tmp_path):
    root = tmp_path / "MPIIGaze" / "Data" / "Normalized"
    for i in range(5):
        _write_synthetic_day(root / f"p{i:02}" / "day01.mat",
                             people_seed=i)
    first = build_eye_sample_cache(tmp_path, eye="left", per_person=2, seed=17)
    second = build_eye_sample_cache(tmp_path, eye="left", per_person=2, seed=17)
    rows, images, gaze, counts = first
    assert len(rows) == 10
    assert images.shape == (10, 36, 60)
    assert gaze.shape == (10, 3)
    assert all(value == 3 for value in counts.values())
    assert [row["sample_id"] for row in rows] == [
        row["sample_id"] for row in second[0]
    ]
    assert np.array_equal(images, second[1])
    assert len({row["sample_id"] for row in rows}) == len(rows)


def test_nested_folds_disjoint_and_reproducible():
    participants = [f"p{i:02}" for i in range(5)]
    a = participant_fold_table(participants, seed=17, inner_people=2)
    b = participant_fold_table(participants[::-1], seed=17, inner_people=2)
    pd.testing.assert_frame_equal(a, b)
    assert a.fold_id.nunique() == len(participants)
    assert a.loc[a.role.eq("test"), "participant_id"].is_unique
    table = pd.DataFrame({
        "participant_id": np.repeat(participants, 3),
        "sample_id": [f"{x}:{k}" for x in participants for k in range(3)],
    })
    for fold in a.fold_id.unique():
        idx = outer_fold_indices(table, a, fold)
        assert set(idx) == {"train", "validation", "test"}
        assert sum(len(values) for values in idx.values()) == len(table)
        people = [set(table.iloc[values].participant_id)
                  for values in idx.values()]
        assert people[0].isdisjoint(people[1])
        assert people[0].isdisjoint(people[2])
        assert people[1].isdisjoint(people[2])
        assert len(people[2]) == 1


def test_image_cache_identity_and_prediction_aggregation(tmp_path):
    samples = pd.DataFrame({
        "sample_id": ["p00:x", "p00:y", "p01:z"],
        "participant_id": ["p00", "p00", "p01"],
        "source_mat": ["x.mat", "x.mat", "z.mat"],
        "gaze_x": [0., 0., 1.],
        "gaze_y": [0., 1., 0.],
        "gaze_z": [1., 0., 0.],
    })
    samples_path = tmp_path / "02_sample_index.csv"
    cache_path = tmp_path / "02_normalised_eye_cache.npz"
    samples.to_csv(samples_path, index=False)
    images = np.zeros((3, 36, 60), dtype=np.uint8)
    targets = samples[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(np.float32)
    np.savez_compressed(
        cache_path, sample_id=samples.sample_id.to_numpy(str),
        images=images, gaze=targets,
    )
    loaded, actual_images, actual_targets = load_image_cache(
        samples_path, cache_path)
    assert np.array_equal(loaded.sample_id, samples.sample_id)
    assert np.array_equal(actual_images, images)
    assert np.array_equal(actual_targets, targets)
    prediction = samples[["sample_id", "participant_id"]].copy()
    prediction["model"] = "mean_direction"
    prediction["fold_id"] = prediction.participant_id
    prediction[["target_x", "target_y", "target_z"]] = targets
    prediction[["pred_x", "pred_y", "pred_z"]] = targets
    scored, grouped = person_angular_metrics(prediction)
    assert len(grouped) == 2
    assert grouped.n_samples.sum() == 3
    assert np.allclose(scored.angular_error_deg, 0., atol=1e-6)
    samples.loc[0, "gaze_x"] = 1.
    samples.to_csv(samples_path, index=False)
    with pytest.raises(ValueError, match="targets"):
        load_image_cache(samples_path, cache_path)


def test_ridge_baseline_trains_only_on_supplied_rows():
    train_x = np.arange(48, dtype=float).reshape(12, 4)
    train_y = normalise_directions(
        np.column_stack((np.arange(12) + 1., np.ones(12),
                         np.full(12, 2.))))
    test_x = np.array([[2., 3., 4., 5.], [6., 7., 8., 9.]])
    model, prediction = fit_ridge_and_predict(
        train_x, train_y, test_x, 10.)
    assert prediction.shape == (2, 3)
    assert np.allclose(np.linalg.norm(prediction, axis=1), 1., atol=1e-6)
    assert hasattr(model, "predict")


def test_notebook_01_to_04_code_is_parseable_and_uses_shared_io():
    import json

    stems = [
        "01_mpiigaze_source_audit_and_protocol",
        "02_eye_images_geometry_and_quality",
        "03_classical_gaze_estimation_baselines",
        "04_cnn_and_transfer_learning",
    ]
    for stage, stem in enumerate(stems, 1):
        file = STUDY / "code" / "notebooks" / (stem + ".ipynb")
        nb = json.loads(file.read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 7
        assert all(c["execution_count"] is None and not c["outputs"]
                   for c in code_cells)
        for cell in code_cells:
            source = "".join(cell["source"])
            # Colab-only %cd is a valid IPython line magic.
            python_source = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith("%cd"))
            ast.parse(python_source)
        all_code = "\n".join("".join(c["source"]) for c in code_cells)
        assert 'setup_colab("vision_based_gaze_estimation")' in all_code
        assert "plot_style(" in all_code
        assert "save_figure(" in all_code
        assert "download_outputs_from_colab(" in all_code
        assert (f'"{stage:02}"' in all_code
                or f'download_outputs_from_colab(output_dir, {stage})' in all_code)
        if stage in (3, 4):
            assert "RUN_OUTER_TEST=False" in all_code or (
                "RUN_OUTER_TEST = False" in all_code)


def test_study_svg_figures_preserve_scientific_bands():
    """Validate editable five-band figures, node structure and XML identity."""
    from xml.etree import ElementTree

    figures = [
        "01_mpiigaze_source_audit_and_protocol.svg",
        "02_eye_images_geometry_and_quality.svg",
        "03_classical_gaze_estimation_baselines.svg",
        "04_cnn_and_transfer_learning.svg",
    ]
    for filename in figures:
        root = ElementTree.parse(STUDY / "images" / filename).getroot()
        assert root.attrib["viewBox"] == "0 0 1700 816"
        ids = [el.attrib["id"] for el in root.iter()
               if "id" in el.attrib]
        assert len(ids) == len(set(ids))
        assert all(f"band-{i}-" in " ".join(ids) for i in range(5))
        assert sum(x.startswith("evidence-card-") for x in ids) == 4
        assert sum(x.startswith("workflow-node-") for x in ids) == 7
        assert sum(x.startswith("workflow-connector-") for x in ids) == 6
        title = next(el.text for el in root.iter() if el.attrib.get("id")=="title")
        assert "MPIIGaze" in title or "Gaze" in title
