"""Synthetic and static contracts; no original ZuCo data are reproduced."""
from __future__ import annotations

import ast
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "eeg_eye_tracking_fusion"
sys.path.insert(0, str(PROJECT / "code" / "src"))

from alignment import (eye_sample_times, ensure_disjoint_registry,
                       window_registry)
from features import summarise_eeg, summarise_eye
from models import CLASSICAL_ORDER, MODEL_ORDER, metrics
from zuco_io import select_run_paths, split_people


def _fake_paths(subject: str, task: str, run: int):
    stem = f"{subject}/eeg/{subject}_task-{task}_run-{run}"
    return [f"{stem}_{suffix}" for suffix in (
        "eeg.eeg", "eeg.vhdr", "eeg.vmrk", "eeg.json",
        "events.tsv", "channels.tsv",
        "recording-eyetrack_physio.tsv.gz",
        "recording-eyetrack_physio.json")]


def test_discovery_requires_complete_paired_source_files():
    subject = "sub-YAC"
    paths = _fake_paths(subject, "nr", 1) + _fake_paths(subject, "tsr", 1)
    selected = select_run_paths(paths, subject, runs_per_task=1)
    assert len(selected) == 2
    assert {run["task"] for run in selected} == {"nr", "tsr"}
    assert all(set(("eeg", "eye", "header", "events")) <= set(x)
               for x in selected)
    assert select_run_paths(paths[:-1], subject) == []


def test_split_is_deterministic_and_person_disjoint():
    people = [f"sub-Q{i:02d}" for i in range(10)]
    a = split_people(people, seed=17)
    b = split_people(list(reversed(people)), seed=17)
    pd.testing.assert_frame_equal(a, b)
    assert a.participant_id.is_unique
    assert set(a.split) == {"train", "validation", "test"}
    assert len(a) == 10
    with pytest.raises(ValueError):
        split_people(people[:4])


def test_bids_timebase_and_eye_eligibility():
    metadata = {"StartTime": -0.02, "SamplingFrequency": 500.}
    t = eye_sample_times(metadata, 10000)
    assert np.isclose(t[0], -0.02)
    assert np.isclose(t[1] - t[0], .002)
    eye = pd.DataFrame({"eye1_x_coordinate": np.ones(10000),
                        "eye1_y_coordinate": np.ones(10000),
                        "eye1_pupil_size": np.ones(10000)})
    events = pd.DataFrame({"onset": [1.0, 4.0],
                           "duration": [.2, .4],
                           "trial_type": ["fixation", "saccade"]})

    class Raw:
        info = {"sfreq": 500.}
        n_times = 10000
        times = np.arange(10000)/500.

    task = {"participant_id": "sub-Q00", "task": "nr",
            "run": 1, "split": "train"}
    table, quality = window_registry(task, Raw(), eye, metadata, events,
                                     window_seconds=4., max_windows=3)
    assert len(table) == 3
    assert table.eligible.all()
    assert table.window_id.is_unique
    assert quality["drift_verified"] is False
    assert "BIDS" in quality["alignment_basis"]


def test_paired_split_registry_rejects_overlap():
    table = pd.DataFrame([
        {"window_id": "w1", "participant_id": "p1", "task": "nr",
         "split": "train", "eligible": True},
        {"window_id": "w2", "participant_id": "p1", "task": "tsr",
         "split": "train", "eligible": True},
    ])
    ensure_disjoint_registry(table)
    table.loc[1, "split"] = "test"
    with pytest.raises(ValueError, match="multiple split"):
        ensure_disjoint_registry(table)


def test_modality_features_and_fixed_label_metrics():
    sfreq = 500
    t = np.arange(sfreq*2) / sfreq
    eeg = np.stack([np.sin(2*np.pi*10*t),
                    np.sin(2*np.pi*6*t)]) * 1e-6
    summary = summarise_eeg(eeg, sfreq)
    assert summary.shape == (4,)
    assert np.isfinite(summary).all()
    eye_time = np.arange(1000) / 500
    eye = pd.DataFrame({"eye1_x_coordinate": np.linspace(0, 100, 1000),
                        "eye1_y_coordinate": np.linspace(0, 100, 1000),
                        "eye1_pupil_size": np.ones(1000)})
    ev = pd.DataFrame({"onset": [.5], "duration": [.7],
                       "trial_type": ["fixation"]})
    v = summarise_eye(eye, eye_time, ev, 0, 2)
    assert np.isclose(v[0], 1)
    assert np.isfinite(v).all()
    result = metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert np.isclose(result["accuracy"], .75)
    assert np.isclose(result["balanced_accuracy"], .75)
    assert set(CLASSICAL_ORDER).issubset(MODEL_ORDER)


def test_six_complete_colab_notebooks_and_stage_contracts():
    notebook_dir = PROJECT / "code" / "notebooks"
    paths = sorted(notebook_dir.glob("*.ipynb"))
    assert len(paths) == 6
    for index, path in enumerate(paths, 1):
        assert path.name.startswith(f"{index:02d}_")
        obj = json.loads(path.read_text(encoding="utf-8"))
        cells = obj["cells"]
        assert cells[0]["cell_type"] == "markdown"
        assert "EEG–Eye-Tracking Multimodal Research" in "".join(cells[0]["source"])
        assert "Colab Repository Setup" in "".join(cells[1]["source"])
        assert "Imports and Research Environment" in "".join(cells[3]["source"])
        assert len(cells) >= 17
        assert len([c for c in cells if c["cell_type"] == "code"]) >= 8
        checkout = "".join(cells[2]["source"])
        imports = "".join(cells[4]["source"])
        assert "%cd {repo_path}" in checkout
        assert "git", "clone" in checkout
        assert 'root = setup_colab("eeg_eye_tracking_fusion")' in imports
        assert 'from core.output_download import download_outputs_from_colab' in imports
        assert 'from core.plot_style import plot_style' in imports
        for cell_index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compilable = (source.replace('%cd {repo_path}', 'pass')
                          if cell_index == 2 else source)
            compile(compilable, str(path), "exec")
            assert not cell.get("outputs")
            assert "drive.mount" not in source
            if cell_index >= 5:
                parsed = ast.parse(source)
                assert not any(isinstance(node, (ast.Import, ast.ImportFrom))
                               for node in ast.walk(parsed))
        ending = "".join(cells[-1]["source"])
        assert 'download_outputs_from_colab(' in ending
        assert f'"{index:02d}"' in ending
    last = json.loads(paths[-1].read_text())
    body = "\n".join("".join(c["source"]) for c in last["cells"])
    assert "RUN_FINAL_TEST = False" in body
    assert "06_model_selection.csv" in body


def test_seven_v1_five_band_schematics_without_claimed_results():
    figures = sorted((PROJECT / "images").glob("*.svg"))
    assert len(figures) == 7
    for path in figures:
        root = ET.parse(path).getroot()
        assert root.attrib["viewBox"] == "0 0 1700 816"
        assert root.attrib["width"] == "1700"
        assert root.attrib["height"] == "816"
        ids = {n.attrib.get("id") for n in root.iter()}
        assert {f"band-{i}-{label}" for i, label in enumerate((
            "header", "context", "analysis", "workflow", "summary"))} <= ids
        svg = path.read_text()
        assert "DejaVu Sans" in svg
        assert "[SCIENTIFIC FIGURE" not in svg
        assert "[Supported detail]" not in svg
        assert "[No inherited results]" not in svg


def test_two_scientific_documents_retained_in_named_structure():
    readme = (PROJECT / "README.md").read_text()
    methods = (PROJECT / "RESEARCH_METHODS_AND_EVIDENCE.md").read_text()
    assert len([l for l in readme.splitlines() if l.startswith("## ")]) == 6
    assert len([l for l in methods.splitlines() if l.startswith("## ")]) == 9
    assert "not yet" in readme.lower()
    assert "not yet" in methods.lower()
    assert "100% accuracy" not in methods
