"""Synthetic and static contracts; no original ZuCo data are reproduced."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "eeg_eye_tracking_fusion"
sys.path.insert(0, str(PROJECT / "code" / "src"))

from alignment import (eye_sample_times, ensure_disjoint_registry,
                       valid_eye_rows, window_registry)
from features import (
    summarise_eeg, summarise_eye, verify_temporal_feature_alignment,
)
from models import CLASSICAL_ORDER, MODEL_ORDER, metrics
from zuco_io import (
    API, DATA, S3, download_file, read_eye_physio, select_run_paths,
    session, split_people,
)


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


def test_headerless_eye_physio_uses_bids_sidecar_columns(tmp_path):
    """Read every original sample and preserve sidecar column order."""
    path = tmp_path / "sub-YAC_task-nr_recording-eyetrack_physio.tsv.gz"
    original = pd.DataFrame([
        [37.0, 15.0, 80.0],
        [38.0, 16.0, 81.0],
        ["n/a", 17.0, 82.0],
        [40.0, 18.0, 83.0],
    ])
    original.to_csv(
        path, sep="\t", header=False, index=False, compression="gzip"
    )
    metadata = {
        "SamplingFrequency": 500.0,
        "StartTime": -0.035,
        "Columns": [
            "eye1_x_coordinate", "eye1_y_coordinate", "eye1_pupil_size"
        ],
    }
    eye = read_eye_physio(path, metadata)
    assert eye.columns.tolist() == metadata["Columns"]
    assert len(eye) == 4  # First actual gaze sample must not be dropped.
    assert eye.iloc[0]["eye1_x_coordinate"] == 37.0
    assert eye.iloc[0]["eye1_y_coordinate"] == 15.0
    assert pd.isna(eye.iloc[2]["eye1_x_coordinate"])
    t = eye_sample_times(metadata, len(eye))
    assert np.isclose(t[0], -0.035)
    assert np.isclose(t[-1], -0.029)
    eye_valid = valid_eye_rows(eye)
    assert eye_valid.tolist() == [True, True, False, True]


def test_eye_physio_requires_valid_named_bids_columns(tmp_path):
    path = tmp_path / "sub-YAC_task-nr_recording-eyetrack_physio.tsv.gz"
    with gzip.open(path, "wt") as file:
        file.write("1\t2\t3\n")
    for metadata in (
        {},
        {"Columns": []},
        {"Columns": ["eye1_x_coordinate", "eye1_x_coordinate"]},
        {"Columns": ["eye1_x_coordinate", "eye1_pupil_size"]},
    ):
        with pytest.raises(ValueError, match="BIDS physio|gaze columns"):
            read_eye_physio(path, metadata)


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



def test_nemar_session_retries_temporary_service_errors():
    retry = session().get_adapter("https://").max_retries
    assert 503 in retry.status_forcelist
    assert retry.total >= 2
    assert retry.raise_on_status is False


def test_nemar_503_falls_back_to_verified_public_s3(tmp_path):
    """Resolve a real-shaped NEMAR Git-annex key without any network access."""
    relative = (
        "sub-YAC/eeg/"
        "sub-YAC_task-nr_run-1_recording-eyetrack_physio.tsv.gz"
    )
    content = gzip.compress(bytes(range(256)) * 300)
    digest = hashlib.sha256(content).hexdigest()
    key = f"SHA256E-s{len(content)}--{digest}.tsv.gz"
    visits = []

    class Response:
        def __init__(self, status=200, data=b"", pointer=None):
            self.status_code = status
            self.data = data
            self.pointer = pointer

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(
                    f"{self.status_code} upstream unavailable",
                    response=self,
                )

        def iter_content(self, chunk_size):
            yield self.data

        def json(self):
            return {"type": "symlink", "target": self.pointer}

    class FakeSession:
        def get(self, url, **_kwargs):
            visits.append(url)
            if url == f"{DATA}/{relative}":
                return Response(status=503)
            if url == f"{API}/{relative}":
                return Response(pointer=f"/annex/objects/{key}")
            if url == f"{S3}/{key}":
                return Response(data=content)
            raise AssertionError(f"Unexpected external URL: {url}")

    fake = FakeSession()
    path = download_file(relative, tmp_path, fake)
    assert path.read_bytes() == content
    assert len(visits) == 3
    assert visits == [
        f"{DATA}/{relative}",
        f"{API}/{relative}",
        f"{S3}/{key}",
    ]
    assert download_file(relative, tmp_path, fake) == path
    assert len(visits) == 3  # Already cached source is reused.
    assert not list(tmp_path.rglob("*.part"))


def test_nemar_fallback_rejects_mismatched_source_hash(tmp_path):
    relative = "sub-YAC/eeg/sub-YAC_task-nr_run-1_recording-eyetrack_physio.tsv.gz"
    content = gzip.compress(bytes(range(256)) * 300)
    digest = "a" * 64  # Intentionally different from actual content.
    key = f"SHA256E-s{len(content)}--{digest}.tsv.gz"

    class Response:
        def __init__(self, status, data=b"", target=None):
            self.status_code, self.data, self.target = status, data, target

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

        def json(self):
            return {"target": self.target}

        def iter_content(self, chunk_size):
            yield self.data

    class FakeSession:
        def get(self, url, **_kwargs):
            if url.startswith(DATA):
                return Response(503)
            if url.startswith(API):
                return Response(200, target=f"/annex/objects/{key}")
            if url == f"{S3}/{key}":
                return Response(200, data=content)
            raise AssertionError(url)

    with pytest.raises(RuntimeError, match="Neither NEMAR"):
        download_file(relative, tmp_path, FakeSession())
    assert not (tmp_path / relative).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_nemar_invalid_source_path_does_not_trigger_s3(tmp_path):
    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, _url, **_kwargs):
            self.calls += 1
            response = requests.Response()
            response.status_code = 404
            response.url = _url
            response.raise_for_status()
            return response

    fake = FakeSession()
    with pytest.raises(requests.HTTPError):
        download_file("sub-YAC/eeg/missing.tsv", tmp_path, fake)
    assert fake.calls == 1
    assert not list(tmp_path.rglob("*.part"))



def test_saved_zuco_source_evidence_reconciles():
    """Check recorded source IDs, participant roles and acquisition evidence."""
    output = PROJECT / "output"
    required = (
        "01_source_manifest.csv", "01_split_assignments.csv",
        "01_source_eligibility.csv", "01_download_inventory.csv",
        "01_acquisition_metadata.csv", "01_protocol.json",
        "01_source_overview.png",
    )
    assert all((output / file).is_file() for file in required)
    assert not (output / "README.md").exists()

    source = pd.read_csv(
        output / "01_source_manifest.csv", dtype={"participant_id": str}
    )
    roles = pd.read_csv(
        output / "01_split_assignments.csv", dtype={"participant_id": str}
    )
    eligible = pd.read_csv(
        output / "01_source_eligibility.csv", dtype={"participant_id": str}
    )
    inventory = pd.read_csv(output / "01_download_inventory.csv")
    metadata = pd.read_csv(output / "01_acquisition_metadata.csv")
    protocol = json.loads((output / "01_protocol.json").read_text())

    participants = source.participant_id.nunique()
    assert participants == roles.participant_id.nunique()
    assert participants == protocol["max_participants"]
    assert len(source) == 2 * participants * protocol["runs_per_task"]
    assert set(source.task) == {"nr", "tsr"}
    assert source.groupby("participant_id").task.nunique().eq(2).all()
    assert source.groupby("participant_id").split.nunique().eq(1).all()

    same_roles = source[["participant_id", "split"]].merge(
        roles, on="participant_id", validate="many_to_one",
        suffixes=("_manifest", "_registry"),
    )
    assert same_roles.split_manifest.eq(same_roles.split_registry).all()
    assert int(eligible.eligible.sum()) >= participants

    key = ["participant_id", "task", "run"]
    for evidence in (inventory, metadata):
        assert not evidence.duplicated(key).any()
        assert source[key].merge(
            evidence[key], on=key, how="left", indicator=True,
            validate="one_to_one",
        )["_merge"].eq("both").all()

    assert inventory.all_files_present.all()
    assert (inventory.source_megabytes > 0).all()
    assert metadata.eeg_sfreq.gt(0).all()
    assert metadata.eye_sfreq.gt(0).all()
    assert metadata.eeg_channels.gt(0).all()
    assert (output / "01_source_overview.png").stat().st_size > 1000


def test_saved_zuco_paired_window_evidence_is_source_consistent():
    """Reconcile real Stage-02 CSVs without assuming fixed window counts."""
    folder = PROJECT / "output"
    file_names = (
        "02_window_registry.csv",
        "02_alignment_quality.csv",
        "02_window_summary.csv",
        "02_alignment_quality.png",
    )
    assert all((folder / name).is_file() for name in file_names)
    source = pd.read_csv(
        folder / "01_source_manifest.csv", dtype={"participant_id": str}
    )
    windows = pd.read_csv(
        folder / "02_window_registry.csv", dtype={"participant_id": str}
    )
    quality = pd.read_csv(
        folder / "02_alignment_quality.csv", dtype={"participant_id": str}
    )
    summary = pd.read_csv(folder / "02_window_summary.csv")
    protocol = json.loads((folder / "01_protocol.json").read_text())

    assert not windows.empty
    assert windows.window_id.is_unique
    assert set(windows.participant_id) == set(source.participant_id)
    assert windows.eligible.dtype == bool
    assert quality.drift_verified.eq(False).all()
    assert windows.eye_valid_fraction.between(0, 1).all()
    np.testing.assert_allclose(
        windows.end_seconds - windows.start_seconds,
        protocol["window_seconds"], atol=1e-7,
    )
    assert (folder / "02_alignment_quality.png").stat().st_size > 1000

    keys = ["participant_id", "task", "run"]
    role_check = windows.merge(
        source[keys + ["split"]], on=keys, how="left",
        validate="many_to_one", suffixes=("_window", "_source"),
    )
    assert role_check.split_source.notna().all()
    assert role_check.split_window.eq(role_check.split_source).all()
    observed_run = (
        windows.groupby(keys).agg(
            total_windows=("window_id", "size"),
            eligible_windows=("eligible", "sum"),
        ).reset_index()
    )
    expected_run = quality[keys + ["total_windows", "eligible_windows"]]
    pd.testing.assert_frame_equal(
        observed_run.sort_values(keys).reset_index(drop=True),
        expected_run.sort_values(keys).reset_index(drop=True),
        check_dtype=False,
    )
    for win in windows.itertuples(index=False):
        per_run = quality.loc[
            quality.participant_id.eq(win.participant_id)
            & quality.task.eq(win.task)
            & quality.run.eq(win.run)
        ].iloc[0]
        assert win.start_seconds >= per_run.common_start_seconds - 1e-6
        assert win.end_seconds <= per_run.common_end_seconds + 1e-6

    observed_summary = (
        windows.groupby(["split", "task"]).agg(
            total=("window_id", "size"),
            eligible=("eligible", "sum"),
            people=("participant_id", "nunique"),
        ).reset_index()
    )
    pd.testing.assert_frame_equal(
        observed_summary.sort_values(["split", "task"])
        .reset_index(drop=True),
        summary.sort_values(["split", "task"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_feature_notebook_preserves_frozen_source_and_reports_exclusions():
    """Stage 03 should not hide cohort changes or failed feature windows."""
    nb = json.loads(
        (PROJECT / "code" / "notebooks"
         / "03_preprocessing_features_and_neurophysiology.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code"
    )
    assert "02_window_registry.csv" in source
    assert "02_alignment_quality.csv" in source
    assert "split_window" in source and "split_source" in source
    assert "ensure_disjoint_registry(registry)" in source
    assert "features.window_id.duplicated().any()" in source
    assert "preceding_issues" in source
    assert "group_support" in source
    assert "03_temporal_sequences.npz" in source
    assert not any(
        line.strip().startswith("assert ") for line in source.splitlines()
    )


def test_saved_zuco_matched_features_and_temporal_array_alignment():
    """Inspect actual Stage-03 outputs against Stage-02 provenance."""
    folder = PROJECT / "output"
    required = [
        "03_features.csv", "03_temporal_sequences.npz",
        "03_exclusion_reasons.csv", "03_modality_quality.csv",
        "03_temporal_example.png", "03_fixation_locked_diagnostic.png",
    ]
    assert all((folder / name).is_file() for name in required)
    features = pd.read_csv(
        folder / "03_features.csv", dtype={"participant_id": str}
    )
    windows = pd.read_csv(
        folder / "02_window_registry.csv", dtype={"participant_id": str}
    )
    quality = pd.read_csv(
        folder / "03_modality_quality.csv", dtype={"participant_id": str}
    )
    excluded = pd.read_csv(folder / "03_exclusion_reasons.csv")
    selected = windows.loc[windows.eligible]
    assert features.window_id.is_unique
    assert set(features.window_id) == set(selected.window_id)
    assert len(features) == len(selected)
    assert set(excluded.columns) == {"window_id", "reason"}
    assert excluded.empty

    original = features[
        ["window_id", "participant_id", "task", "run", "split"]
    ].merge(
        selected[["window_id", "participant_id", "task", "run", "split"]],
        on="window_id", how="left", validate="one_to_one",
        suffixes=("_feature", "_registry"),
    )
    for name in ("participant_id", "task", "run", "split"):
        assert original[f"{name}_feature"].eq(
            original[f"{name}_registry"]
        ).all()

    observed = features.groupby(
        ["participant_id", "split", "task"]
    ).size().rename("n_windows").reset_index()
    columns = ["participant_id", "split", "task"]
    pd.testing.assert_frame_equal(
        observed.sort_values(columns).reset_index(drop=True),
        quality.sort_values(columns).reset_index(drop=True),
        check_dtype=False,
    )
    eeg_names = [c for c in features.columns if c.startswith("eeg_")]
    gaze_names = [c for c in features.columns if c.startswith("gaze_")]
    assert len(eeg_names) == 4 and len(gaze_names) == 4
    assert np.isfinite(features[eeg_names + gaze_names]).all().all()
    with np.load(
        folder / "03_temporal_sequences.npz", allow_pickle=False
    ) as temporal:
        verify_temporal_feature_alignment(temporal, features)
        for field in ("eeg", "gaze"):
            assert temporal[field].shape == (len(features), 6, 4)
        assert temporal["eeg"].shape == temporal["gaze"].shape
        wrong_order = features.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="do not align"):
            verify_temporal_feature_alignment(temporal, wrong_order)


def test_classical_notebook_uses_same_validation_ids_and_no_asserts():
    """Stage 04 preserves original windows and checks model evidence."""
    path = (
        PROJECT / "code" / "notebooks"
        / "04_unimodal_baselines_and_classical_fusion.ipynb"
    )
    obj = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(c["source"]) for c in obj["cells"]
        if c["cell_type"] == "code"
    )
    assert '02_window_registry.csv' in code
    assert '03_features.csv' in code
    assert 'set(table.window_id) != set(expected.window_id)' in code
    assert 'matched' in code and 'validate="one_to_one"' in code
    assert 'prediction_table.duplicated(["model", "window_id"])' in code
    assert 'model_support.eq(len(valid)).all()' in code
    assert 'probabilities(model, x_val)' in code
    assert "y_test" not in code
    assert not any(
        line.strip().startswith("assert ") for line in code.splitlines()
    )


def test_alignment_notebook_preserves_scientific_role_checks_without_asserts():
    """Notebook 02 verifies source roles and explicitly diagnoses empty runs."""
    notebook = json.loads(
        (PROJECT / "code" / "notebooks"
         / "02_eeg_gaze_alignment_and_quality.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "split_manifest" in code and "split_registry" in code
    assert "missing_pairs" in code
    assert "eligible_windows.eq(0)" in code
    assert "ensure_disjoint_registry(registry)" in code
    assert not any(
        line.strip().startswith("assert ") for line in code.splitlines()
    )

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
    # Require transparent Test boundaries, not one historical status phrase.
    # Documentation must remain valid as individual stages are executed.
    assert "Validation" in readme and "Validation" in methods
    assert "Test" in readme and "Test" in methods
    assert "RUN_FINAL_TEST" in readme and "RUN_FINAL_TEST" in methods
    assert "100% accuracy" not in methods
