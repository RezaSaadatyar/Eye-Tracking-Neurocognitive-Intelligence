"""Synthetic-only contract tests; not evidence of GazeBase execution."""
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "eye_tracking_foundations" / "code" / "src"))
sys.path.insert(0, str(PROJECT / "python_modules"))

from gaze_source import discover_records, read_record, read_stage
from gaze_preprocessing import preprocess_record
from gaze_streaming import preprocess_stage_file
from eye_events import ivt_labels, idt_labels, extract_events, label_agreement
from gaze_statistics import participant_summary, summarise_participant_metrics
from core.stage_outputs import ensure_stage_inputs


def _example_record(n=500):
    raw = pd.DataFrame({
        "record_id":"S_1001_S1_FXS", "participant_id":"001", "task":"FXS",
        "sample_index":np.arange(n), "n":np.arange(n, dtype=float),
        "x": np.r_[np.zeros(n//2), np.full(n-n//2,2.)],
        "y":np.zeros(n), "val":np.zeros(n,int),
        "lab":np.ones(n,int),
    })
    raw.loc[100:105,"val"]=1
    raw.loc[100:105,["x","y"]]=np.nan
    return raw


def test_invalid_gaze_breaks_continuity():
    prepared = preprocess_record(_example_record())
    assert prepared.loc[102, "valid_for_motion"] == False
    assert pd.isna(prepared.loc[106, "velocity_dva_s"])
    assert prepared.loc[99, "segment_id"] != prepared.loc[107, "segment_id"]
    assert prepared.loc[99, "x"] == 0.0


def test_ivt_idt_keep_unclassified_samples():
    frame = preprocess_record(_example_record())
    ivt = ivt_labels(frame)
    idt = idt_labels(frame)
    assert len(ivt) == len(frame)
    assert len(idt) == len(frame)
    assert ivt[102] == "unknown" and idt[102] == "unknown"
    assert "saccade" not in set(idt)
    events = extract_events(frame, ivt, "IVT")
    assert (events.duration_ms >= 0).all()
    agreement = label_agreement(frame, ivt)
    assert agreement["eligible_samples"] <= len(frame)


def test_participant_bootstrap_requires_independent_support():
    features = pd.DataFrame({
        "participant_id":["001", "001", "002"],
        "task":["FXS", "FXS", "FXS"],
        "mean_fixation_ms":[100.0, 120.0, 200.0],
    })
    grouped = participant_summary(features,("mean_fixation_ms",))
    assert len(grouped) == 2
    stats = summarise_participant_metrics(grouped,("mean_fixation_ms",),draws=150)
    assert stats.iloc[0].n_independent_participants == 2
    assert np.isfinite(stats.iloc[0].bootstrap_ci_lower)


def test_explicit_zip_restore_and_no_overwrite(tmp_path):
    output_dir = tmp_path / "eye_tracking_foundations" / "output"
    output_dir.mkdir(parents=True)
    upload = tmp_path / "uploads"
    upload.mkdir()
    expected_zip = upload / "eye_tracking_foundations_Notebook_01_Outputs.zip"
    with ZipFile(expected_zip,"w") as z:
        z.writestr("01_gaze_samples.csv", "record_id,sample_index\nexample,0\n")
    saved = ensure_stage_inputs(output_dir,1,("01_gaze_samples.csv",),upload_dir=upload)
    assert saved[0].read_text().startswith("record_id")
    assert len(ensure_stage_inputs(output_dir,1,("01_gaze_samples.csv",),upload_dir=upload))==1
    with pytest.raises(RuntimeError,match="Incomplete"):
        ensure_stage_inputs(output_dir,1,("01_gaze_samples.csv", "01_missing.csv"),upload_dir=upload)


def test_zip_path_traversal_rejected(tmp_path):
    folder = tmp_path / "eye_tracking_foundations" / "output"
    folder.mkdir(parents=True)
    upload = tmp_path / "upload"
    upload.mkdir()
    with ZipFile(upload/"eye_tracking_foundations_Notebook_02_Outputs.zip","w") as z:
        z.writestr("02_ok.csv", "x\n1\n")
        z.writestr("02_../wrong.csv", "unexpected\n")
    # The filename does not match flat path syntax in the helper.
    with pytest.raises(ValueError,match="Unsafe"):
        ensure_stage_inputs(folder,2,("02_ok.csv",),upload_dir=upload)


def test_ten_complete_participants_in_nested_round_layout(tmp_path):
    folder = tmp_path / "data" / "raw" / "gazebase"
    tasks = {
        "FXS": "Fixations",
        "HSS": "Horizontal_Saccades",
        "RAN": "Random_Saccades",
        "TEX": "Reading",
    }
    for subject in range(1001, 1012):
        # Last participant is incomplete and must not displace a complete one.
        for code, label in tasks.items():
            if subject == 1011 and code == "TEX":
                continue
            path = (folder / "GazeBase_v2_0" / "Round_1"
                    / f"Subject_{subject}" / f"S1_{label}"
                    / f"S_{subject}_S1_{code}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("n,x,y,val\n0,0,0,0\n1,0,0,0\n", encoding="utf-8")
    records = discover_records(folder)
    assert len(records) == 40
    assert records.participant_id.nunique() == 10
    assert set(records.task) == set(tasks)
    assert all("/S1_" in path.replace("\\", "/") for path in records.path)
    assert "011" not in set(records.participant_id)


def test_streaming_stage_matches_record_preprocessing(tmp_path):
    """Chunk boundaries do not alter sample-level scientific calculations."""
    a = _example_record(117)
    b = _example_record(117).copy()
    b["record_id"] = "S_1002_S1_FXS"
    b["participant_id"] = "002"
    b["n"] += 17
    rows = pd.concat([a, b], ignore_index=True)
    input_csv = tmp_path / "01_gaze_samples.csv"
    output_csv = tmp_path / "02_preprocessed_samples.csv"
    rows.to_csv(input_csv, index=False)
    summary, example = preprocess_stage_file(
        input_csv, output_csv, chunksize=13,
        expected_records={"S_1001_S1_FXS", "S_1002_S1_FXS"},
    )
    actual = pd.read_csv(output_csv, dtype={"participant_id": str})
    expected = pd.concat(
        [preprocess_record(a), preprocess_record(b)],
        ignore_index=True,
    )
    assert len(summary) == 2
    assert len(example) == 117
    assert len(actual) == len(rows)
    for col in ("x_smooth", "y_smooth", "velocity_dva_s", "segment_id"):
        np.testing.assert_allclose(
            actual[col].to_numpy(dtype=float),
            expected[col].to_numpy(dtype=float),
            equal_nan=True,
        )
    assert set(actual["record_id"]) == {"S_1001_S1_FXS", "S_1002_S1_FXS"}


def test_streaming_rejects_disordered_repeated_record(tmp_path):
    a = _example_record(20)
    b = a.copy()
    b["record_id"] = "S_1002_S1_FXS"
    source = tmp_path / "01_gaze_samples.csv"
    destination = tmp_path / "02_preprocessed_samples.csv"
    pd.concat([a, b, a], ignore_index=True).to_csv(source, index=False)
    with pytest.raises(ValueError, match="non-contiguous"):
        preprocess_stage_file(source, destination, chunksize=7)
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()
