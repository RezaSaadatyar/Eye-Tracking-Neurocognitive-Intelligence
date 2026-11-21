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
from gaze_streaming import preprocess_stage_file, validate_notebook_01_inputs
from gaze_diagnostics import inspect_velocity_peaks
from eye_events import ivt_labels, idt_labels, extract_events, label_agreement
from gaze_features import record_features, feature_dictionary
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
        expected_record_counts={"S_1001_S1_FXS": 117, "S_1002_S1_FXS": 117},
    )
    actual = pd.read_csv(output_csv, dtype={"participant_id": str})
    expected = pd.concat(
        [preprocess_record(a), preprocess_record(b)],
        ignore_index=True,
    )
    assert len(summary) == 2
    assert (summary["n_segments"] == 2).all()  # Invalid spans are not valid runs.
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


def test_notebook_01_compact_input_alignment():
    manifest = pd.DataFrame({
        "record_id": ["S_1001_S1_FXS", "S_1002_S1_FXS"],
        "participant_id": ["001", "002"],
        "task": ["FXS", "FXS"],
    })
    quality = manifest.copy()
    quality["n_samples"] = [117, 119]
    assert validate_notebook_01_inputs(manifest, quality) == {
        "S_1001_S1_FXS": 117, "S_1002_S1_FXS": 119,
    }
    wrong = quality.copy()
    wrong.loc[1, "task"] = "TEX"
    with pytest.raises(ValueError, match="task differs"):
        validate_notebook_01_inputs(manifest, wrong)
    wrong = quality.copy()
    wrong.loc[1, "n_samples"] = 0
    with pytest.raises(ValueError, match="positive integers"):
        validate_notebook_01_inputs(manifest, wrong)


def test_streaming_rejects_notebook_01_row_count_mismatch(tmp_path):
    record = _example_record(117)
    source = tmp_path / "01_gaze_samples.csv"
    destination = tmp_path / "02_preprocessed_samples.csv"
    record.to_csv(source, index=False)
    with pytest.raises(ValueError, match="source row count mismatch"):
        preprocess_stage_file(
            source, destination, chunksize=19,
            expected_records={"S_1001_S1_FXS"},
            expected_record_counts={"S_1001_S1_FXS": 118},
        )
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_notebook_zip_restore_uses_bounded_streaming(tmp_path, monkeypatch):
    folder = tmp_path / "eye_tracking_foundations" / "output"
    folder.mkdir(parents=True)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    payload = b"a,b,c\n" + b"1,2,3\n" * 400_000
    with ZipFile(uploads / "eye_tracking_foundations_Notebook_01_Outputs.zip", "w") as archive:
        archive.writestr("01_gaze_samples.csv", payload)
        archive.writestr("01_source_manifest.csv", "record_id\nS_1001_S1_FXS\n")
    def forbid_read(self, *args, **kwargs):
        raise AssertionError("ZIP content must not be materialised by ZipFile.read")
    monkeypatch.setattr(ZipFile, "read", forbid_read)
    restored = ensure_stage_inputs(
        folder, 1, ("01_gaze_samples.csv", "01_source_manifest.csv"),
        upload_dir=uploads,
    )
    assert restored[0].stat().st_size == len(payload)
    assert restored[0].read_bytes() == payload


def _synthetic_peak_records(tmp_path):
    frames = []
    for subject, amplitude in ((1001, 100.), (1002, 80.), (1003, 40.), (1004, 10.)):
        x = np.zeros(150)
        x[62:] = amplitude
        raw = pd.DataFrame({
            "record_id": f"S_{subject}_S1_FXS",
            "participant_id": f"{subject-1000:03d}",
            "task": "FXS",
            "sample_index": np.arange(150),
            "n": np.arange(150),
            "x": x,
            "y": np.zeros(150),
            "val": np.zeros(150, dtype=int),
        })
        frames.append(preprocess_record(raw))
    prepared = pd.concat(frames, ignore_index=True)
    path = tmp_path / "02_preprocessed_samples.csv"
    prepared.to_csv(path, index=False)
    summary = (
        prepared.groupby("record_id").velocity_dva_s.max()
        .reset_index(name="max_velocity_dva_s")
    )
    return path, summary


def test_velocity_peak_windows_match_original_rows(tmp_path):
    """Windows remain record-local and retain the raw validity/time evidence."""
    source, summary = _synthetic_peak_records(tmp_path)
    review, windows = inspect_velocity_peaks(
        source, summary, top_records=3, half_window_samples=40, chunksize=17
    )
    assert review.record_id.tolist() == [
        "S_1001_S1_FXS", "S_1002_S1_FXS", "S_1003_S1_FXS"
    ]
    assert len(windows) == 243
    assert (review.peak_sample_index == 62).all()
    assert (review.delta_t_ms == 1).all()
    assert (review.peak_val == 0).all()
    assert (review.previous_val == 0).all()
    assert review.same_valid_run.all()
    np.testing.assert_allclose(
        review.peak_velocity_dva_s, review.reconstructed_velocity_dva_s
    )


def test_velocity_peak_review_rejects_mismatched_summary(tmp_path):
    source, summary = _synthetic_peak_records(tmp_path)
    altered = summary.copy()
    altered.loc[
        altered.record_id.eq("S_1001_S1_FXS"), "max_velocity_dva_s"
    ] = 999999.0
    with pytest.raises(ValueError, match="Velocity maximum differs"):
        inspect_velocity_peaks(source, altered, top_records=3, chunksize=17)


def test_scanpath_features_follow_event_time_order():
    """Per-record path length must use source event order, not CSV row order."""
    record = pd.DataFrame({
        "record_id": ["S_1001_S1_FXS"] * 6,
        "participant_id": ["001"] * 6,
        "task": ["FXS"] * 6,
        "valid_for_motion": [True, True, True, True, False, True],
        "dP": [10.0, 12.0, 14.0, 16.0, np.nan, -1.0],
    })
    events = pd.DataFrame({
        "record_id": ["S_1001_S1_FXS"] * 4,
        "method": ["IVT"] * 4,
        "event_type": ["fixation", "fixation", "fixation", "saccade"],
        "start_sample": [300, 100, 200, 250],
        "duration_ms": [120.0, 110.0, 100.0, 8.0],
        "mean_x_dva": [3.0, 0.0, 3.0, np.nan],
        "mean_y_dva": [4.0, 0.0, 0.0, np.nan],
        "peak_speed_dva_s": [np.nan, np.nan, np.nan, 125.0],
    })
    observed = record_features(record, events)
    assert observed["n_fixations_ivt"] == 3
    assert observed["n_saccades_ivt"] == 1
    assert observed["scanpath_length_dva"] == pytest.approx(7.0)
    assert observed["valid_fraction"] == pytest.approx(5 / 6)
    assert observed["pupil_median_uncalibrated"] == pytest.approx(13.0)
    assert "scanpath_length_dva" in feature_dictionary().feature.tolist()
