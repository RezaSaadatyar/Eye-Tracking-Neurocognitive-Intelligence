"""Small geometry and structure contracts; synthetic vectors only."""

from pathlib import Path
import io
import json
import sys
import tarfile
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "vision_based_gaze_estimation"
sys.path.insert(0, str(STUDY / "code" / "src"))
from gaze_geometry import angular_error_degrees
from mpiigaze_io import (source_inventory, participant_source_support,
                          extract_source_archive)


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
    for i, name in enumerate(names):
        path = STUDY / "code" / "notebooks" / (name + ".ipynb")
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert nb["cells"][0]["cell_type"] == "markdown"
        if i == 0:
            assert len(nb["cells"]) == 21
            assert len([c for c in nb["cells"] if c["cell_type"] == "code"]) == 10
            source = [c["source"] for c in nb["cells"]
                      if c["cell_type"] == "code"]
            assert all("".join(code).splitlines()[0].startswith("# ===")
                       for code in source)
            assert "ensure_mpiigaze_source(raw_dir)" in "".join(source[2])
            assert "setup_colab(\"vision_based_gaze_estimation\")" in (
                "".join(source[1]))
    study_readme = (STUDY / "README.md").read_text(encoding="utf-8")
    methods = (STUDY / "RESEARCH_METHODS_AND_EVIDENCE.md").read_text(
        encoding="utf-8")
    assert sum(line.startswith("## ") for line in study_readme.splitlines()) == 6
    assert sum(line.startswith("## ") for line in methods.splitlines()) == 9


def test_mpiigaze_source_manifest_and_participant_support(tmp_path):
    """A file inventory reports containers, never invented image counts."""
    normal = tmp_path / "Data" / "Normalized" / "p00"
    original = tmp_path / "Data" / "Original" / "p00" / "Day01"
    normal.mkdir(parents=True)
    original.mkdir(parents=True)
    (normal / "day01.mat").write_bytes(b"not a real MATLAB header")
    (original / "eye.jpg").write_bytes(b"synthetic")
    table = source_inventory(tmp_path)
    assert len(table) == 2
    assert set(table.component) == {"Normalized", "Original"}
    support = participant_source_support(table)
    assert len(support) == 1
    assert support.iloc[0].participant_id == "p00"
    assert support.iloc[0].normalised_mat_files == 1
    assert support.iloc[0].observed_day_folders == 1


def test_mpiigaze_safe_zip_extract_and_no_overwrite(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("MPIIGaze/Data/Normalized/p00/day01.mat", b"synthetic")
    destination = tmp_path / "extracted"
    extract_source_archive(archive, destination)
    assert (destination / "MPIIGaze/Data/Normalized/p00/day01.mat").is_file()
    with pytest.raises(FileExistsError):
        extract_source_archive(archive, destination)


def test_mpiigaze_rejects_zip_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", b"forbidden")
    destination = tmp_path / "source"
    with pytest.raises(ValueError, match="Unsafe archive"):
        extract_source_archive(archive, destination)
    assert not (tmp_path / "outside.txt").exists()
    assert not destination.exists()


def test_mpiigaze_safe_tar_root_and_rejects_symlink(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        root = tarfile.TarInfo("./")
        root.type = tarfile.DIRTYPE
        tar.addfile(root)
        payload = b"synthetic"
        entry = tarfile.TarInfo("./Data/Normalized/p00/day01.mat")
        entry.size = len(payload)
        tar.addfile(entry, io.BytesIO(payload))
    destination = tmp_path / "unpacked"
    extract_source_archive(archive, destination)
    assert (destination / "Data/Normalized/p00/day01.mat").is_file()

    malicious = tmp_path / "link.tar.gz"
    with tarfile.open(malicious, "w:gz") as tar:
        entry = tarfile.TarInfo("link")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../outside"
        tar.addfile(entry)
    with pytest.raises(ValueError, match="Links"):
        extract_source_archive(malicious, tmp_path / "bad")


def test_archive_companions_excluded_from_day_mat_selection(tmp_path):
    from mpiigaze_io import normalised_mat_files, is_normalised_mat_path

    relative = Path("MPIIGaze/Data/Normalized/p00/day01.mat")
    sidecars = [
        Path("__MACOSX") / relative,
        relative.with_name("._day01.mat"),
        relative.with_name("notes.mat"),
    ]
    for item in [relative, *sidecars]:
        path = tmp_path / item
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
    assert normalised_mat_files(tmp_path) == [tmp_path / relative]
    assert all(not is_normalised_mat_path(item) for item in sidecars)
    support = participant_source_support(source_inventory(tmp_path))
    assert support.iloc[0].normalised_mat_candidate_paths == 4
    assert support.iloc[0].normalised_mat_files == 1
