# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Read verified normalised MPIIGaze eye images and matching gaze vectors.

The source MAT structure is checked explicitly. All participant, day and
sample identifiers are retained for grouped evaluation; none is a predictor.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


# ========================== 01. Source MATLAB Decoding =======================
def _unwrap(value):
    """Unwrap scalar MATLAB object containers without dropping observations."""
    while (isinstance(value, np.ndarray) and value.dtype == object
           and value.size == 1):
        value = value.item()
    return value


def _field(record, name):
    record = _unwrap(record)
    if isinstance(record, dict):
        return record.get(name)
    if hasattr(record, name):
        return getattr(record, name)
    if isinstance(record, np.void) and record.dtype.names:
        return record[name] if name in record.dtype.names else None
    if (isinstance(record, np.ndarray) and record.dtype.names
            and name in record.dtype.names):
        return record[name]
    return None


def _normalised_gaze(gaze):
    """Return (N, 3) gaze unit vectors in the source's normalised frame.

    MPIIGaze's normalised labels can use (theta, phi) radians:
    theta=asin(-y), phi=atan2(-x,-z), as given by the source provider.
    """
    values = np.asarray(_unwrap(gaze), dtype=np.float64)
    if values.ndim == 1 and values.size in (2, 3):
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D gaze-label matrix, got {values.shape}")
    if values.shape[1] not in (2, 3) and values.shape[0] in (2, 3):
        values = values.T
    if values.shape[1] == 2:
        if not np.isfinite(values).all() or np.max(np.abs(values)) > np.pi + 1e-5:
            raise ValueError("Expected finite MPIIGaze gaze angles in radians")
        theta, phi = values[:, 0], values[:, 1]
        vectors = np.column_stack((
            -np.cos(theta) * np.sin(phi),
            -np.sin(theta),
            -np.cos(theta) * np.cos(phi),
        ))
    elif values.shape[1] == 3:
        vectors = values
    else:
        raise ValueError(f"Expected two angles or a 3D gaze vector; got {values.shape}")
    lengths = np.linalg.norm(vectors, axis=1)
    if (not np.isfinite(vectors).all() or
            not np.isfinite(lengths).all() or (lengths <= 0).any()):
        raise ValueError("Missing or invalid gaze direction labels")
    return (vectors / lengths[:, None]).astype(np.float32)


def _image_stack(images, n_samples):
    """Resolve (N,H,W) and (H,W,N) eye patch arrays by label alignment."""
    images = _unwrap(images)
    if isinstance(images, np.ndarray) and images.dtype == object:
        frames = [np.asarray(_unwrap(frame)) for frame in images.flat]
        if len(frames) != n_samples:
            raise ValueError("Object-encoded image count differs from gaze labels")
        images = np.stack(frames)
    else:
        images = np.asarray(images)
    images = np.squeeze(images)
    if images.ndim == 2 and n_samples == 1:
        images = images[None, ...]
    if images.ndim != 3:
        raise ValueError(f"Expected a stack of grayscale eye images: {images.shape}")
    if images.shape[0] != n_samples:
        if images.shape[-1] == n_samples:
            images = np.moveaxis(images, -1, 0)
        else:
            raise ValueError("Source image and gaze label counts do not align")
    if (min(images.shape[1:]) < 8 or
            not np.isfinite(images).all()):
        raise ValueError("Invalid image shape or non-finite eye image pixels")
    images = np.asarray(images, dtype=np.float32)
    if images.max() <= 1.0 and images.min() >= 0:
        images = images * 255.0
    if images.min() < 0 or images.max() > 255.0:
        raise ValueError("Image pixels are outside the supported grayscale range")
    return np.rint(images).astype(np.uint8)


def read_normalised_eye_mat(path: Path, eye: str = "left"):
    """Read an original per-day MAT file as aligned uint8 images and 3D vectors.

    The selected eye determines the normalised viewpoint. No left/right
    swapping, image flipping or unknown geometry transformation is applied.
    MATLAB v5 data structs are supported; HDF5 source variants must be
    inspected and given a separately verified loader before model fitting.
    """
    if eye not in ("left", "right"):
        raise ValueError("eye must be 'left' or 'right'")
    path = Path(path)
    try:
        record = loadmat(path, simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError(
            "MATLAB v7.3 eye image schema requires a verified HDF5 decoder"
        ) from exc
    root = record.get("data")
    if root is None:
        raise ValueError(
            f"Expected original MPIIGaze variable 'data' in {path}")
    side = _field(root, eye)
    if side is None:
        raise ValueError(
            f"Expected a '{eye}' eye field in normalised MAT: {path}")
    gaze = _field(side, "gaze")
    images = _field(side, "image")
    if gaze is None or images is None:
        raise ValueError(
            f"Normalised {eye} MAT lacks paired 'image' and 'gaze': {path}")
    targets = _normalised_gaze(gaze)
    pixels = _image_stack(images, len(targets))
    if len(pixels) != len(targets):
        raise ValueError("Eye image–target row alignment failed")
    return pixels, targets


# ======================= 02. Deterministic Sample Selection ==================
def _participant_id(path: Path):
    parts = [part.lower() for part in path.parts if re.fullmatch(
        r"p\d{2}", part, flags=re.IGNORECASE)]
    if len(parts) != 1:
        raise ValueError(f"Expected one MPIIGaze participant path: {path}")
    return parts[0]


def build_eye_sample_cache(
    raw_dir: Path, *, eye: str = "left", per_person: int = 500,
    seed: int = 17,
):
    """Create a bounded deterministic per-person sample and image cache.

    Reservoir sampling observes each labelled source example once and
    limits memory use without selecting examples by their gaze outcomes.
    Source-published sample counts must not be inferred from MAT file totals.

    Returns (sample_rows, uint8 images, float32 gaze, source_counts).
    """
    if isinstance(per_person, bool) or per_person < 1:
        raise ValueError("per_person must be a positive integer")
    normalised = sorted(
        path for path in Path(raw_dir).rglob("*.mat")
        if "normalized" in {part.lower() for part in path.parts}
        and re.fullmatch(r"p\d{2}", path.parent.name, re.IGNORECASE)
    )
    if not normalised:
        raise FileNotFoundError(
            f"No official per-day Normalized/<participant>/*.mat files under {raw_dir}")
    reservoirs, observed, rngs = {}, {}, {}
    for path in normalised:
        person = _participant_id(path)
        if person not in reservoirs:
            reservoirs[person] = []
            observed[person] = 0
            rngs[person] = np.random.default_rng(
                np.random.SeedSequence([seed, int(person[1:])]))
        pixels, vectors = read_normalised_eye_mat(path, eye=eye)
        rng = rngs[person]
        relative = path.relative_to(raw_dir).as_posix()
        for i in range(len(pixels)):
            observed[person] += 1
            item = ({
                "sample_id": f"{person}:{relative}:{eye}:{i}",
                "participant_id": person,
                "source_mat": relative,
                "source_row": i,
                "eye": eye,
                "gaze_x": float(vectors[i, 0]),
                "gaze_y": float(vectors[i, 1]),
                "gaze_z": float(vectors[i, 2]),
            }, pixels[i], vectors[i])
            pool = reservoirs[person]
            if len(pool) < per_person:
                pool.append(item)
            else:
                position = int(rng.integers(0, observed[person]))
                if position < per_person:
                    pool[position] = item
    rows, images, gaze = [], [], []
    for person in sorted(reservoirs):
        for record, pixel, vector in sorted(
                reservoirs[person], key=lambda x: x[0]["sample_id"]):
            rows.append(record)
            images.append(pixel)
            gaze.append(vector)
    if not rows:
        raise ValueError("Original MPIIGaze source yielded no usable eye samples")
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise ValueError(
            f"Inconsistent normalised eye image dimensions: {sorted(shapes)}")
    return (rows, np.stack(images).astype(np.uint8),
            np.stack(gaze).astype(np.float32), observed)
