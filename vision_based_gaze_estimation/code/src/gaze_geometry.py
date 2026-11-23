# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Norm-checked 3D gaze angular error in degrees.

This module supplies a geometric measurement only; it does not resolve a
dataset's coordinate frame, camera calibration or gaze-vector convention.
"""

from __future__ import annotations

import numpy as np


# ======================= 01. Angular Error in Degrees ========================
def angular_error_degrees(reference, predicted):
    """Return angular errors between matching (..., 3) vector arrays.

    Both vectors must be non-zero, finite and expressed in the same frame.
    The result is in degrees; it does not aggregate across participants.
    """
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if (reference.ndim < 1 or predicted.shape != reference.shape
            or reference.shape[-1] != 3):
        raise ValueError("Expected matching non-empty (..., 3) arrays")
    if not reference.size or not (np.isfinite(reference).all()
                                  and np.isfinite(predicted).all()):
        raise ValueError("Gaze vectors are empty or non-finite")
    norms = np.linalg.norm(reference, axis=-1) * np.linalg.norm(
        predicted, axis=-1)
    if (not np.isfinite(norms).all() or (norms <= 0).any()):
        raise ValueError("Gaze vectors must have finite, non-zero norms")
    cosine = np.sum(reference * predicted, axis=-1) / norms
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
