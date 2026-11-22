"""Modality-separated, non-overlapping EEG and gaze window features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch

from alignment import eye_sample_times

EEG_FIELDS = ("theta_log_power", "alpha_log_power",
              "beta_log_power", "eeg_rms_uv")
GAZE_FIELDS = ("valid_fraction", "speed_median_px_s",
               "pupil_median", "fixation_fraction")


def eeg_channels(raw, count: int = 16) -> list[str]:
    """Select a bounded fixed set of actual EEG channels, not a montage claim."""
    candidates = [ch for ch in raw.ch_names if ch.startswith("E") and
                  ch[1:].isdigit()]
    if len(candidates) < 4:
        raise ValueError("Insufficient actual EEG channels")
    return candidates[:min(count, len(candidates))]


def clean_eeg(raw, start: float, stop: float, channels: list[str]) -> np.ndarray:
    """Read only one window, average-reference and bandpass 1–40 Hz."""
    sfreq = float(raw.info["sfreq"])
    first = max(0, int(round(start * sfreq)))
    last = min(raw.n_times, int(round(stop * sfreq)))
    if last - first < 4 * sfreq:
        raise ValueError("Insufficient EEG window samples")
    data = raw.get_data(picks=channels, start=first, stop=last)
    if not np.isfinite(data).all():
        raise ValueError("EEG window has non-finite samples")
    data = data - np.mean(data, axis=0, keepdims=True)
    high = min(40., sfreq / 2 - 1)
    filt = butter(4, [1., high], btype="bandpass", fs=sfreq,
                  output="sos")
    return sosfiltfilt(filt, data, axis=-1)


def summarise_eeg(values: np.ndarray, sfreq: float) -> np.ndarray:
    """Mean channel power (theta/alpha/beta) and RMS, no claimed localisation."""
    hz, psd = welch(values, sfreq, nperseg=min(values.shape[-1],
                                              int(sfreq)), axis=-1)
    res = []
    for low, high in ((4, 8), (8, 13), (13, 30)):
        included = (hz >= low) & (hz < high)
        power = trapezoid(psd[:, included], hz[included], axis=-1)
        res.append(np.log10(np.mean(power) + 1e-16))
    res.append(float(np.sqrt(np.mean(values ** 2)) * 1e6))
    return np.asarray(res, dtype=np.float32)


def summarise_eye(eye: pd.DataFrame, eye_t: np.ndarray,
                  events: pd.DataFrame, start: float, stop: float) -> np.ndarray:
    mask = (eye_t >= start) & (eye_t < stop)
    xy = eye.loc[mask, ["eye1_x_coordinate", "eye1_y_coordinate"]].apply(
        pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(xy).all(axis=1)
    fraction = float(valid.mean()) if len(valid) else 0.
    freq = 1 / float(np.median(np.diff(eye_t)))
    dt = np.diff(xy, axis=0)
    transitions = valid[1:] & valid[:-1]
    speed = np.linalg.norm(dt[transitions], axis=1) * freq
    speed = speed[np.isfinite(speed)]
    median_speed = float(np.median(speed)) if len(speed) else np.nan
    pupil = pd.to_numeric(eye.loc[mask, "eye1_pupil_size"],
                          errors="coerce").to_numpy(dtype=float)
    pupil = pupil[np.isfinite(pupil)]
    pupil_med = float(np.median(pupil)) if len(pupil) else np.nan
    ev = events.copy()
    onset = pd.to_numeric(ev.onset, errors="coerce")
    duration = pd.to_numeric(ev.duration, errors="coerce")
    fix = ev.trial_type.eq("fixation")
    overlap = (np.minimum(onset + duration, stop) -
               np.maximum(onset, start)).clip(lower=0)
    fix_frac = float(overlap[fix].sum() / (stop - start))
    return np.asarray([fraction, median_speed, pupil_med,
                       min(fix_frac, 1.)], dtype=np.float32)


def extract_window(raw, eye, events, eye_meta, row,
                   eeg_picks: list[str], bin_seconds: float = 2.) -> tuple:
    """Aggregate six 2-s bins of a 12-s paired window."""
    start, stop = float(row["start_seconds"]), float(row["end_seconds"])
    length = stop - start
    bins = int(round(length / bin_seconds))
    if bins < 1 or not np.isclose(bins * bin_seconds, length):
        raise ValueError("Window does not divide into equal temporal bins")
    sfreq = float(raw.info["sfreq"])
    eeg_window = clean_eeg(raw, start, stop, eeg_picks)
    eye_t = eye_sample_times(eye_meta, len(eye))
    signal, gaze = [], []
    for i in range(bins):
        b0, b1 = start + i * bin_seconds, start + (i+1)*bin_seconds
        lo = int(round(i * bin_seconds * sfreq))
        hi = int(round((i + 1) * bin_seconds * sfreq))
        signal.append(summarise_eeg(eeg_window[:, lo:hi], sfreq))
        gaze.append(summarise_eye(eye, eye_t, events, b0, b1))
    return np.stack(signal), np.stack(gaze)


def verify_temporal_feature_alignment(
    sequences, table: pd.DataFrame, expected_bins: int = 6
) -> None:
    """Check original-unit EEG/gaze sequence order against recorded means.

    Source CSV means were calculated from the same numerical temporal arrays.
    Comparing their per-window means verifies row identity without loading
    object-encoded text inside an older NPZ with unsafe pickle enabled.
    """
    if table.empty or table.window_id.duplicated().any():
        raise ValueError("Empty or duplicated paired feature identities")
    for key, fields in (("eeg", EEG_FIELDS), ("gaze", GAZE_FIELDS)):
        array = np.asarray(sequences[key])
        expected_shape = (len(table), expected_bins, len(fields))
        if array.shape != expected_shape or array.dtype.kind not in "fi":
            raise ValueError(
                f"Invalid {key} temporal shape or dtype: {array.shape}"
            )
        summary = table[
            [f"{key}_{field}" for field in fields]
        ].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(array, axis=1)
        if not np.allclose(
            means, summary, rtol=3e-5, atol=1e-4, equal_nan=True
        ):
            raise ValueError(
                f"{key.upper()} temporal rows do not align with "
                "the saved per-window feature means"
            )
