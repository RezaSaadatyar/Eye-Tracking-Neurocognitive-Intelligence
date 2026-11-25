"""Explicit pre-answer feature allowlists; trial summaries and ordered fixations."""
import numpy as np
import pandas as pd

GAZE = ["fix_count", "fix_total_ms", "fix_mean_ms", "fix_sd_ms",
        "x_sd_px", "y_sd_px", "path_px", "backward_word_rate",
        "visited_word_fraction", "mean_word_dwell_ms"]
TEXT = ["word_count", "mean_word_chars", "punctuation_fraction", "advanced"]
SEQ = ["CURRENT_FIX_DURATION", "CURRENT_FIX_X", "CURRENT_FIX_Y",
       "CURRENT_FIX_INTEREST_AREA_INDEX"]
ID = ["participant_id", "trial_index"]


def _numeric(frame, columns):
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan)


def build_features(trials, fix, ia, max_fixations=256):
    """Use all IA words for text summaries; fixation order is source start time."""
    keep = trials[ID + ["trial_id"]]
    fix = fix.merge(keep, on=ID, validate="many_to_one")
    ia = ia.merge(keep, on=ID, validate="many_to_one")
    rows, sequences, masks = [], [], []
    fg, ig = fix.groupby("trial_id"), ia.groupby("trial_id")
    for trial in trials.itertuples(index=False):
        f, w = fg.get_group(trial.trial_id), ig.get_group(trial.trial_id)
        a = _numeric(f, ["CURRENT_FIX_START"] + SEQ)
        a = a.dropna(subset=["CURRENT_FIX_START", *SEQ[:3]])
        a = a[a.CURRENT_FIX_DURATION.gt(0)].sort_values("CURRENT_FIX_START", kind="stable")
        if a.empty:
            raise ValueError(f"Frozen cohort lacks valid gaze: {trial.trial_id}")
        words = _numeric(w, ["IA_ID", "IA_DWELL_TIME", "IA_FIXATION_COUNT",
                             "word_length", "wordfreq_frequency", "gpt2_surprisal"])
        if words.IA_ID.isna().any() or words.IA_ID.duplicated().any():
            raise ValueError(f"Invalid IA identity: {trial.trial_id}")
        if words.empty or w.paragraph.nunique() != 1:
            raise ValueError("Missing paragraph text")
        # Text-only statistics derive from the presented paragraph or full IA report,
        # never from the subset of fixated words or dwell-weighted text.
        paragraph = w.paragraph.iloc[0]
        tokens = paragraph.split()
        if not tokens:
            raise ValueError("Empty paragraph")
        # Source lexical summaries are descriptive only, not model inputs:
        # report coverage and layout segmentation may vary between trials.
        full_words = len(words) == len(tokens)
        d = a.CURRENT_FIX_DURATION.to_numpy()
        xy = a[["CURRENT_FIX_X", "CURRENT_FIX_Y"]].to_numpy()
        wi = a.CURRENT_FIX_INTEREST_AREA_INDEX
        observed = wi.where(wi.isin(words.IA_ID)).to_numpy()
        steps = np.diff(observed)
        valid_steps = np.isfinite(steps) & (steps != 0)
        valid_ia = words.IA_FIXATION_COUNT.dropna()
        if (valid_ia < 0).any():
            raise ValueError("Negative fixation counts")
        row = dict(trial_id=trial.trial_id, fix_count=len(d), fix_total_ms=d.sum(),
                   fix_mean_ms=d.mean(), fix_sd_ms=d.std(),
                   x_sd_px=xy[:, 0].std(), y_sd_px=xy[:, 1].std(),
                   path_px=np.linalg.norm(np.diff(xy, axis=0), axis=1).sum(),
                   backward_word_rate=(steps[valid_steps] < 0).mean() if valid_steps.any() else np.nan,
                   visited_word_fraction=valid_ia.gt(0).mean() if len(valid_ia) else np.nan,
                   mean_word_dwell_ms=words.IA_DWELL_TIME.mean(),
                   word_count=len(tokens), mean_word_chars=np.mean([len(t) for t in tokens]),
                   punctuation_fraction=sum(not c.isalnum() and not c.isspace()
                                            for c in paragraph)/len(paragraph),
                   mean_word_frequency_bits=words.wordfreq_frequency.mean() if full_words else np.nan,
                   mean_word_surprisal_bits=words.gpt2_surprisal.mean() if full_words else np.nan,
                   advanced=int(trial.difficulty_level == "Adv"),
                   sequence_truncated=int(len(a) > max_fixations))
        if trial.difficulty_level not in ("Adv", "Ele"):
            raise ValueError("Unexpected difficulty label")
        # Retain missing/off-word IA indices as -1, not as a genuine word index.
        arr = a[SEQ].to_numpy(dtype=np.float32)
        arr[:, 3] = np.where(np.isfinite(observed), observed, -1)
        arr = arr[:max_fixations]
        padded = np.zeros((max_fixations, len(SEQ)), np.float32)
        mask = np.zeros(max_fixations, bool)
        padded[:len(arr)], mask[:len(arr)] = arr, True
        rows.append(row)
        sequences.append(padded)
        masks.append(mask)
    if not rows:
        raise ValueError("No usable fixations")
    table = trials.merge(pd.DataFrame(rows), on="trial_id", validate="one_to_one")
    if len(table) != len(trials) or list(table.trial_id) != [x["trial_id"] for x in rows]:
        raise ValueError("Sequence and trial order disagree")
    return table, np.stack(sequences), np.stack(masks)


def load_sequences(path, table):
    with np.load(path, allow_pickle=False) as cache:
        if not np.array_equal(cache["trial_id"], table.trial_id.to_numpy(str)):
            raise ValueError("Sequence cache does not match feature table")
        x, mask = cache["x"], cache["mask"]
    if (len(x) != len(table) or x.ndim != 3 or x.shape[-1] != len(SEQ)
            or x.shape[:2] != mask.shape or mask.dtype != bool
            or not mask.any(axis=1).all() or not np.isfinite(x).all()
            or (np.diff(mask.astype(int), axis=1) > 0).any()):
        raise ValueError("Invalid sequence shape, mask or values")
    return x, mask
