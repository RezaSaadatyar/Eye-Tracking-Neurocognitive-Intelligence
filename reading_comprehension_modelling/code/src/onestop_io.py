"""OneStop ordinary paragraph reports: acquisition, identities and cohort design."""
from pathlib import Path
import shutil
import zipfile
import csv
import numpy as np
import pandas as pd
import requests

SOURCE = "https://osf.io/zn9sq/"
RESOURCES = {"fixations": "ne4az", "ia": "xkgfz"}
KEY = ["participant_id", "trial_index"]
META = ["article_batch", "article_id", "paragraph_id", "difficulty_level",
        "onestopqa_question_id", "question_preview", "practice_trial",
        "repeated_reading_trial", "selected_answer", "is_correct", "paragraph"]
FIX = ["CURRENT_FIX_START", "CURRENT_FIX_DURATION", "CURRENT_FIX_X",
       "CURRENT_FIX_Y", "CURRENT_FIX_INTEREST_AREA_INDEX"]
IA = ["IA_ID", "IA_LABEL", "IA_DWELL_TIME", "IA_FIXATION_COUNT",
      "word_length", "wordfreq_frequency", "gpt2_surprisal"]


def acquire_reports(raw_dir):
    """Download the two official ordinary-reading ZIPs or reuse local reports."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for kind, osf_id in RESOURCES.items():
        target = raw_dir / f"{kind}_Paragraph.csv"
        if not target.exists():
            archive = raw_dir / f"{kind}_Paragraph.zip"
            if not archive.exists():
                temp = archive.with_suffix(".part")
                with requests.get(f"https://osf.io/download/{osf_id}/",
                                  stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with temp.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                if not zipfile.is_zipfile(temp):
                    raise ValueError(f"OSF response is not a ZIP: {osf_id}")
                temp.replace(archive)
            with zipfile.ZipFile(archive) as z:
                candidates = [n for n in z.namelist()
                              if Path(n).name.lower().startswith(f"{kind}_paragraph")
                              and n.lower().endswith(".csv")
                              and "__MACOSX" not in n and not Path(n).name.startswith("._")]
                if len(candidates) != 1:
                    raise ValueError(f"Expected one paragraph CSV in {archive}: {candidates}")
                temp = target.with_suffix(".part")
                with z.open(candidates[0]) as source, temp.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                temp.replace(target)
        reports[kind] = target
    return reports


def read_report(path, kind):
    """Inspect separator and select documented columns; never infer a new label."""
    required = KEY + META + (FIX if kind == "fixations" else IA)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    sep = "\t" if "\t" in header else ","
    columns = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig").columns
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{kind} report missing documented fields: {missing}")
    df = pd.read_csv(path, sep=sep, usecols=required, dtype=str,
                     encoding="utf-8-sig", na_values=["."], low_memory=False)
    if df[KEY].isna().any().any():
        raise ValueError(f"Missing trial identity in {kind}")
    return df


def boolean(series):
    return series.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})


def trial_metadata(df):
    """Require one outcome and one source identity per participant/trial."""
    cols = KEY + META
    unique = df[cols].drop_duplicates()
    if unique.duplicated(KEY).any():
        raise ValueError("Inconsistent metadata within a participant/trial")
    return unique


def make_cohort(fix, ia):
    """Match report trials and explicitly account for all excluded trials."""
    left, right = trial_metadata(fix), trial_metadata(ia)
    joined = left.merge(right, on=KEY, how="outer", suffixes=("", "_ia"),
                        indicator=True, validate="one_to_one")
    common = joined["_merge"].eq("both")
    for col in META:
        a, b = joined[col], joined[col + "_ia"]
        if not (a.eq(b) | (a.isna() & b.isna()))[common].all():
            raise ValueError(f"Report metadata disagree: {col}")
    reason = pd.Series("eligible", index=joined.index)
    reason.loc[~common] = "unmatched_report"
    for field, label in [("question_preview", "not_ordinary"),
                         ("practice_trial", "practice_or_unknown"),
                         ("repeated_reading_trial", "repeated_or_unknown")]:
        bad = ~boolean(joined[field]).eq(False).fillna(False)
        reason.loc[reason.eq("eligible") & bad] = label
    answer = joined.selected_answer.str.strip().str.upper()
    outcome = boolean(joined.is_correct)
    reason.loc[reason.eq("eligible") & (~answer.isin(list("ABCD")) | outcome.isna())] = "missing_answer"
    # Determine usable gaze support before freezing participant splits.
    numeric = fix[["CURRENT_FIX_START", "CURRENT_FIX_DURATION", "CURRENT_FIX_X",
                   "CURRENT_FIX_Y"]].apply(pd.to_numeric, errors="coerce")
    usable = np.isfinite(numeric).all(axis=1) & numeric.CURRENT_FIX_DURATION.gt(0)
    supported = pd.MultiIndex.from_frame(fix.loc[usable, KEY].drop_duplicates())
    has_gaze = pd.MultiIndex.from_frame(joined[KEY]).isin(supported)
    reason.loc[reason.eq("eligible") & ~has_gaze] = "no_valid_fixation"
    good = reason.eq("eligible")
    if not outcome[good].eq(answer[good].eq("A")).all():
        raise ValueError("Answer correctness disagrees with source annotation")
    trial = joined.loc[good, KEY + META].copy()
    if trial.empty or trial[["article_batch", "article_id", "paragraph_id",
                             "difficulty_level", "onestopqa_question_id", "paragraph"]].isna().any().any():
        raise ValueError("No usable cohort or missing source identity")
    trial["is_correct"] = outcome[good].astype(int)
    trial["trial_id"] = trial.participant_id + ":" + trial.trial_index
    if trial.trial_id.duplicated().any():
        raise ValueError("Duplicate trial IDs")
    trial["article_key"] = trial.article_batch + ":" + trial.article_id
    trial["paragraph_key"] = trial.article_key + ":" + trial.paragraph_id
    trial["question_key"] = trial.paragraph_key + ":" + trial.onestopqa_question_id
    if trial.groupby("participant_id").article_batch.nunique().max() != 1:
        raise ValueError("Participant appears in multiple article batches")
    # Never export answer choices or source text with derived model tables.
    cols = ["trial_id"] + KEY + ["article_batch", "article_key", "paragraph_key",
                                "question_key", "difficulty_level", "is_correct"]
    return trial[cols].sort_values("trial_id").reset_index(drop=True), (
        reason.value_counts().rename_axis("disposition").reset_index(name="n_trials"))


def make_splits(trials, seed=17):
    """A: unseen people within batches; B: final article batch held out entirely."""
    people = trials[["participant_id", "article_batch"]].drop_duplicates()
    if people.participant_id.duplicated().any():
        raise ValueError("Multiple batches per participant")
    batches = sorted(people.article_batch.unique())
    if len(batches) != 3:
        raise ValueError("Need three article batches for the two specified protocols")
    rng = np.random.default_rng(seed)
    rows = []
    for batch, group in people.groupby("article_batch", sort=True):
        ids = rng.permutation(sorted(group.participant_id))
        if len(ids) < 5:
            raise ValueError("Too few participants in an article batch")
        nv = max(1, round(len(ids)*.2))
        nt = max(1, round(len(ids)*.2))
        for i, p in enumerate(ids):
            a = "validation" if i < nv else "test" if i < nv + nt else "train"
            b = "test" if batch == batches[-1] else "validation" if i < nv else "train"
            rows.extend([("A_unseen_people", p, a), ("B_unseen_batch", p, b)])
    splits = pd.DataFrame(rows, columns=["protocol", "participant_id", "role"])
    validate_splits(trials, splits)
    return splits


def validate_trial_index(expected, observed):
    """Match the complete cohort and labels without relying on file order."""
    cols = list(expected.columns)
    if expected.trial_id.duplicated().any() or observed.trial_id.duplicated().any():
        raise ValueError("Duplicate trial identity")
    a = expected[cols].sort_values("trial_id").reset_index(drop=True).astype(str)
    b = observed[cols].sort_values("trial_id").reset_index(drop=True).astype(str)
    if not a.equals(b):
        raise ValueError("Feature/source identities or labels disagree")


def validate_splits(trials, splits):
    if trials.empty or trials.trial_id.isna().any() or trials.trial_id.duplicated().any():
        raise ValueError("Missing or duplicate trial identity")
    if set(splits.protocol) != {"A_unseen_people", "B_unseen_batch"}:
        raise ValueError("Unexpected evaluation protocols")
    if not set(splits.role).issubset({"train", "validation", "test"}):
        raise ValueError("Unexpected participant role")
    for protocol, group in splits.groupby("protocol"):
        if group.participant_id.duplicated().any() or set(group.participant_id) != set(trials.participant_id):
            raise ValueError("Incomplete or overlapping participant roles")
        joined = trials.merge(group, on="participant_id", validate="many_to_one")
        for role in ["train", "validation", "test"]:
            part = joined[joined.role.eq(role)]
            if set(part.is_correct) != {0, 1}:
                raise ValueError(f"Both outcome classes needed in {protocol}/{role}")
        if protocol == "B_unseen_batch":
            train = set(joined.loc[joined.role.ne("test"), "article_key"])
            test = set(joined.loc[joined.role.eq("test"), "article_key"])
            if train & test:
                raise ValueError("Article leakage in protocol B")
