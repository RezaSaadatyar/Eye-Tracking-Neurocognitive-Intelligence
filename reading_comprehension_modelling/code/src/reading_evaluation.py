"""Matched validation selection, held-out metrics and participant-cluster uncertainty."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score)
from reading_models import CLASSICAL, NEURAL


def metric(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= .5).astype(int)
    both = len(np.unique(y)) == 2
    return dict(n_trials=len(y), accuracy=accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, labels=[0,1], average="macro", zero_division=0),
                balanced_accuracy=balanced_accuracy_score(y,pred) if both else np.nan,
                auroc=roc_auc_score(y,p) if both else np.nan,
                incorrect_auprc=average_precision_score(1-y,1-p) if both else np.nan,
                incorrect_precision=precision_score(y,pred,pos_label=0,zero_division=0),
                incorrect_recall=recall_score(y,pred,pos_label=0,zero_division=0),
                brier=brier_score_loss(y,p))


def evaluate(table, splits, predictions):
    models = CLASSICAL + NEURAL
    if predictions.duplicated(["protocol", "model", "trial_id"]).any():
        raise ValueError("Repeated predictions")
    if not predictions.p_correct.between(0,1).all():
        raise ValueError("Invalid probability")
    joined = []
    for protocol, split in splits.groupby("protocol"):
        expected = table.merge(split[["participant_id","role"]], on="participant_id",
                               validate="many_to_one")
        expected = expected[expected.role.isin(["validation","test"])]
        rows = predictions[predictions.protocol.eq(protocol)]
        if set(rows.model) != set(models):
            raise ValueError("Missing model predictions")
        for model, part in rows.groupby("model"):
            if set(part.trial_id) != set(expected.trial_id):
                raise ValueError(f"Missing trial coverage: {protocol}/{model}")
        full = rows.merge(expected[["trial_id","role","is_correct","participant_id",
                                   "article_key","difficulty_level"]],
                          on=["trial_id","role"], validate="many_to_one")
        if len(full) != len(rows):
            raise ValueError("Prediction role mismatch")
        joined.append(full)
    data = pd.concat(joined, ignore_index=True)
    if len(data) != len(predictions):
        raise ValueError("Unexpected prediction protocol")
    val = pd.DataFrame([
        dict(protocol=protocol, model=model, n_people=g.participant_id.nunique(),
             **metric(g.is_correct, g.p_correct))
        for (protocol, model), g in data[data.role.eq("validation")].groupby(["protocol","model"])
    ])
    # Fix the Validation winner before calculating any Test performance.
    selected = val.sort_values(["macro_f1","model"],ascending=[False,True]).groupby(
        "protocol",sort=False).head(1).reset_index(drop=True)
    test, persons, strata, calib = [], [], [], []
    for (protocol, model), g in data[data.role.eq("test")].groupby(["protocol","model"]):
        test.append(dict(protocol=protocol, model=model, n_people=g.participant_id.nunique(),
                         **metric(g.is_correct, g.p_correct)))
        if not g.empty:
            for person, part in g.groupby("participant_id"):
                persons.append(dict(protocol=protocol,model=model,participant_id=person,
                                    **metric(part.is_correct, part.p_correct)))
            for level, part in g.groupby("difficulty_level"):
                strata.append(dict(protocol=protocol,model=model,difficulty=level,
                                   n_people=part.participant_id.nunique(),
                                   **metric(part.is_correct,part.p_correct)))
            g = g.assign(bin=pd.cut(g.p_correct, np.linspace(0,1,11), include_lowest=True))
            for bin_, part in g.groupby("bin",observed=True):
                calib.append(dict(protocol=protocol,model=model,probability_bin=str(bin_),
                                  n_trials=len(part),mean_probability=part.p_correct.mean(),
                                  observed_fraction=part.is_correct.mean()))
    return data, val, selected, pd.DataFrame(test), pd.DataFrame(persons), pd.DataFrame(strata), pd.DataFrame(calib)


def participant_bootstrap(data, draws=1000, seed=17):
    """Paired resampling of participant confusion counts, conditional on fixed fits."""
    rows, paired = [], []
    for protocol, frame in data[data.role.eq("test")].groupby("protocol"):
        people = sorted(frame.participant_id.unique())
        rng = np.random.default_rng(seed)
        sample = rng.integers(0,len(people),size=(draws,len(people)))
        values, point = {}, {}
        for model, g in frame.groupby("model"):
            point[model] = metric(g.is_correct,g.p_correct)["macro_f1"]
            counts = []
            for p in people:
                part = g[g.participant_id.eq(p)]
                y, pred = part.is_correct.to_numpy(int), (part.p_correct.to_numpy()>=.5).astype(int)
                counts.append(np.bincount(2*y+pred,minlength=4))
            arr = np.asarray(counts)[sample].sum(1)
            tn, fp, fn, tp = arr.T
            f0 = np.divide(2*tn,2*tn+fp+fn,out=np.zeros(draws),where=(2*tn+fp+fn)>0)
            f1 = np.divide(2*tp,2*tp+fp+fn,out=np.zeros(draws),where=(2*tp+fp+fn)>0)
            score = (f0+f1)/2
            values[model]=score
            low, high = np.quantile(score,[.025,.975])
            rows.append(dict(protocol=protocol,model=model,n_people=len(people),
                             macro_f1_low=low,macro_f1_high=high))
        # Matched modality contrasts within the same estimator family.
        for family in ("logistic","forest","xgb","tfidf"):
            for other in (("text",) if family=="tfidf" else ("text","gaze")):
                a,b=f"{family}_fusion",f"{family}_{other}"
                diff=values[a]-values[b]
                low,high=np.quantile(diff,[.025,.975])
                paired.append(dict(protocol=protocol,model_a=a,model_b=b,
                                   difference=point[a]-point[b],
                                   difference_low=low,difference_high=high))
    return pd.DataFrame(rows),pd.DataFrame(paired)
