"""Fixed-label metrics, classical estimators and compact temporal models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ("nr", "tsr")
CLASS_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
CLASSICAL_ORDER = ("class_prior", "gaze_logistic", "eeg_logistic",
                   "early_fusion_logistic", "gaze_forest",
                   "eeg_forest", "late_equal_fusion")
NEURAL_ORDER = ("gaze_gru", "eeg_gru", "gated_fusion_gru")
MODEL_ORDER = CLASSICAL_ORDER + NEURAL_ORDER


def metrics(y_true, y_pred) -> dict:
    """Fixed NR/TSR classes, including when a class is never predicted."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    if not len(y):
        raise ValueError("Empty evaluation table")
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=[0, 1],
                                       average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p))}


def estimator(name: str, seed: int = 17):
    """Identical fixed training protocol; no Test-dependent hyperparameters."""
    if name == "class_prior":
        return DummyClassifier(strategy="prior")
    if name in ("gaze_logistic", "eeg_logistic", "early_fusion_logistic"):
        return make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(),
                             LogisticRegression(max_iter=2000,
                                                class_weight="balanced",
                                                random_state=seed))
    if name in ("gaze_forest", "eeg_forest"):
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=150,
                                                    min_samples_leaf=3,
                                                    class_weight="balanced",
                                                    random_state=seed,
                                                    n_jobs=-1))
    raise KeyError(f"Unsupported estimator {name}")


def feature_columns(table: pd.DataFrame, group: str) -> list[str]:
    prefix = {"gaze": "gaze_", "eeg": "eeg_", "fused": None}[group]
    columns = [col for col in table.columns
               if col.startswith(("gaze_", "eeg_"))]
    return [c for c in columns if prefix is None or c.startswith(prefix)]


def matrix(table: pd.DataFrame, group: str) -> np.ndarray:
    cols = feature_columns(table, group)
    if not cols:
        raise ValueError(f"No predictor columns for {group}")
    return table[cols].to_numpy(dtype=float)


def probabilities(model, x: np.ndarray) -> np.ndarray:
    """Return two columns in the immutable NR, TSR order."""
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(LABELS)), dtype=float)
    for source_col, class_id in enumerate(model.classes_):
        out[:, int(class_id)] = raw[:, source_col]
    return out


def record_predictions(table: pd.DataFrame, name: str,
                       probabilities_2d: np.ndarray) -> pd.DataFrame:
    if probabilities_2d.shape != (len(table), 2):
        raise ValueError("Expected [n_windows, 2] class probabilities")
    output = table[["window_id", "participant_id", "task", "split"]].copy()
    output["model"] = name
    output["target_index"] = table.task.map(CLASS_TO_INDEX).to_numpy()
    output["pred_index"] = probabilities_2d.argmax(axis=1)
    output["p_nr"] = probabilities_2d[:, 0]
    output["p_tsr"] = probabilities_2d[:, 1]
    return output


def torch_network(name: str, eeg_features: int, gaze_features: int,
                 hidden: int = 16):
    """Build the same compact architecture used for training and inference."""
    import torch
    from torch import nn

    class SingleGRU(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 2))

        def forward(self, eeg, gaze):
            series = gaze if name == "gaze_gru" else eeg
            _, last = self.gru(series)
            return self.head(last[-1])

    class GatedGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.eeg = nn.Linear(eeg_features, hidden)
            self.gaze = nn.Linear(gaze_features, hidden)
            self.gate = nn.Linear(2 * hidden, hidden)
            self.rnn = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 2))

        def forward(self, eeg, gaze):
            a = torch.tanh(self.eeg(eeg))
            b = torch.tanh(self.gaze(gaze))
            weight = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
            fused = weight * a + (1 - weight) * b
            _, last = self.rnn(fused)
            return self.head(last[-1])

    if name == "gaze_gru":
        return SingleGRU(gaze_features)
    if name == "eeg_gru":
        return SingleGRU(eeg_features)
    if name == "gated_fusion_gru":
        return GatedGRU()
    raise KeyError(name)


def predict_neural(model, eeg, gaze, device="cpu", batch_size=256):
    import torch
    model.to(device)
    model.eval()
    blocks = []
    with torch.no_grad():
        for i in range(0, len(eeg), batch_size):
            a = torch.as_tensor(eeg[i:i+batch_size], dtype=torch.float32,
                                device=device)
            b = torch.as_tensor(gaze[i:i+batch_size], dtype=torch.float32,
                                device=device)
            blocks.append(torch.softmax(model(a, b), dim=1).cpu().numpy())
    return np.vstack(blocks)
