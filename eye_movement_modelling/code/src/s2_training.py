# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Training and record-aggregated validation for four gaze sequence models."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from s2_metrics import metric_row, record_probabilities
from s2_models import make_model, parameter_count


def seed_everything(seed: int = 17) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for_training() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_windows(model, x: np.ndarray, *,
                    device: torch.device, batch_size: int = 128) -> np.ndarray:
    """Return all four class probabilities without using source Test to fit."""
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(
                x[start:start + batch_size].astype(np.float32)
            ).to(device)
            result = torch.softmax(model(batch), dim=1)
            results.append(result.cpu().numpy())
    if not results:
        raise ValueError("No windows to score")
    return np.concatenate(results)


def fit_temporal(
    name: str, arrays: dict, split: pd.DataFrame,
    save_to: str | Path, *, seed: int = 17,
    epochs: int = 8, patience: int = 3,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit on train windows, select epoch on validation records only.

    The participant-disjoint Test is never used here. Best checkpoint is
    saved by validation record-level Macro F1, with loss as a tiebreaker.
    """
    seed_everything(seed)
    for key in ("x", "y", "record_id"):
        if key not in arrays:
            raise ValueError(f"Sequence archive missing {key}")
    if len(arrays["x"]) != len(arrays["record_id"]):
        raise ValueError("Window/source identifier misalignment")
    membership = dict(zip(split.record_id.astype(str), split.split))
    roles = np.asarray([membership[str(r)] for r in arrays["record_id"]])
    train, valid = roles == "train", roles == "validation"
    if not train.any() or not valid.any():
        raise ValueError("Need training and validation windows")
    device = device_for_training()
    model = make_model(name, channels=arrays["x"].shape[2]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01,
    )
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(arrays["x"][train].astype(np.float32)),
            torch.from_numpy(arrays["y"][train].astype(np.int64)),
        ),
        shuffle=True, batch_size=batch_size,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    x_valid = arrays["x"][valid].astype(np.float32)
    y_valid = arrays["y"][valid].astype(int)
    record_valid = arrays["record_id"][valid].astype(str)
    best_score, best_loss, best_epoch = -np.inf, np.inf, -1
    logs = []
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss, total = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss += loss.item() * len(batch_y)
            total += len(batch_y)
        valid_proba = predict_windows(model, x_valid, device=device)
        # Measured validation loss is window-based; selection metric is
        # record-based, with validation people disjoint from training.
        valid_loss = float(
            loss_function(
                torch.tensor(np.log(np.clip(valid_proba, 1e-9, 1.0)),
                             dtype=torch.float32),
                torch.tensor(y_valid, dtype=torch.long),
            ).item()
        )
        records = record_probabilities(
            record_valid, y_valid, valid_proba,
        )
        unique_people = split.loc[
            split.split.eq("validation"), "participant_id"
        ].nunique()
        row = metric_row(
            records.target_index, records.pred_index,
            model=name, partition="validation",
            n_participants=int(unique_people),
        )
        row["epoch"] = epoch
        row["train_loss"] = float(sum_loss / max(total, 1))
        row["validation_window_loss"] = valid_loss
        logs.append(row)
        score = row["macro_f1"]
        improved = (score > best_score + 1e-10) or (
            abs(score - best_score) <= 1e-10
            and valid_loss < best_loss
        )
        if improved:
            best_score, best_loss, best_epoch = score, valid_loss, epoch
            torch.save({
                "model_name": name,
                "channels": arrays["x"].shape[2],
                "state_dict": model.state_dict(),
                "best_epoch": epoch,
                "validation_macro_f1": best_score,
                "parameters": parameter_count(model),
            }, save_to)
        if epoch - best_epoch >= patience:
            break
    selected, info = load_temporal(save_to, device=device)
    final_probs = predict_windows(selected, x_valid, device=device)
    predictions = record_probabilities(
        record_valid, y_valid, final_probs,
    )
    predictions["model"] = name
    predictions["partition"] = "validation"
    return pd.DataFrame(logs), predictions, info


def load_temporal(path: str | Path, *, device=None):
    """Load only state-dict checkpoints produced by this repository."""
    device = device or device_for_training()
    info = torch.load(path, map_location=device, weights_only=True)
    model = make_model(
        info["model_name"], channels=int(info["channels"]),
    ).to(device)
    model.load_state_dict(info["state_dict"])
    model.eval()
    return model, info


def temporal_record_predictions(
    checkpoint: str | Path, arrays: dict, split: pd.DataFrame,
    *, partition: str,
) -> pd.DataFrame:
    """Predict once on the requested group only using frozen weights."""
    membership = dict(zip(split.record_id.astype(str), split.split))
    roles = np.asarray([membership[str(r)] for r in arrays["record_id"]])
    subset = roles == partition
    if not subset.any():
        raise ValueError(f"No sequence windows in {partition}")
    model, info = load_temporal(checkpoint)
    device = device_for_training()
    probabilities = predict_windows(model, arrays["x"][subset], device=device)
    frame = record_probabilities(
        arrays["record_id"][subset].astype(str),
        arrays["y"][subset].astype(int),
        probabilities,
    )
    frame["model"] = info["model_name"]
    frame["partition"] = partition
    return frame
