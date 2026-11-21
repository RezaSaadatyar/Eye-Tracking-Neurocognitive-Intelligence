# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Four deliberately small temporal gaze classifiers for matched comparisons."""
from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, channels: int = 4, hidden: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=7, padding=3),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden * 2, 4)

    def forward(self, x):
        return self.head(self.features(x.transpose(1, 2)).squeeze(-1))


class DilatedResidualBlock(nn.Module):
    """Non-causal dilated convolutions for a full-window offline task."""
    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.ReLU(), nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class TCN(nn.Module):
    def __init__(self, channels: int = 4, hidden: int = 32):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, kernel_size=1)
        self.temporal = nn.Sequential(
            DilatedResidualBlock(hidden, 1),
            DilatedResidualBlock(hidden, 2),
            DilatedResidualBlock(hidden, 4),
        )
        self.head = nn.Linear(hidden, 4)

    def forward(self, x):
        z = self.temporal(self.input(x.transpose(1, 2)))
        return self.head(z.mean(dim=-1))


class GazeGRU(nn.Module):
    def __init__(self, channels: int = 4, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(
            input_size=channels, hidden_size=hidden,
            num_layers=1, batch_first=True,
        )
        self.head = nn.Linear(hidden, 4)

    def forward(self, x):
        _, last = self.gru(x)
        return self.head(last[-1])


class GazeTransformer(nn.Module):
    def __init__(self, channels: int = 4, hidden: int = 32):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.embed = nn.Linear(channels, hidden)
        self.position = nn.Embedding(512, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=0.1, batch_first=True, activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False,
        )
        self.head = nn.Linear(hidden, 4)

    def forward(self, x):
        shorter = self.pool(x.transpose(1, 2)).transpose(1, 2)
        if shorter.size(1) > 512:
            raise ValueError("Transformer pooled window exceeds positions")
        loc = torch.arange(shorter.size(1), device=x.device)
        z = self.embed(shorter) + self.position(loc)[None, :, :]
        return self.head(self.encoder(z).mean(dim=1))


MODEL_NAMES = ("cnn", "tcn", "gru", "transformer")


def make_model(name: str, *, channels: int = 4):
    mapping = {
        "cnn": CNN1D, "tcn": TCN,
        "gru": GazeGRU, "transformer": GazeTransformer,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported model {name}; expected {MODEL_NAMES}")
    return mapping[name](channels=channels, hidden=32)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
