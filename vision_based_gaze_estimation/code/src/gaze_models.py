# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Classical image descriptors and compact gaze-direction neural models.

Images are source-normalised grayscale eye patches; augmentation, fitting
and checkpoint selection use only the inner training/validation people.
"""

from __future__ import annotations

import copy

import numpy as np


# ==================== 01. Handcrafted Image Baseline =========================
def hog_descriptors(images):
    """Convert uint8 (N,H,W) eye patches to fixed-length HOG descriptors."""
    from skimage.feature import hog

    if images.ndim != 3 or min(images.shape[1:]) < 12:
        raise ValueError("Expected grayscale eye image arrays (N,H,W)")
    descriptors = []
    for image in images:
        feature = hog(
            image.astype(np.float32) / 255.0,
            orientations=6, pixels_per_cell=(6, 6),
            cells_per_block=(1, 1), feature_vector=True,
        )
        descriptors.append(feature)
    return np.stack(descriptors).astype(np.float32)


def normalise_directions(vectors):
    """Convert predicted 3D gaze regressions to finite unit vectors."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Expected (N,3) 3D gaze predictions")
    norms = np.linalg.norm(vectors, axis=1)
    if (not np.isfinite(norms).all() or (norms < 1e-10).any()):
        raise ValueError("Predicted 3D gaze vectors must be non-zero and finite")
    return (vectors / norms[:, None]).astype(np.float32)


def fit_ridge_and_predict(train_features, targets, test_features, alpha):
    """Train preprocessing and Ridge exclusively on permitted fit rows."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    model.fit(train_features, targets)
    return model, normalise_directions(model.predict(test_features))


# ==================== 02. Compact Gaze CNN and ResNet-18 =====================
def build_gaze_network(name, *, pretrained=False):
    """Build a (N,1,H,W) -> (N,3) model; download weights only if requested."""
    import torch
    from torch import nn

    if name == "small_cnn":
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 3)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
        # Preserve useful pretrained RGB filters by averaging across channels.
        if weights is not None:
            pretrained_model = resnet18(weights=weights)
            with torch.no_grad():
                model.conv1.weight.copy_(
                    pretrained_model.conv1.weight.mean(dim=1, keepdim=True))
        model.fc = nn.Linear(model.fc.in_features, 3)
        return model
    raise ValueError("Unknown gaze CNN model")


def _image_loader(images, targets, indices, batch_size, shuffle):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    pixel = torch.tensor(
        images[indices, None, :, :].astype(np.float32) / 255.0)
    truth = torch.tensor(targets[indices], dtype=torch.float32)
    return DataLoader(
        TensorDataset(pixel, truth), batch_size=batch_size,
        shuffle=shuffle, num_workers=0)


def predict_network(model, images, indices, device, *, batch_size=128):
    """Predict and normalise 3D vectors without accessing any target labels."""
    import torch

    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            x = torch.tensor(images[
                indices[start:start + batch_size], None, :, :
            ], device=device, dtype=torch.float32) / 255.0
            y = torch.nn.functional.normalize(
                model(x), p=2, dim=1, eps=1e-8)
            result.append(y.cpu().numpy())
    return normalise_directions(np.concatenate(result))


# ======================= 03. Inner Validation Training =======================
def train_gaze_network(
    name, images, targets, fit_rows, valid_rows, *,
    device, epochs=8, batch_size=128, seed=17,
    pretrained=False, learning_rate=1e-3, patience=3,
):
    """Select an epoch on inner Validation, never on outer Test labels."""
    import torch

    from gaze_geometry import angular_error_degrees

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    network = build_gaze_network(name, pretrained=pretrained).to(device)
    optimiser = torch.optim.AdamW(
        network.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = _image_loader(images, targets, fit_rows, batch_size, True)
    history, best_weights, best_error, best_epoch = [], None, float("inf"), None
    stale = 0
    for epoch in range(1, epochs + 1):
        network.train()
        loss_sum, observed = 0.0, 0
        for image, gaze in loader:
            image, gaze = image.to(device), gaze.to(device)
            optimiser.zero_grad()
            prediction = torch.nn.functional.normalize(
                network(image), p=2, dim=1, eps=1e-8)
            loss = (1.0 - (prediction * gaze).sum(dim=1)).mean()
            loss.backward()
            optimiser.step()
            loss_sum += float(loss.detach().cpu()) * len(image)
            observed += len(image)
        valid = predict_network(network, images, valid_rows, device)
        score = float(np.mean(angular_error_degrees(
            targets[valid_rows], valid)))
        history.append({
            "epoch": epoch,
            "training_cosine_loss": loss_sum / observed,
            "inner_validation_mean_angular_error_deg": score,
        })
        if score < best_error:
            best_error = score
            best_weights = {key: value.detach().cpu().clone()
                            for key, value in network.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_weights is None:
        raise RuntimeError(f"No valid {name} training checkpoint selected")
    network.load_state_dict(best_weights)
    return network.cpu(), history, {
        "best_epoch": best_epoch,
        "inner_validation_mean_angular_error_deg": best_error,
        "parameters": sum(param.numel() for param in network.parameters()),
        "pretrained": bool(pretrained),
    }
