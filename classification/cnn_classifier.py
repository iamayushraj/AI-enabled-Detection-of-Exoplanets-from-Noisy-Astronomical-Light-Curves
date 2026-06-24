"""
1D Convolutional Neural Network Classifier for ExoplanetAI

Classifies phase-folded light curves directly using a 1D CNN.
Works on the raw time-series shape rather than extracted features.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = ["transit", "eclipsing_binary", "variable_star", "blend", "noise"]
PHASE_FOLD_BINS = 200   # Number of bins for phase-folded input


class TransitCNN(nn.Module):
    """
    1D CNN for light curve classification.

    Architecture:
        Input (1, 200) → Conv1D(32) → Conv1D(64) → Conv1D(128)
        → GlobalAvgPool → FC(64) → Dropout → FC(5)
    """

    def __init__(self, input_length: int = PHASE_FOLD_BINS, n_classes: int = 5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def prepare_phase_folded_input(time: np.ndarray, flux: np.ndarray,
                                period: float, t0: float = 0.0,
                                n_bins: int = PHASE_FOLD_BINS) -> np.ndarray:
    """
    Phase-fold and bin a light curve for CNN input.

    Returns
    -------
    binned : np.ndarray
        Shape (n_bins,) — binned phase-folded flux.
    """
    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]

    if len(t) < 10 or period <= 0:
        return np.ones(n_bins)

    # Phase fold
    phase = ((t - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    # Bin by phase
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    binned = np.ones(n_bins)

    for i in range(n_bins):
        mask_bin = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask_bin.sum() > 0:
            binned[i] = np.median(f[mask_bin])

    # Normalize
    med = np.median(binned)
    if med != 0:
        binned = binned / med

    return binned


def train_cnn(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              n_epochs: int = 50, batch_size: int = 32,
              lr: float = 0.001, device: str = None) -> dict:
    """
    Train the 1D CNN classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training data, shape (n_samples, n_bins).
    y_train : np.ndarray
        Training labels.
    X_val, y_val : optional
        Validation data.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    device : str
        'cuda' or 'cpu'. Auto-detected if None.

    Returns
    -------
    result : dict
        Keys: 'model', 'train_losses', 'val_accuracies'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransitCNN(input_length=X_train.shape[1]).to(device)

    # Reshape for Conv1d: (batch, 1, length)
    X_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    val_accuracies = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)

        # Validation
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val).unsqueeze(1).to(device)
                y_val_t = torch.LongTensor(y_val).to(device)
                val_out = model(X_val_t)
                val_pred = val_out.argmax(dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
                val_accuracies.append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} — Loss: {avg_loss:.4f} — Val Acc: {val_acc:.4f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} — Loss: {avg_loss:.4f}")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "device": device,
    }


def predict_cnn(model: TransitCNN, phase_folded: np.ndarray,
                device: str = "cpu") -> dict:
    """
    Predict using the CNN model.

    Parameters
    ----------
    model : TransitCNN
        Trained CNN model.
    phase_folded : np.ndarray
        Phase-folded binned flux (1D array).
    device : str
        Device to run on.

    Returns
    -------
    prediction : dict
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(phase_folded).unsqueeze(0).unsqueeze(0).to(device)
        output = model(x)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    class_id = int(np.argmax(probs))
    return {
        "class": CLASS_NAMES[class_id],
        "class_id": class_id,
        "confidence": round(float(probs[class_id]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
    }


def save_cnn(model: TransitCNN, filepath: str):
    """Save CNN model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"✅ CNN model saved to {filepath}")


def load_cnn(filepath: str, device: str = "cpu") -> TransitCNN:
    """Load CNN model."""
    model = TransitCNN()
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model
