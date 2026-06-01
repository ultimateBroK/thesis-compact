from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sklearn.base import BaseEstimator, ClassifierMixin


def derive_rolling_sequences(X: pd.DataFrame, sequence_length: int) -> torch.Tensor:
    """Vectorized rolling sequence builder with pre-padding (repeat first row)."""
    values = X.to_numpy(dtype=np.float32)
    n, d = values.shape
    if n == 0:
        return torch.empty(0, sequence_length, d)
    # Pre-pad with sequence_length-1 copies of the first row
    padded = np.vstack([np.tile(values[0:1], (sequence_length - 1, 1)), values])
    # Strided windows of length sequence_length
    shape = (n, sequence_length, d)
    strides = (padded.strides[0], padded.strides[0], padded.strides[1])
    sequences = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return torch.from_numpy(sequences.copy())


class GRUNet(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = torch.nn.Dropout(dropout)
        dir_mult = 2 if bidirectional else 1
        self.output = torch.nn.Linear(hidden_size * dir_mult, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        # Take last time step; bidirectional output is already concatenated
        pooled = out[:, -1, :]
        return self.output(self.dropout(pooled))


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(pred, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class GRUClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        sequence_length: int = 8,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 64,
        bidirectional: bool = True,
        random_state: int = 42,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.random_state = random_state
        self.focal_gamma = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if len(self.classes_) == 1:
            self.model_ = None
            return self

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device
        use_amp = device.type == "cuda"

        train_x = derive_rolling_sequences(X, self.sequence_length).to(device)
        y_idx = np.searchsorted(self.classes_, y)
        train_y = torch.as_tensor(y_idx, dtype=torch.long, device=device)
        class_counts = np.bincount(y_idx, minlength=len(self.classes_))
        class_weights = len(y) / (len(self.classes_) * np.maximum(class_counts, 1))
        class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

        model = GRUNet(
            self.n_features_in_, self.hidden_size, len(self.classes_),
            num_layers=self.num_layers, dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = FocalLoss(gamma=self.focal_gamma, weight=class_weights_tensor)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        num_samples = len(train_x)
        indices = np.arange(num_samples)

        model.train()
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = loss_fn(model(train_x[batch_idx]), train_y[batch_idx])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        self.model_ = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            return np.full((len(X), max(len(self.classes_), 1)), 1.0 / max(len(self.classes_), 1))

        self.model_.eval()
        sequences = derive_rolling_sequences(X, self.sequence_length).to(self.device_)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                logits = self.model_(sequences[start : start + self.batch_size])
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(chunks)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
