"""
Lightweight Transformer Model (thesis Sec 3.5.3 / 4.9).

The thesis's own finding: on structured tabular flow data, a small
self-attention sequence model underperforms the classical Random
Forest and needs far more tuning/data to close the gap (Sec 4.9,
6.6). This module exists to demonstrate and reproduce that result,
not because it's the recommended production model — the API and
deployment layers use the Random Forest classifier/forecaster
(Sec 3.7.2).

Requires torch: `pip install torch` (kept optional/importable-only so
the rest of the system works without it).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = config.SEQ_LEN):
    """Sliding-window sequence construction (Sec 4.9)."""
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)


def train_transformer(df: pd.DataFrame, epochs: int = config.TRANSFORMER_EPOCHS):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "torch is not installed. Run `pip install torch` to use the "
            "Transformer model — it's the optional, lower-priority model "
            "in this system (see module docstring)."
        ) from exc

    from src.features.engineer import build_forecasting_frame

    X_df, y_series = build_forecasting_frame(df)
    X = X_df.select_dtypes(include=[np.number]).fillna(0).values.astype("float32")
    y = y_series.values.astype("float32")

    X_seq, y_seq = make_sequences(X, y, config.SEQ_LEN)
    split = int(len(X_seq) * (1 - config.TEST_SIZE))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    n_features = X.shape[1]

    class TinyTransformer(nn.Module):
        def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.encoder(x)
            x = x.mean(dim=1)  # pool over the sequence dimension
            return self.head(x).squeeze(-1)

    model = TinyTransformer(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=config.TRANSFORMER_BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        logger.info("epoch %d/%d  loss=%.4f", epoch + 1, epochs, total_loss / len(train_ds))

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy()
    mae = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    logger.info("Transformer -> MAE=%.2f  RMSE=%.2f (expect worse than Random Forest, per thesis)", mae, rmse)
    return model, {"mae": mae, "rmse": rmse}


if __name__ == "__main__":
    import argparse

    from src.data.load import load_raw_dataset
    from src.data.preprocess import run_preprocessing

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=None)
    args = parser.parse_args()

    clean = (
        pd.read_csv(args.input_csv)
        if args.input_csv
        else run_preprocessing(load_raw_dataset(), balance=False)
    )
    _, metrics = train_transformer(clean)
    print(metrics)
