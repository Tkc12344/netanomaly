"""
Traffic forecasting.

Uses a Random Forest Regressor on a time-shifted target so the model
only ever learns from past observations. Split is chronological (not
random/stratified) to avoid leaking future data into training —
classification uses a stratified split; forecasting uses a time-based
split.

A persist (last-value) baseline is scored on the same holdout: predict
the current traffic volume for the next step.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import config
from src.features.engineer import build_forecasting_frame

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    mae: float
    rmse: float
    persist_mae: float
    persist_rmse: float
    beats_persist: bool

    def as_dict(self) -> dict:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "persist_mae": self.persist_mae,
            "persist_rmse": self.persist_rmse,
            "beats_persist": self.beats_persist,
        }


def time_based_split(X: pd.DataFrame, y: pd.Series, test_size: float = config.TEST_SIZE):
    split_idx = int(len(X) * (1 - test_size))
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


def persist_forecast(X_test: pd.DataFrame) -> np.ndarray:
    """One-step persist baseline: current volume predicts the next volume."""
    if config.TRAFFIC_VOLUME_COLUMN not in X_test.columns:
        raise KeyError(
            f"Persist baseline needs {config.TRAFFIC_VOLUME_COLUMN} in X_test"
        )
    return X_test[config.TRAFFIC_VOLUME_COLUMN].to_numpy()


def train_forecaster(df: pd.DataFrame, n_estimators: int | None = None):
    X, y = build_forecasting_frame(df)
    X_train, X_test, y_train, y_test = time_based_split(X, y)

    model = RandomForestRegressor(
        n_estimators=n_estimators or config.N_ESTIMATORS,
        max_depth=20,
        min_samples_leaf=5,
        random_state=config.RANDOM_STATE,
        n_jobs=config.TRAIN_N_JOBS,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    persist_pred = persist_forecast(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    persist_mae = mean_absolute_error(y_test, persist_pred)
    persist_rmse = float(np.sqrt(mean_squared_error(y_test, persist_pred)))
    result = ForecastResult(
        mae=mae,
        rmse=rmse,
        persist_mae=persist_mae,
        persist_rmse=persist_rmse,
        beats_persist=mae < persist_mae,
    )
    logger.info(
        "Forecaster -> MAE=%.2f  RMSE=%.2f  persist MAE=%.2f  beats_persist=%s",
        result.mae,
        result.rmse,
        result.persist_mae,
        result.beats_persist,
    )
    return model, result, X.columns.tolist(), (y_test, y_pred)


def save_forecaster(
    model,
    feature_columns: list[str],
    path=None,
    *,
    metrics: dict | None = None,
) -> None:
    from src.models.artifacts import default_forecaster_card, pin_estimator_threads

    path = path or config.FORECASTER_PATH
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pin_estimator_threads(model)
    joblib.dump(model, path)
    with open(config.FORECASTER_COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f)
    default_forecaster_card(type(model).__name__, feature_columns, metrics)
    logger.info("Saved forecaster -> %s", path)


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

    model, result, feature_cols, _ = train_forecaster(clean)
    print(json.dumps(result.as_dict(), indent=2))
    save_forecaster(model, feature_cols, metrics=result.as_dict())
