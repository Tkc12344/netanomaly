"""
Feature engineering.

- Current-state features: already present as raw flow columns (packet
  counts, flow duration, byte rates).
- Statistical features: rolling mean/std over the traffic-volume
  series, computed within a source file so days are not mixed.
- Temporal features: a time-shifted target variable for forecasting,
  shifted within the same group.
- Contextual features: pass through as-is (ports/protocols already in
  the numeric feature set after preprocessing).
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data.preprocess import drop_non_features


def infer_group_column(df: pd.DataFrame) -> str | None:
    if config.SOURCE_FILE_COLUMN in df.columns:
        return config.SOURCE_FILE_COLUMN
    return None


def add_statistical_features(
    df: pd.DataFrame,
    window: int = 20,
    group_col: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    col = config.TRAFFIC_VOLUME_COLUMN
    if col not in df.columns:
        return df

    mean_name = f"{col}_roll_mean"
    std_name = f"{col}_roll_std"
    if group_col is None or group_col not in df.columns:
        group_col = infer_group_column(df)

    if group_col:
        grouped = df.groupby(group_col, sort=False)[col]
        df[mean_name] = grouped.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[std_name] = grouped.transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0)
        )
    else:
        series = df[col]
        df[mean_name] = series.rolling(window, min_periods=1).mean()
        df[std_name] = series.rolling(window, min_periods=1).std().fillna(0)
    return df


def shift_target(
    df: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON,
    group_col: str | None = None,
) -> pd.Series:
    col = config.TRAFFIC_VOLUME_COLUMN
    group_col = group_col or infer_group_column(df)
    if group_col and group_col in df.columns:
        return df.groupby(group_col, sort=False)[col].shift(-horizon)
    return df[col].shift(-horizon)


def build_forecasting_frame(
    df: pd.DataFrame, horizon: int = config.FORECAST_HORIZON
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds (X, y) for the forecasting task using a time-shifted target:
    y[t] = traffic_volume[t + horizon] *within the same source file*.
    Features at time t are current-state (including current volume);
    future rows are not used as inputs.
    """
    df = df.copy()
    group_col = infer_group_column(df)
    df = add_statistical_features(df, group_col=group_col)
    y = shift_target(df, horizon=horizon, group_col=group_col)

    feature_df = drop_non_features(df)
    valid = y.notna()
    X = feature_df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    return X, y
