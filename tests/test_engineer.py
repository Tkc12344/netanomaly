import pandas as pd

from src import config
from src.features.engineer import add_statistical_features, build_forecasting_frame


def test_rolling_does_not_cross_source_files():
    df = pd.DataFrame(
        {
            config.TRAFFIC_VOLUME_COLUMN: [10.0, 20.0, 30.0, 1000.0, 2000.0],
            config.SOURCE_FILE_COLUMN: ["a.csv", "a.csv", "a.csv", "b.csv", "b.csv"],
            config.LABEL_COLUMN: [0, 0, 0, 0, 0],
        }
    )
    out = add_statistical_features(df, window=3)
    mean_col = f"{config.TRAFFIC_VOLUME_COLUMN}_roll_mean"
    assert out[mean_col].iloc[3] == 1000.0
    assert out[mean_col].iloc[4] == 1500.0
    assert out[mean_col].iloc[2] == 20.0


def test_forecast_target_does_not_shift_across_files():
    df = pd.DataFrame(
        {
            config.TRAFFIC_VOLUME_COLUMN: [10.0, 20.0, 30.0, 1000.0, 2000.0],
            config.SOURCE_FILE_COLUMN: ["a.csv", "a.csv", "a.csv", "b.csv", "b.csv"],
            config.LABEL_COLUMN: [0, 0, 0, 0, 0],
            config.ATTACK_TYPE_COLUMN: ["BENIGN"] * 5,
            config.TIME_COLUMN: pd.date_range("2026-01-01", periods=5, freq="s"),
        }
    )
    X, y = build_forecasting_frame(df, horizon=1)
    assert len(X) == 3
    assert list(y) == [20.0, 30.0, 2000.0]
    assert config.TIME_COLUMN not in X.columns
    assert config.LABEL_COLUMN not in X.columns
    assert config.SOURCE_FILE_COLUMN not in X.columns
