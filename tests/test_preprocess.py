import numpy as np
import pandas as pd

from src import config
from src.data.preprocess import (
    balance_classes,
    drop_leakage_columns,
    drop_non_features,
    handle_missing_and_infinite,
    run_preprocessing,
    standardize_columns,
    transform_label,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow ID": ["a", "b", "c", "d"],
            "Source IP": ["1.1.1.1"] * 4,
            "Flow Duration": [1.0, np.inf, 3.0, np.nan],
            "Timestamp": [
                "2026-01-01 00:00:03",
                "2026-01-01 00:00:01",
                "2026-01-01 00:00:02",
                "2026-01-01 00:00:04",
            ],
            "Label": ["BENIGN", "DoS Hulk", "BENIGN", "PortScan"],
        }
    )


def test_standardize_columns_strips_and_underscores():
    df = _sample_df()
    out = standardize_columns(df)
    assert "Flow_ID" in out.columns
    assert "Flow_Duration" in out.columns


def test_handle_missing_and_infinite_fills_zero():
    df = standardize_columns(_sample_df())
    out = handle_missing_and_infinite(df)
    assert out["Flow_Duration"].isna().sum() == 0
    assert np.isinf(out["Flow_Duration"]).sum() == 0
    assert out["Flow_Duration"].iloc[1] == 0  # was inf
    assert out["Flow_Duration"].iloc[3] == 0  # was nan


def test_transform_label_binary_and_keeps_attack_type():
    df = standardize_columns(_sample_df())
    out = transform_label(df)
    assert set(out["Label"].unique()) <= {0, 1}
    assert out["Label"].iloc[0] == 0  # BENIGN
    assert out["Label"].iloc[1] == 1  # DoS Hulk
    assert out[config.ATTACK_TYPE_COLUMN].iloc[1] == "DoS Hulk"
    assert out[config.ATTACK_TYPE_COLUMN].iloc[0] == "BENIGN"


def test_drop_leakage_columns_removes_identifiers_keeps_time():
    df = standardize_columns(_sample_df())
    out = drop_leakage_columns(df)
    assert "Flow_ID" not in out.columns
    assert "Source_IP" not in out.columns
    assert "Timestamp" in out.columns


def test_balance_classes_equalizes_counts():
    df = pd.DataFrame({"Label": [0] * 90 + [1] * 10, "x": range(100)})
    balanced = balance_classes(df)
    counts = balanced["Label"].value_counts()
    assert counts[0] == counts[1] == 10


def test_run_preprocessing_sorts_time_and_keeps_natural_mix():
    raw = pd.DataFrame(
        {
            "Flow ID": list("abcdef"),
            "Source IP": ["1.1.1.1"] * 6,
            "Flow Duration": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Total Length of Fwd Packets": [10, 20, 30, 40, 50, 60],
            "Timestamp": [
                "2026-01-01 00:00:05",
                "2026-01-01 00:00:01",
                "2026-01-01 00:00:02",
                "2026-01-01 00:00:03",
                "2026-01-01 00:00:04",
                "2026-01-01 00:00:06",
            ],
            "Label": ["BENIGN"] * 5 + ["DoS Hulk"],
            "Source_File": ["day1.csv"] * 6,
        }
    )
    out = run_preprocessing(raw)
    assert list(out["Total_Length_of_Fwd_Packets"]) == [20, 30, 40, 50, 10, 60]
    assert out["Timestamp"].is_monotonic_increasing
    assert "Flow_ID" not in out.columns
    assert config.ATTACK_TYPE_COLUMN in out.columns
    assert out["Label"].mean() == 1 / 6
    features = drop_non_features(out)
    assert config.TIME_COLUMN not in features.columns
    assert config.ATTACK_TYPE_COLUMN not in features.columns
    assert config.SOURCE_FILE_COLUMN not in features.columns
    assert "Total_Length_of_Fwd_Packets" in features.columns


def test_object_infinity_is_coerced_not_dropped():
    raw = pd.DataFrame(
        {
            "Flow Duration": [1.0, 2.0],
            "Flow Bytes/s": ["Infinity", "10.5"],
            "Total Length of Fwd Packets": [1.0, 2.0],
            "Timestamp": ["2026-01-01 00:00:01", "2026-01-01 00:00:02"],
            "Label": ["BENIGN", "BENIGN"],
        }
    )
    out = run_preprocessing(raw)
    assert "Flow_Bytes/s" in out.columns
    assert out["Flow_Bytes/s"].iloc[0] == 0
    assert out["Flow_Bytes/s"].iloc[1] == 10.5
