import pandas as pd

from src.data.generate_synthetic import make_synthetic_chunk


def test_synthetic_timestamps_are_monotonic_and_classes_overlap():
    df = make_synthetic_chunk(2000, 0.15, pd.Timestamp("2026-01-01"), file_index=0)
    ts = pd.to_datetime(df["Timestamp"])
    assert ts.is_monotonic_increasing
    assert set(df["Label"].unique()) >= {"BENIGN", "DoS Hulk"}

    benign = df.loc[df["Label"] == "BENIGN", "Flow Duration"]
    dos = df.loc[df["Label"] == "DoS Hulk", "Flow Duration"]
    assert dos.quantile(0.10) < benign.quantile(0.90)
    assert dos.mean() / benign.mean() < 3.5


def test_synthetic_files_do_not_share_a_timeline():
    a = make_synthetic_chunk(50, 0.1, pd.Timestamp("2026-01-01"), file_index=0)
    b = make_synthetic_chunk(50, 0.1, pd.Timestamp("2026-01-02"), file_index=1)
    assert pd.to_datetime(a["Timestamp"]).max() < pd.to_datetime(b["Timestamp"]).min()
