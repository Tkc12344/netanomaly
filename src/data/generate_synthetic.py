"""
Generates synthetic network-flow data shaped like CICIDS2017, so the
whole pipeline can be exercised (and CI can run) before the real
dataset is downloaded and dropped into data/raw/.

Not a substitute for the real dataset — swap it out once you have
CICIDS2017 CSVs in place (see README "CICIDS2017").

Attack rows share a backbone distribution with benign traffic and only
add attack-shaped offsets, so a default tree cannot separate classes
by a global scale shift.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

ATTACK_LABELS = ["DoS Hulk", "PortScan", "Bot", "Web Attack", "Infiltration"]
COMMON_PORTS = np.array([80, 443, 53, 22, 123, 25, 3389, 8080])


def _dest_ports(n: int, label: str) -> np.ndarray:
    if label == "PortScan":
        return RNG.integers(1, 65535, n)
    if label == "Web Attack":
        return RNG.choice(np.array([80, 443]), size=n)
    if label == "DoS Hulk":
        return np.full(n, 80)
    common = RNG.choice(COMMON_PORTS, size=n)
    ephemeral = RNG.integers(1024, 65535, n)
    return np.where(RNG.random(n) < 0.7, common, ephemeral)


def _features_for_label(n: int, label: str) -> pd.DataFrame:
    duration = RNG.lognormal(mean=6.2, sigma=0.9, size=n)
    fwd = RNG.poisson(10, n) + 1
    bwd = RNG.poisson(7, n)
    fwd_len = RNG.lognormal(mean=7.2, sigma=0.8, size=n)
    bwd_len = RNG.lognormal(mean=6.8, sigma=0.8, size=n)
    fwd_mean = RNG.lognormal(mean=4.5, sigma=0.5, size=n)
    bwd_mean = RNG.lognormal(mean=4.2, sigma=0.5, size=n)
    flow_bps = RNG.lognormal(mean=8.5, sigma=1.0, size=n)
    flow_pps = RNG.lognormal(mean=3.5, sigma=0.8, size=n)
    pkt_std = RNG.lognormal(mean=3.8, sigma=0.6, size=n)

    if label == "DoS Hulk":
        duration = duration * RNG.uniform(1.15, 2.4, n)
        fwd = fwd + RNG.poisson(18, n)
        fwd_len = fwd_len * RNG.uniform(1.1, 1.8, n)
        flow_bps = flow_bps * RNG.uniform(1.2, 2.2, n)
        flow_pps = flow_pps * RNG.uniform(1.2, 2.0, n)
    elif label == "PortScan":
        duration = np.minimum(duration, RNG.lognormal(mean=3.8, sigma=0.6, size=n))
        fwd = RNG.poisson(1, n) + 1
        bwd = RNG.poisson(0, n)
        fwd_len = RNG.lognormal(mean=5.0, sigma=0.5, size=n)
    elif label == "Web Attack":
        fwd_len = fwd_len * RNG.uniform(0.5, 1.15, n)
        fwd_mean = fwd_mean * RNG.uniform(0.6, 1.1, n)
    elif label == "Bot":
        flow_bps = flow_bps * RNG.uniform(0.85, 1.5, n)
        fwd = fwd + RNG.poisson(3, n)

    return pd.DataFrame(
        {
            "Flow Duration": duration,
            "Total Fwd Packets": fwd,
            "Total Backward Packets": bwd,
            "Total Length of Fwd Packets": fwd_len,
            "Total Length of Bwd Packets": bwd_len,
            "Fwd Packet Length Mean": fwd_mean,
            "Bwd Packet Length Mean": bwd_mean,
            "Flow Bytes/s": flow_bps,
            "Flow Packets/s": flow_pps,
            "Packet Length Std": pkt_std,
            "Destination Port": _dest_ports(n, label),
            "Protocol": RNG.choice(np.array([6, 17]), size=n, p=[0.8, 0.2]),
        }
    )


def make_synthetic_chunk(
    n_rows: int,
    anomaly_rate: float,
    start_time: pd.Timestamp,
    file_index: int = 0,
) -> pd.DataFrame:
    """
    One time-ordered file. Classes are mixed along the timeline; rows are
    never shuffled after timestamps are assigned.
    """
    n_anom = int(n_rows * anomaly_rate)
    n_benign = n_rows - n_anom
    labels = np.array(
        ["BENIGN"] * n_benign + list(RNG.choice(ATTACK_LABELS, size=n_anom))
    )
    RNG.shuffle(labels)
    timestamps = pd.date_range(start_time, periods=n_rows, freq="s")

    parts = []
    for lab in np.unique(labels):
        mask = labels == lab
        part = _features_for_label(int(mask.sum()), str(lab))
        part["Timestamp"] = timestamps[mask]
        part["Label"] = lab
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df.insert(0, "Flow ID", [f"flow-{file_index}-{i}" for i in range(len(df))])
    df.insert(1, "Source IP", RNG.integers(1, 255, len(df)).astype(str))
    df.insert(2, "Destination IP", RNG.integers(1, 255, len(df)).astype(str))
    df.insert(3, "Source Port", RNG.integers(1024, 65535, len(df)))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--anomaly-rate", type=float, default=0.08)
    parser.add_argument("--files", type=int, default=3, help="split across N csvs")
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "raw"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_per_file = args.rows // args.files
    for i in range(args.files):
        start = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
        df = make_synthetic_chunk(
            rows_per_file, args.anomaly_rate, start, file_index=i
        )
        out_path = args.out_dir / f"synthetic_traffic_part{i + 1}.csv"
        df.to_csv(out_path, index=False)
        print(f"wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
