"""
Inference latency measurement.

Times per-sample prediction for the chosen model so you can check
whether it is fast enough for your hardware and serving budget.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def measure_latency(model, X_sample: pd.DataFrame, n_runs: int = 200) -> dict:
    row = X_sample.sample(n=1, random_state=0)
    # warm up (first call often pays a one-off JIT/allocation cost)
    model.predict(row)

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(row)
        timings.append((time.perf_counter() - start) * 1000)

    timings = np.array(timings)
    return {
        "mean_ms": float(timings.mean()),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "n_runs": n_runs,
    }


if __name__ == "__main__":
    import argparse
    import json

    import joblib

    from src import config

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()

    from src.data.preprocess import drop_non_features

    model = joblib.load(config.CLASSIFIER_PATH)
    df = pd.read_csv(args.input_csv)
    X = drop_non_features(df)

    print(json.dumps(measure_latency(model, X, args.n_runs), indent=2))
