"""
End-to-end pipeline runner.

    python -m src.pipeline

Runs: load -> preprocess -> train classifier -> train forecaster ->
save models -> measure latency, and prints a summary that mirrors
Chapter 5 (Results) of the thesis.
"""
from __future__ import annotations

import argparse
import json
import logging

from src.data.load import load_raw_dataset
from src.data.preprocess import drop_non_features, run_preprocessing
from src.models.classifier import save_model, select_best, train_and_compare
from src.models.forecaster import save_forecaster, train_forecaster
from src.models.latency import measure_latency

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-forecaster", action="store_true")
    args = parser.parse_args()

    logger.info("Step 1/5: loading + integrating raw data")
    raw = load_raw_dataset()

    logger.info("Step 2/5: preprocessing (natural class mix, time-sorted)")
    clean = run_preprocessing(raw, balance=False)

    logger.info(
        "Step 3/5: training classifiers "
        "(dummy baseline / scaled Logistic Regression / Random Forest / Gradient Boosting)"
    )
    fitted, results, feature_cols = train_and_compare(clean)
    best_model, best_result = select_best(fitted, results)
    save_model(
        best_model,
        feature_cols,
        model_name=best_result.model_name,
        metrics=best_result.as_dict(),
    )

    summary = {
        "classification": [r.as_dict() for r in results],
        "best_classifier": best_result.model_name,
    }

    if not args.skip_forecaster:
        logger.info("Step 4/5: training forecaster (same time-sorted frame, not rebalanced)")
        forecaster, forecast_result, forecast_cols, _ = train_forecaster(clean)
        save_forecaster(forecaster, forecast_cols, metrics=forecast_result.as_dict())
        summary["forecasting"] = forecast_result.as_dict()

    logger.info("Step 5/5: measuring inference latency on the chosen classifier")
    X_for_latency = drop_non_features(clean)
    summary["latency"] = measure_latency(best_model, X_for_latency)

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
