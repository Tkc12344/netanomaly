"""
Model explainability.

Uses SHAP to attribute each prediction to feature contributions, and
reports global feature importance (packet statistics, flow duration,
port activity tend to dominate the anomaly decision).

Requires shap: `pip install shap`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _top_contributions(
    columns: list[str], values: np.ndarray, top_k: int
) -> list[dict]:
    items = [
        {"feature": name, "contribution": float(value)}
        for name, value in zip(columns, values)
    ]
    items.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return items[:top_k]


def _tree_inputs(model, X: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    """Unwrap sklearn Pipelines so TreeExplainer sees the tree, not the wrapper."""
    if not isinstance(model, Pipeline):
        return model, X
    estimator = model.steps[-1][1]
    if len(model.steps) == 1:
        return estimator, X
    transformed = model[:-1].transform(X)
    columns = list(X.columns)
    if getattr(transformed, "shape", (0, 0))[1] != len(columns):
        columns = [f"f{i}" for i in range(transformed.shape[1])]
    X_explain = pd.DataFrame(transformed, columns=columns, index=X.index)
    return estimator, X_explain


def explain_instance(
    model, X_row: pd.DataFrame, top_k: int = 15
) -> tuple[list[dict], str]:
    """
    Per-row feature contributions for the same vector /classify just scored.

    Linear pipelines use scaled coefficients (no extra dependency). Tree
    models use SHAP when it is installed.
    """
    if isinstance(model, Pipeline) and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            scaler = model.named_steps.get("scaler")
            transformed = scaler.transform(X_row) if scaler is not None else X_row.to_numpy()
            contrib = np.asarray(transformed)[0] * np.asarray(clf.coef_[0])
            return _top_contributions(list(X_row.columns), contrib, top_k), "coefficients"

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "shap is not installed; tree explanations require `pip install shap`"
        ) from exc

    estimator, X_explain = _tree_inputs(model, X_row)
    explainer = shap.TreeExplainer(estimator)
    raw = explainer.shap_values(X_explain)
    if isinstance(raw, list):
        values = np.asarray(raw[1] if len(raw) > 1 else raw[0])
    elif hasattr(raw, "values"):
        values = np.asarray(raw.values)
    else:
        values = np.asarray(raw)
    if values.ndim == 3:
        values = values[:, :, -1]
    row = values[0]
    return _top_contributions(list(X_explain.columns), row, top_k), "shap"


def explain_model(model, X_sample: pd.DataFrame, max_rows: int = 500):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("shap is not installed. Run `pip install shap`.") from exc

    X_sample = X_sample.sample(n=min(max_rows, len(X_sample)), random_state=42)
    estimator, X_explain = _tree_inputs(model, X_sample)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_explain)

    # For binary classifiers, shap_values is a list [class0, class1] in
    # older SHAP versions; normalize to the positive-class contributions.
    values = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_abs_shap = np.abs(values).mean(axis=0)
    importance = (
        pd.Series(mean_abs_shap, index=X_explain.columns)
        .sort_values(ascending=False)
    )
    logger.info("Top contributing features:\n%s", importance.head(10))
    return importance, shap_values, X_sample


if __name__ == "__main__":
    import argparse

    import joblib

    from src import config

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="preprocessed CSV with the label column")
    args = parser.parse_args()

    from src.data.preprocess import drop_non_features

    model = joblib.load(config.CLASSIFIER_PATH)
    df = pd.read_csv(args.input_csv)
    X = drop_non_features(df)

    importance, _, _ = explain_model(model, X)
    print(importance.head(15))
