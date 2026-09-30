"""
Classification models.

Trains a most-frequent dummy baseline, Logistic Regression (scaled),
Random Forest, and Gradient Boosting on the anomaly-detection task.

The frame is split first (stratified) so the holdout keeps the natural
class mix. Real models are fit with balanced sample weights on the
training fold only. Per-attack recall is reported even though the
served decision is binary.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.data.preprocess import drop_non_features

logger = logging.getLogger(__name__)

MODEL_FACTORIES = {
    "dummy_most_frequent": lambda: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            ),
        ]
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        n_jobs=config.TRAIN_N_JOBS,
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        random_state=config.RANDOM_STATE
    ),
}


@dataclass
class ClassificationResult:
    model_name: str
    precision: float
    recall: float
    f1: float
    auc_pr: float
    confusion: list = field(default_factory=list)
    prevalence: float = 0.0
    per_attack_recall: dict = field(default_factory=dict)
    is_baseline: bool = False

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_pr": self.auc_pr,
            "confusion_matrix": self.confusion,
            "prevalence": self.prevalence,
            "per_attack_recall": self.per_attack_recall,
            "is_baseline": self.is_baseline,
        }


def classification_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    y = df[config.LABEL_COLUMN]
    attack = (
        df[config.ATTACK_TYPE_COLUMN]
        if config.ATTACK_TYPE_COLUMN in df.columns
        else None
    )
    X = drop_non_features(df)
    return X, y, attack


def split_data(df: pd.DataFrame):
    """
    Stratified split on the binary label. Returns
    X_train, X_test, y_train, y_test, attack_type_test.
    """
    X, y, attack = classification_matrices(df)
    stratify = y if y.nunique() > 1 else None
    if attack is not None:
        X_tr, X_te, y_tr, y_te, _, a_te = train_test_split(
            X,
            y,
            attack,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=stratify,
        )
        return X_tr, X_te, y_tr, y_te, a_te
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )
    return X_tr, X_te, y_tr, y_te, None


def per_attack_recall(
    y_pred, attack_types: pd.Series | None
) -> dict[str, float]:
    if attack_types is None:
        return {}
    y_pred = np.asarray(y_pred)
    attacks = attack_types.to_numpy()
    out: dict[str, float] = {}
    for name in pd.unique(attacks):
        if name == config.BENIGN_LABEL:
            continue
        mask = attacks == name
        if mask.sum() == 0:
            continue
        out[str(name)] = float((y_pred[mask] == 1).mean())
    return out


def evaluate(
    model,
    X_test,
    y_test,
    model_name: str,
    attack_types: pd.Series | None = None,
) -> ClassificationResult:
    y_pred = model.predict(X_test)
    y_score = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )
    y_true = np.asarray(y_test)
    return ClassificationResult(
        model_name=model_name,
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        auc_pr=average_precision_score(y_test, y_score),
        confusion=confusion_matrix(y_test, y_pred).tolist(),
        prevalence=float(np.mean(y_true == 1)),
        per_attack_recall=per_attack_recall(y_pred, attack_types),
        is_baseline=model_name.startswith("dummy_"),
    )


def _fit(model, X, y, sample_weight=None):
    if sample_weight is None:
        model.fit(X, y)
        return model
    if isinstance(model, Pipeline):
        last = model.steps[-1][0]
        model.fit(X, y, **{f"{last}__sample_weight": sample_weight})
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def train_and_compare(
    df: pd.DataFrame, model_names: list[str] | None = None
) -> tuple[dict, list[ClassificationResult], list[str]]:
    """Trains the requested classifiers and returns fitted models + results + feature columns."""
    X_train, X_test, y_train, y_test, attack_test = split_data(df)
    names = model_names or list(MODEL_FACTORIES)
    sample_weight = compute_sample_weight("balanced", y_train)

    fitted = {}
    results = []
    for name in names:
        logger.info("Training %s ...", name)
        model = MODEL_FACTORIES[name]()
        if name.startswith("dummy_"):
            model.fit(X_train, y_train)
        else:
            _fit(model, X_train, y_train, sample_weight)
        fitted[name] = model
        result = evaluate(
            model, X_test, y_test, name, attack_types=attack_test
        )
        results.append(result)
        logger.info(
            "%s -> F1=%.4f  AUC-PR=%.4f  precision=%.4f  recall=%.4f  prevalence=%.4f",
            name,
            result.f1,
            result.auc_pr,
            result.precision,
            result.recall,
            result.prevalence,
        )
        if result.per_attack_recall:
            logger.info("%s per-attack recall: %s", name, result.per_attack_recall)

    return fitted, results, X_train.columns.tolist()


def select_best(fitted: dict, results: list[ClassificationResult]):
    """Pick the highest-F1 real model. Dummy baselines are never selected."""
    candidates = [r for r in results if not r.is_baseline]
    if not candidates:
        candidates = results
    best = max(candidates, key=lambda r: r.f1)
    return fitted[best.model_name], best


def save_model(
    model,
    feature_columns: list[str],
    path=None,
    *,
    model_name: str | None = None,
    metrics: dict | None = None,
) -> None:
    from src.models.artifacts import default_classifier_card, pin_estimator_threads

    path = path or config.CLASSIFIER_PATH
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pin_estimator_threads(model)
    joblib.dump(model, path)
    with open(config.FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f)
    default_classifier_card(
        model_name or type(model).__name__,
        feature_columns,
        metrics,
    )
    logger.info("Saved classifier -> %s", path)


if __name__ == "__main__":
    import argparse

    from src.data.load import load_raw_dataset
    from src.data.preprocess import run_preprocessing

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default=None,
        help="pre-processed CSV; else re-runs preprocessing",
    )
    args = parser.parse_args()

    if args.input_csv:
        clean = pd.read_csv(args.input_csv)
    else:
        clean = run_preprocessing(load_raw_dataset())

    fitted, results, feature_cols = train_and_compare(clean)
    best_model, best_result = select_best(fitted, results)
    print(json.dumps([r.as_dict() for r in results], indent=2))
    print(f"\nBest model: {best_result.model_name} (F1={best_result.f1:.4f})")
    save_model(
        best_model,
        feature_cols,
        model_name=best_result.model_name,
        metrics=best_result.as_dict(),
    )
