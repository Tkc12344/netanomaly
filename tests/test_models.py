import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.data.generate_synthetic import make_synthetic_chunk
from src.data.preprocess import run_preprocessing
from src.models.classifier import (
    classification_matrices,
    select_best,
    split_data,
    train_and_compare,
)
from src.models.forecaster import train_forecaster


def _tiny_clean(n_rows: int = 400, anomaly_rate: float = 0.2) -> pd.DataFrame:
    raw = make_synthetic_chunk(
        n_rows, anomaly_rate, pd.Timestamp("2026-01-01"), file_index=0
    )
    raw[config.SOURCE_FILE_COLUMN] = "synthetic.csv"
    return run_preprocessing(raw)


def test_test_fold_keeps_natural_prevalence():
    clean = _tiny_clean()
    _, _, _, y_test, _ = split_data(clean)
    assert abs(float(clean[config.LABEL_COLUMN].mean()) - float(y_test.mean())) < 0.06


def test_classification_matrix_drops_non_features():
    clean = _tiny_clean()
    X, y, attack = classification_matrices(clean)
    assert config.TIME_COLUMN not in X.columns
    assert config.ATTACK_TYPE_COLUMN not in X.columns
    assert config.LABEL_COLUMN not in X.columns
    assert config.SOURCE_FILE_COLUMN not in X.columns
    assert attack is not None
    assert set(y.unique()) <= {0, 1}


def test_dummy_baseline_and_scaled_logistic_regression():
    clean = _tiny_clean()
    fitted, results, cols = train_and_compare(
        clean, model_names=["dummy_most_frequent", "logistic_regression"]
    )
    assert isinstance(fitted["logistic_regression"], Pipeline)
    dummy = next(r for r in results if r.model_name == "dummy_most_frequent")
    assert dummy.is_baseline
    assert dummy.prevalence == results[1].prevalence
    assert dummy.per_attack_recall
    best_model, best = select_best(fitted, results)
    assert best.model_name != "dummy_most_frequent"
    assert best_model is fitted[best.model_name]
    for name in config.NON_FEATURE_COLUMNS:
        assert name not in cols


def test_saved_forest_uses_single_thread_for_serving():
    from sklearn.ensemble import RandomForestClassifier

    from src.models.artifacts import pin_estimator_threads

    model = RandomForestClassifier(n_estimators=8, n_jobs=-1, random_state=0)
    pin_estimator_threads(model)
    assert model.n_jobs == 1


def test_forecaster_reports_persist_baseline():
    clean = _tiny_clean(n_rows=250)
    _, result, cols, (y_test, y_pred) = train_forecaster(clean, n_estimators=15)
    assert result.persist_mae >= 0
    assert result.persist_rmse >= 0
    assert isinstance(result.beats_persist, bool)
    assert config.TRAFFIC_VOLUME_COLUMN in cols
    assert config.TIME_COLUMN not in cols
    assert len(y_pred) == len(y_test)
