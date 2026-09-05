import pandas as pd
import pytest

from src import config


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "classifier.joblib")
    monkeypatch.setattr(config, "FORECASTER_PATH", tmp_path / "forecaster.joblib")
    monkeypatch.setattr(config, "FEATURE_COLUMNS_PATH", tmp_path / "feature_columns.json")
    monkeypatch.setattr(
        config, "FORECASTER_COLUMNS_PATH", tmp_path / "forecaster_feature_columns.json"
    )
    monkeypatch.setattr(config, "CLASSIFIER_CARD_PATH", tmp_path / "classifier_card.json")
    monkeypatch.setattr(config, "FORECASTER_CARD_PATH", tmp_path / "forecaster_card.json")
    return tmp_path


@pytest.fixture
def trained_artifacts(model_dir):
    from src.data.generate_synthetic import make_synthetic_chunk
    from src.data.preprocess import run_preprocessing
    from src.models.classifier import save_model, select_best, train_and_compare
    from src.models.forecaster import save_forecaster, train_forecaster

    raw = make_synthetic_chunk(320, 0.2, pd.Timestamp("2026-01-01"), file_index=0)
    raw[config.SOURCE_FILE_COLUMN] = "ci.csv"
    clean = run_preprocessing(raw)
    fitted, results, cols = train_and_compare(
        clean, model_names=["dummy_most_frequent", "logistic_regression"]
    )
    best, best_result = select_best(fitted, results)
    save_model(
        best,
        cols,
        model_name=best_result.model_name,
        metrics=best_result.as_dict(),
    )
    model, forecast_result, forecast_cols, _ = train_forecaster(clean, n_estimators=12)
    save_forecaster(model, forecast_cols, metrics=forecast_result.as_dict())
    return {"classifier_cols": cols, "forecaster_cols": forecast_cols}


@pytest.fixture
def api_client(trained_artifacts):
    from fastapi.testclient import TestClient

    from src.api.main import app, reset_models

    reset_models()
    with TestClient(app) as client:
        yield client
    reset_models()


@pytest.fixture
def empty_api_client(model_dir):
    from fastapi.testclient import TestClient

    from src.api.main import app, reset_models

    reset_models()
    with TestClient(app) as client:
        yield client
    reset_models()
