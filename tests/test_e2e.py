"""End-to-end: generate CSVs → pipeline CLI → API classify/forecast/explain."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src import config

ROOT = Path(__file__).resolve().parents[1]


def _complete(columns, fill=1.0):
    return {col: float(fill) for col in columns}


def test_e2e_generate_pipeline_api(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    models = tmp_path / "models"
    raw.mkdir()
    models.mkdir()

    env = {**os.environ, "DATA_RAW_DIR": str(raw), "MODELS_DIR": str(models)}
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "src.data.generate_synthetic",
            "--rows",
            "600",
            "--files",
            "1",
            "--out-dir",
            str(raw),
        ],
        cwd=ROOT,
        env=env,
    )
    subprocess.check_call(
        [sys.executable, "-m", "src.pipeline"],
        cwd=ROOT,
        env=env,
    )

    clf = models / "classifier.joblib"
    cols_path = models / "feature_columns.json"
    fc_cols_path = models / "forecaster_feature_columns.json"

    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "CLASSIFIER_PATH", clf)
    monkeypatch.setattr(config, "FEATURE_COLUMNS_PATH", cols_path)
    monkeypatch.setattr(config, "FORECASTER_PATH", models / "forecaster.joblib")
    monkeypatch.setattr(config, "FORECASTER_COLUMNS_PATH", fc_cols_path)
    monkeypatch.setattr(config, "CLASSIFIER_CARD_PATH", models / "classifier_card.json")
    monkeypatch.setattr(config, "FORECASTER_CARD_PATH", models / "forecaster_card.json")

    from fastapi.testclient import TestClient

    from src.api.main import app, reset_models

    reset_models()
    classifier_cols = json.loads(cols_path.read_text())
    forecast_cols = json.loads(fc_cols_path.read_text())
    with TestClient(app) as client:
        ready = client.get("/ready")
        classify = client.post("/classify", json={"features": _complete(classifier_cols)})
        forecast = client.post("/forecast", json={"features": _complete(forecast_cols)})
        explain = client.post("/explain", json={"features": _complete(classifier_cols)})
        assert ready.status_code == 200
        assert classify.status_code == 200
        assert "is_anomalous" in classify.json()
        assert forecast.status_code == 200
        assert "predicted_traffic_volume" in forecast.json()
        assert explain.status_code == 200
    reset_models()
