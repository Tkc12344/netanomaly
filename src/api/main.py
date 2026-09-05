"""
Inference API for the network anomaly detection / traffic forecasting
system. Loads the models trained by src/pipeline.py and serves them
over HTTP.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    GET  /health     -> liveness (process up)
    GET  /ready      -> 503 unless the classifier is loaded
    POST /classify   -> complete feature vector -> anomaly yes/no
    POST /forecast   -> complete feature vector -> next traffic volume
    POST /explain    -> same vector as /classify -> feature contributions
"""
from __future__ import annotations

import json
import logging
import math
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from src import config
from src.api.schemas import (
    ClassificationResponse,
    ExplainResponse,
    FeatureContribution,
    FlowFeatures,
    ForecastResponse,
    HealthResponse,
    ReadyResponse,
)
from src.models.artifacts import read_model_card
from src.models.explain import explain_instance

logger = logging.getLogger(__name__)

MODELS: dict = {
    "classifier": None,
    "forecaster": None,
    "classifier_cols": None,
    "forecaster_cols": None,
    "classifier_card": None,
    "forecaster_card": None,
}


def reset_models() -> None:
    for key in MODELS:
        MODELS[key] = None


def _load_models() -> None:
    if config.CLASSIFIER_PATH.exists() and config.FEATURE_COLUMNS_PATH.exists():
        MODELS["classifier"] = joblib.load(config.CLASSIFIER_PATH)
        with open(config.FEATURE_COLUMNS_PATH) as f:
            MODELS["classifier_cols"] = json.load(f)
        MODELS["classifier_card"] = read_model_card(config.CLASSIFIER_CARD_PATH)
        logger.info("Loaded classifier from %s", config.CLASSIFIER_PATH)
    else:
        logger.warning("No classifier found at %s — /classify will 503", config.CLASSIFIER_PATH)

    if config.FORECASTER_PATH.exists() and config.FORECASTER_COLUMNS_PATH.exists():
        MODELS["forecaster"] = joblib.load(config.FORECASTER_PATH)
        with open(config.FORECASTER_COLUMNS_PATH) as f:
            MODELS["forecaster_cols"] = json.load(f)
        MODELS["forecaster_card"] = read_model_card(config.FORECASTER_CARD_PATH)
        logger.info("Loaded forecaster from %s", config.FORECASTER_PATH)
    else:
        logger.warning("No forecaster found at %s — /forecast will 503", config.FORECASTER_PATH)


def classifier_ready() -> bool:
    return MODELS["classifier"] is not None and MODELS["classifier_cols"] is not None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    _load_models()
    yield


app = FastAPI(
    title="Network Anomaly Detection & Traffic Forecasting API",
    version="0.2.0",
    lifespan=lifespan,
)


def _require_features(features: dict, columns: list[str]) -> None:
    missing = [col for col in columns if col not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Missing required features", "missing": missing},
        )
    non_finite = [
        col
        for col in columns
        if not math.isfinite(float(features[col]))
    ]
    if non_finite:
        raise HTTPException(
            status_code=422,
            detail={"message": "Non-finite feature values", "invalid": non_finite},
        )


def _vector_to_frame(features: dict, columns: list[str]) -> pd.DataFrame:
    row = {col: float(features[col]) for col in columns}
    return pd.DataFrame([row], columns=columns)


def _model_label(model, card: dict | None) -> str:
    if card and card.get("model_name"):
        return str(card["model_name"])
    if isinstance(model, Pipeline):
        return type(model.steps[-1][1]).__name__
    return type(model).__name__


def _model_version(card: dict | None) -> str | None:
    if card and card.get("version"):
        return str(card["version"])
    return None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = classifier_ready()
    return HealthResponse(
        status="ok",
        ready=loaded,
        classifier_loaded=loaded,
        forecaster_loaded=MODELS["forecaster"] is not None,
    )


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    if not classifier_ready():
        raise HTTPException(503, "Classifier not loaded — run the training pipeline first.")
    return ReadyResponse(
        status="ready",
        classifier_loaded=True,
        forecaster_loaded=MODELS["forecaster"] is not None,
    )


@app.post("/classify", response_model=ClassificationResponse)
def classify(payload: FlowFeatures) -> ClassificationResponse:
    model = MODELS["classifier"]
    columns = MODELS["classifier_cols"]
    if model is None or columns is None:
        raise HTTPException(503, "Classifier not loaded — run the training pipeline first.")

    _require_features(payload.features, columns)
    X = _vector_to_frame(payload.features, columns)
    start = time.perf_counter()
    pred = model.predict(X)[0]
    proba = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None
    latency_ms = (time.perf_counter() - start) * 1000

    return ClassificationResponse(
        is_anomalous=bool(pred),
        anomaly_probability=proba,
        model=_model_label(model, MODELS["classifier_card"]),
        model_version=_model_version(MODELS["classifier_card"]),
        latency_ms=latency_ms,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(payload: FlowFeatures) -> ForecastResponse:
    model = MODELS["forecaster"]
    columns = MODELS["forecaster_cols"]
    if model is None or columns is None:
        raise HTTPException(503, "Forecaster not loaded — run the training pipeline first.")

    _require_features(payload.features, columns)
    X = _vector_to_frame(payload.features, columns)
    start = time.perf_counter()
    pred = float(model.predict(X)[0])
    latency_ms = (time.perf_counter() - start) * 1000

    return ForecastResponse(
        predicted_traffic_volume=pred,
        horizon_steps=config.FORECAST_HORIZON,
        model=_model_label(model, MODELS["forecaster_card"]),
        model_version=_model_version(MODELS["forecaster_card"]),
        latency_ms=latency_ms,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: FlowFeatures) -> ExplainResponse:
    model = MODELS["classifier"]
    columns = MODELS["classifier_cols"]
    if model is None or columns is None:
        raise HTTPException(503, "Classifier not loaded — run the training pipeline first.")

    _require_features(payload.features, columns)
    X = _vector_to_frame(payload.features, columns)
    try:
        contributions, method = explain_instance(model, X)
    except RuntimeError as exc:
        raise HTTPException(501, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(501, f"Cannot explain this model: {exc}") from exc

    return ExplainResponse(
        contributions=[FeatureContribution(**row) for row in contributions],
        model=_model_label(model, MODELS["classifier_card"]),
        model_version=_model_version(MODELS["classifier_card"]),
        method=method,
    )
