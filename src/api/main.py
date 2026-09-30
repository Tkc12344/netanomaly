"""
Inference API for the network anomaly detection / traffic forecasting
system. Loads the models trained by src/pipeline.py and serves them
over HTTP.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    GET  /           -> service info
    GET  /ui         -> operator console
    GET  /health     -> liveness (process up)
    GET  /ready      -> 503 unless the classifier is loaded
    GET  /schema     -> required feature names for the loaded models
    POST /classify   -> complete feature vector -> anomaly yes/no
    POST /forecast   -> complete feature vector -> next traffic volume
    POST /explain    -> same vector as /classify -> feature contributions
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
    SchemaResponse,
    ServiceInfoResponse,
)
from src.models.artifacts import pin_estimator_threads, read_model_card
from src.models.explain import explain_instance

logger = logging.getLogger(__name__)
API_VERSION = "0.3.0"
STATIC_DIR = Path(__file__).resolve().parent / "static"

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


def _load_json_list(path) -> list | None:
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"{path} must be a JSON list of column names")
    return payload


def _load_models() -> None:
    if config.CLASSIFIER_PATH.exists() and config.FEATURE_COLUMNS_PATH.exists():
        try:
            model = joblib.load(config.CLASSIFIER_PATH)
            pin_estimator_threads(model)
            MODELS["classifier"] = model
            MODELS["classifier_cols"] = _load_json_list(config.FEATURE_COLUMNS_PATH)
            MODELS["classifier_card"] = read_model_card(config.CLASSIFIER_CARD_PATH)
            logger.info("Loaded classifier from %s", config.CLASSIFIER_PATH)
        except Exception:
            logger.exception("Failed to load classifier — /classify stays 503")
            MODELS["classifier"] = None
            MODELS["classifier_cols"] = None
    else:
        logger.warning("No classifier found at %s — /classify will 503", config.CLASSIFIER_PATH)

    if config.FORECASTER_PATH.exists() and config.FORECASTER_COLUMNS_PATH.exists():
        try:
            model = joblib.load(config.FORECASTER_PATH)
            pin_estimator_threads(model)
            MODELS["forecaster"] = model
            MODELS["forecaster_cols"] = _load_json_list(config.FORECASTER_COLUMNS_PATH)
            MODELS["forecaster_card"] = read_model_card(config.FORECASTER_CARD_PATH)
            logger.info("Loaded forecaster from %s", config.FORECASTER_PATH)
        except Exception:
            logger.exception("Failed to load forecaster — /forecast stays 503")
            MODELS["forecaster"] = None
            MODELS["forecaster_cols"] = None
    else:
        logger.warning("No forecaster found at %s — /forecast will 503", config.FORECASTER_PATH)


def classifier_ready() -> bool:
    return MODELS["classifier"] is not None and MODELS["classifier_cols"] is not None


def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = os.environ.get("API_KEY", "").strip()
    if not expected:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _load_models()
    yield


app = FastAPI(
    title="Netanomaly API",
    version=API_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    if request.url.path not in ("/health", "/ready") and not request.url.path.startswith(
        "/ui-static"
    ):
        logger.info(
            "%s %s %s %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
    return response


def _require_features(features: dict, columns: list[str]) -> None:
    missing = [col for col in columns if col not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Missing required features", "missing": missing},
        )
    invalid: list[str] = []
    non_finite: list[str] = []
    for col in columns:
        try:
            value = float(features[col])
        except (TypeError, ValueError):
            invalid.append(col)
            continue
        if not math.isfinite(value):
            non_finite.append(col)
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={"message": "Non-numeric feature values", "invalid": invalid},
        )
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


@app.get("/", response_model=ServiceInfoResponse)
def root() -> ServiceInfoResponse:
    return ServiceInfoResponse(
        service="netanomaly",
        version=API_VERSION,
        docs="/docs",
        health="/health",
        ready="/ready",
        schema="/schema",
        ui="/ui",
    )


@app.get("/ui", include_in_schema=False)
def console_ui() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "UI files are missing from this image.")
    return FileResponse(index)


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


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    if not classifier_ready():
        raise HTTPException(503, "Classifier not loaded — run the training pipeline first.")
    return SchemaResponse(
        classifier_ready=True,
        forecaster_ready=MODELS["forecaster"] is not None,
        classifier_features=list(MODELS["classifier_cols"]),
        forecaster_features=(
            list(MODELS["forecaster_cols"]) if MODELS["forecaster_cols"] else None
        ),
        model_version=_model_version(MODELS["classifier_card"]),
    )


@app.post("/classify", response_model=ClassificationResponse)
def classify(
    payload: FlowFeatures, _: None = Depends(require_api_key)
) -> ClassificationResponse:
    model = MODELS["classifier"]
    columns = MODELS["classifier_cols"]
    if model is None or columns is None:
        raise HTTPException(503, "Classifier not loaded — run the training pipeline first.")

    _require_features(payload.features, columns)
    X = _vector_to_frame(payload.features, columns)
    start = time.perf_counter()
    try:
        pred = model.predict(X)[0]
        proba = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None
    except Exception:
        logger.exception("Classifier inference failed")
        raise HTTPException(500, "Inference failed") from None
    latency_ms = (time.perf_counter() - start) * 1000

    return ClassificationResponse(
        is_anomalous=bool(pred),
        anomaly_probability=proba,
        model=_model_label(model, MODELS["classifier_card"]),
        model_version=_model_version(MODELS["classifier_card"]),
        latency_ms=latency_ms,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(
    payload: FlowFeatures, _: None = Depends(require_api_key)
) -> ForecastResponse:
    model = MODELS["forecaster"]
    columns = MODELS["forecaster_cols"]
    if model is None or columns is None:
        raise HTTPException(503, "Forecaster not loaded — run the training pipeline first.")

    _require_features(payload.features, columns)
    X = _vector_to_frame(payload.features, columns)
    start = time.perf_counter()
    try:
        pred = float(model.predict(X)[0])
    except Exception:
        logger.exception("Forecaster inference failed")
        raise HTTPException(500, "Inference failed") from None
    latency_ms = (time.perf_counter() - start) * 1000

    return ForecastResponse(
        predicted_traffic_volume=pred,
        horizon_steps=config.FORECAST_HORIZON,
        model=_model_label(model, MODELS["forecaster_card"]),
        model_version=_model_version(MODELS["forecaster_card"]),
        latency_ms=latency_ms,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(
    payload: FlowFeatures, _: None = Depends(require_api_key)
) -> ExplainResponse:
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


app.mount("/ui-static", StaticFiles(directory=STATIC_DIR), name="ui-static")
