from __future__ import annotations

from pydantic import BaseModel, Field


class FlowFeatures(BaseModel):
    """
    A single network flow's numeric feature vector. Keys must include
    every column the loaded model was trained on; extra keys are ignored.
    """

    features: dict[str, float] = Field(default_factory=dict)


class ClassificationResponse(BaseModel):
    is_anomalous: bool
    anomaly_probability: float | None = None
    model: str
    model_version: str | None = None
    latency_ms: float


class ForecastResponse(BaseModel):
    predicted_traffic_volume: float
    horizon_steps: int
    model: str
    model_version: str | None = None
    latency_ms: float


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class ExplainResponse(BaseModel):
    contributions: list[FeatureContribution]
    model: str
    model_version: str | None = None
    method: str


class HealthResponse(BaseModel):
    status: str
    ready: bool
    classifier_loaded: bool
    forecaster_loaded: bool


class ReadyResponse(BaseModel):
    status: str
    classifier_loaded: bool
    forecaster_loaded: bool


class SchemaResponse(BaseModel):
    classifier_ready: bool
    forecaster_ready: bool
    classifier_features: list[str] | None = None
    forecaster_features: list[str] | None = None
    model_version: str | None = None


class ServiceInfoResponse(BaseModel):
    service: str
    version: str
    docs: str
    health: str
    ready: str
    schema: str
