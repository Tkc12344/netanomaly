"""Sidecar model cards written next to each joblib artifact."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config

logger = logging.getLogger(__name__)


def feature_hash(columns: list[str]) -> str:
    return hashlib.sha256(",".join(columns).encode()).hexdigest()[:12]


def write_model_card(
    path: Path,
    *,
    model_name: str,
    task: str,
    feature_columns: list[str],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    card = {
        "version": now.strftime("%Y%m%dT%H%M%SZ"),
        "created_at": now.isoformat(),
        "task": task,
        "model_name": model_name,
        "n_features": len(feature_columns),
        "feature_hash": feature_hash(feature_columns),
        "feature_columns": feature_columns,
        "metrics": metrics or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2))
    logger.info("Wrote model card -> %s (version=%s)", path, card["version"])
    return card


def read_model_card(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def default_classifier_card(
    model_name: str, feature_columns: list[str], metrics: dict[str, Any] | None = None
) -> dict[str, Any]:
    return write_model_card(
        config.CLASSIFIER_CARD_PATH,
        model_name=model_name,
        task="classification",
        feature_columns=feature_columns,
        metrics=metrics,
    )


def default_forecaster_card(
    model_name: str, feature_columns: list[str], metrics: dict[str, Any] | None = None
) -> dict[str, Any]:
    return write_model_card(
        config.FORECASTER_CARD_PATH,
        model_name=model_name,
        task="forecasting",
        feature_columns=feature_columns,
        metrics=metrics,
    )
