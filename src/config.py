"""
Central configuration for the network anomaly detection & traffic
forecasting system.

Mirrors the design decisions in Chapter 3 (Methodology) and Chapter 4
(Implementation) of the source thesis:
  - Tiwari, A. (2026). "Network Anomaly Detection and Traffic
    Forecasting Using Machine Learning and Sequence Models."
    CSUN MSc Thesis.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = Path(os.environ.get("DATA_RAW_DIR", ROOT_DIR / "data" / "raw"))
DATA_PROCESSED_DIR = Path(
    os.environ.get("DATA_PROCESSED_DIR", ROOT_DIR / "data" / "processed")
)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT_DIR / "models"))

CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
FORECASTER_PATH = MODELS_DIR / "forecaster.joblib"
TRANSFORMER_PATH = MODELS_DIR / "transformer.pt"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
FORECASTER_COLUMNS_PATH = MODELS_DIR / "forecaster_feature_columns.json"
CLASSIFIER_CARD_PATH = MODELS_DIR / "classifier_card.json"
FORECASTER_CARD_PATH = MODELS_DIR / "forecaster_card.json"

# --------------------------------------------------------------------------
# Preprocessing (Sec 3.3 / 4.4)
# --------------------------------------------------------------------------
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"
ATTACK_TYPE_COLUMN = "Attack_Type"
TIME_COLUMN = "Timestamp"
SOURCE_FILE_COLUMN = "Source_File"

# Identifiers that carry no generalizable signal (Sec 3.3.5 / 4.4).
# Timestamp is *not* leakage for forecasting — it is kept through sort /
# grouped rolling, then stripped before fit via NON_FEATURE_COLUMNS.
LEAKAGE_COLUMNS = [
    "Flow_ID",
    "Source_IP",
    "Src_IP",
    "Destination_IP",
    "Dst_IP",
    "Source_Port",
    "Src_Port",
]

# Present on the preprocessed frame, never passed into a model.
NON_FEATURE_COLUMNS = [
    LABEL_COLUMN,
    ATTACK_TYPE_COLUMN,
    TIME_COLUMN,
    SOURCE_FILE_COLUMN,
]

# Column used to build the time-shifted forecasting target (Sec 3.4.4).
TRAFFIC_VOLUME_COLUMN = "Total_Length_of_Fwd_Packets"
FORECAST_HORIZON = 1  # steps ahead (rows), matches the thesis's next-step framing

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --------------------------------------------------------------------------
# Transformer (Sec 3.5.3 / 4.9) — lightweight, sequence length small on purpose
# --------------------------------------------------------------------------
SEQ_LEN = 10
TRANSFORMER_EPOCHS = 5
TRANSFORMER_BATCH_SIZE = 64
