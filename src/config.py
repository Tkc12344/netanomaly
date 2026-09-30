"""
Central configuration for the network anomaly detection and traffic
forecasting project.
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
# Preprocessing
# --------------------------------------------------------------------------
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"
ATTACK_TYPE_COLUMN = "Attack_Type"
TIME_COLUMN = "Timestamp"
SOURCE_FILE_COLUMN = "Source_File"

# Identifiers that carry no generalizable signal.
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

# Column used to build the time-shifted forecasting target.
TRAFFIC_VOLUME_COLUMN = "Total_Length_of_Fwd_Packets"
FORECAST_HORIZON = 1  # steps ahead (rows)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "200"))
TRAIN_N_JOBS = int(os.environ.get("TRAIN_N_JOBS", "-1"))
SERVE_N_JOBS = 1

# --------------------------------------------------------------------------
# Hugging Face CICIDS-2017 (datasets-server /rows)
# --------------------------------------------------------------------------
HF_ROWS_URL = os.environ.get(
    "HF_ROWS_URL", "https://datasets-server.huggingface.co/rows"
)
HF_DATASET = os.environ.get("HF_DATASET", "San0160/CICIDS-2017")
HF_CONFIG = os.environ.get("HF_CONFIG", "default")
HF_SPLIT = os.environ.get("HF_SPLIT", "train")
HF_PAGE_LENGTH = 100  # datasets-server maximum
HF_DEFAULT_MAX_ROWS = int(os.environ.get("HF_MAX_ROWS", "20000"))
HF_OUTPUT_FILENAME = "cicids2017_huggingface.csv"
# Proxy origin: this dump has no capture Timestamp. row_idx seconds from here
# is concatenation order, not packet-capture time.
HF_TIMESTAMP_ORIGIN = "2017-07-03 00:00:00"
HF_REQUEST_RETRIES = 5
HF_REQUEST_TIMEOUT_S = 60.0

# --------------------------------------------------------------------------
# Transformer — lightweight; sequence length is small on purpose
# --------------------------------------------------------------------------
SEQ_LEN = 10
TRANSFORMER_EPOCHS = 5
TRANSFORMER_BATCH_SIZE = 64
