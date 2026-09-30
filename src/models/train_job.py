"""
Cluster training entrypoint.

If data/raw is empty, either fetches Hugging Face CICIDS-2017
(`DATA_SOURCE=huggingface`) or generates CICIDS-shaped synthetic data.
Then runs the pipeline onto MODELS_DIR (the models PVC) and optionally
pushes artifacts to MODEL_STORAGE_BASE_URL.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

from src import config
from src.models.sync import main as sync_main

logger = logging.getLogger(__name__)


def _ensure_raw_data() -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if any(config.DATA_RAW_DIR.glob("*.csv")):
        logger.info("Using existing CSVs in %s", config.DATA_RAW_DIR)
        return
    source = os.environ.get("DATA_SOURCE", "synthetic").strip().lower()
    if source == "huggingface":
        max_rows = os.environ.get("HF_MAX_ROWS", str(config.HF_DEFAULT_MAX_ROWS))
        logger.info(
            "No raw CSVs found; fetching Hugging Face %s (max_rows=%s)",
            config.HF_DATASET,
            max_rows,
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "src.data.fetch_huggingface",
                "--max-rows",
                max_rows,
                "--out-dir",
                str(config.DATA_RAW_DIR),
            ]
        )
        return
    rows = os.environ.get("TRAIN_ROWS", "8000")
    files = os.environ.get("TRAIN_FILES", "2")
    logger.info("No raw CSVs found; generating %s synthetic rows", rows)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "src.data.generate_synthetic",
            "--rows",
            rows,
            "--files",
            files,
            "--out-dir",
            str(config.DATA_RAW_DIR),
        ]
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_raw_data()
    subprocess.check_call([sys.executable, "-m", "src.pipeline"])
    return sync_main(["--push"])


if __name__ == "__main__":
    sys.exit(main())
