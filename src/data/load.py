"""
Data Loading and Integration (thesis Sec 3.2 / 4.3).

Detects every CSV under data/raw/, reads each into a DataFrame, and
concatenates them into a single unified table — code-driven so the
merge is reproducible and requires no manual file wrangling.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_raw_dataset(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or config.DATA_RAW_DIR
    csv_paths = sorted(glob.glob(str(raw_dir / "*.csv")))

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Either download CICIDS2017 "
            f"there (see README) or run: python -m src.data.generate_synthetic"
        )

    logger.info("Found %d CSV file(s) in %s", len(csv_paths), raw_dir)
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False)
        df[config.SOURCE_FILE_COLUMN] = Path(path).name
        frames.append(df)
        logger.info("  loaded %s -> %s rows, %s cols", path, df.shape[0], df.shape[1])

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %s rows, %s cols", combined.shape[0], combined.shape[1])
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_raw_dataset()
    print(df.shape)
    print(df["Label"].value_counts() if "Label" in df.columns else df.columns.tolist())
