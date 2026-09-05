"""
Data Preprocessing (thesis Sec 3.3 / 4.4).

Steps, in order:
  1. Column standardization      (3.3.2 / 4.4)
  2. Time parse + chronological sort (needed for Sec 3.4 / 3.6.1 forecasting)
  3. Missing / infinite handling (3.3.3 / 4.4)  -> inf -> NaN -> 0
  4. Duplicate-row drop          (CICIDS2017 hygiene)
  5. Label transformation        (3.3.4 / 4.4)  -> BENIGN=0, else 1
     Original attack names are kept in Attack_Type for per-attack recall.
  6. Feature selection           (3.3.5 / 4.4)  -> drop leakage IDs, keep time
  7. Numeric conversion          (3.3.6 / 4.4)  -> drop remaining non-numeric
  8. Class balancing             optional, train-fold only — default off
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_", regex=False)
    )
    return df


def normalize_source_file(df: pd.DataFrame) -> pd.DataFrame:
    """Accept the loader's Source_File column or a legacy __source_file name."""
    df = df.copy()
    if (
        "__source_file" in df.columns
        and config.SOURCE_FILE_COLUMN not in df.columns
    ):
        df = df.rename(columns={"__source_file": config.SOURCE_FILE_COLUMN})
    return df


def parse_and_sort_time(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Timestamp and sort so later rolling / time splits are real."""
    df = df.copy()
    if config.TIME_COLUMN not in df.columns:
        logger.warning(
            "No %s column — leaving row order as-is", config.TIME_COLUMN
        )
        return df

    parsed = pd.to_datetime(df[config.TIME_COLUMN], errors="coerce")
    n_bad = int(parsed.isna().sum())
    if n_bad == len(df):
        logger.warning("Could not parse any timestamps — leaving row order as-is")
        return df
    if n_bad:
        logger.warning("Dropping %s rows with unparseable timestamps", n_bad)
        keep = parsed.notna()
        df = df.loc[keep].copy()
        parsed = parsed.loc[keep]

    df[config.TIME_COLUMN] = parsed
    sort_cols = [config.TIME_COLUMN]
    if config.SOURCE_FILE_COLUMN in df.columns:
        sort_cols.append(config.SOURCE_FILE_COLUMN)
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def handle_missing_and_infinite(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def transform_label(df: pd.DataFrame) -> pd.DataFrame:
    if config.LABEL_COLUMN not in df.columns:
        raise KeyError(
            f"Expected label column '{config.LABEL_COLUMN}' not found. "
            f"Columns present: {list(df.columns)[:10]}..."
        )
    df = df.copy()
    labels = df[config.LABEL_COLUMN].astype(str).str.strip()
    df[config.ATTACK_TYPE_COLUMN] = labels
    df[config.LABEL_COLUMN] = (labels != config.BENIGN_LABEL).astype(int)
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    protected = set(config.NON_FEATURE_COLUMNS)
    to_drop = [
        c
        for c in config.LEAKAGE_COLUMNS
        if c in df.columns and c not in protected
    ]
    logger.info("Dropping leakage columns: %s", to_drop)
    return df.drop(columns=to_drop, errors="ignore")


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strip label / time / file / attack-name columns before model fit."""
    return df.drop(
        columns=[c for c in config.NON_FEATURE_COLUMNS if c in df.columns],
        errors="ignore",
    )


def to_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in config.NON_FEATURE_COLUMNS if c in df.columns]
    protected = df[keep]
    features = df.drop(columns=keep)
    numeric_features = features.select_dtypes(include=[np.number])
    dropped = set(features.columns) - set(numeric_features.columns)
    if dropped:
        logger.info("Dropping non-numeric feature columns: %s", dropped)
    result = numeric_features.copy()
    for column in keep:
        result[column] = protected[column].values
    return result


def balance_classes(df: pd.DataFrame, random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Random undersample. Prefer train-fold sample weights over this."""
    counts = df[config.LABEL_COLUMN].value_counts()
    if len(counts) < 2:
        logger.warning("Only one class present after preprocessing — skipping balancing.")
        return df
    minority_count = counts.min()
    logger.info("Class counts before balancing: %s", counts.to_dict())
    parts = [
        group.sample(n=minority_count, random_state=random_state)
        for _, group in df.groupby(config.LABEL_COLUMN)
    ]
    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logger.info(
        "Class counts after undersampling: %s",
        balanced[config.LABEL_COLUMN].value_counts().to_dict(),
    )
    return balanced


def run_preprocessing(df: pd.DataFrame, balance: bool = False) -> pd.DataFrame:
    """
    Full preprocess. Default keeps the natural class mix and chronological
    order so classification holdout metrics and forecasting splits are valid.
    """
    df = standardize_columns(df)
    df = normalize_source_file(df)
    df = parse_and_sort_time(df)
    df = handle_missing_and_infinite(df)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %s duplicate rows", before - len(df))
    df = transform_label(df)
    df = drop_leakage_columns(df)
    df = to_numeric_only(df)
    if balance:
        logger.warning(
            "balance=True undersamples the full frame, including any later "
            "holdout. Prefer class-balanced sample weights on the train fold."
        )
        df = balance_classes(df)
    return df


if __name__ == "__main__":
    import argparse

    from src.data.load import load_raw_dataset

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--balance",
        action="store_true",
        help="undersample the full frame (inflates holdout metrics; not recommended)",
    )
    parser.add_argument("--out", default=str(config.DATA_PROCESSED_DIR / "clean_dataset.csv"))
    args = parser.parse_args()

    raw = load_raw_dataset()
    clean = run_preprocessing(raw, balance=args.balance)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out, index=False)
    print(f"Wrote {clean.shape[0]} rows x {clean.shape[1]} cols -> {args.out}")
