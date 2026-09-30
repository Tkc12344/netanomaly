"""
Fetch CICIDS-2017 from Hugging Face datasets-server `/rows`.

The API caps each call at 100 rows:

    GET https://datasets-server.huggingface.co/rows
        ?dataset=San0160/CICIDS-2017&config=default&split=train
        &offset=0&length=100

Default fetch is a *strided* sample across the 2.83M-row train split so
Monday BENIGN is not the only class seen. `--sequential` paginates 0,
100, 200, … like the curl example. `--max-rows 0` walks the full split.
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import httpx
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}


def page_offsets(
    num_rows_total: int,
    max_rows: int,
    *,
    page_length: int = config.HF_PAGE_LENGTH,
    sequential: bool = False,
) -> list[int]:
    """Offsets for `/rows` pages of `page_length` (always 100 on this API)."""
    if num_rows_total <= 0 or page_length <= 0:
        return []
    target = num_rows_total if max_rows <= 0 else min(max_rows, num_rows_total)
    pages_needed = min(
        math.ceil(target / page_length),
        math.ceil(num_rows_total / page_length),
    )
    if sequential or pages_needed <= 1:
        return [i * page_length for i in range(pages_needed)]

    stride = max(
        page_length,
        (num_rows_total // pages_needed) // page_length * page_length,
    )
    offsets: list[int] = []
    offset = 0
    for _ in range(pages_needed):
        if offset >= num_rows_total:
            break
        offsets.append(offset)
        offset += stride
    return offsets


def _fetch_page(
    client: httpx.Client,
    offset: int,
    length: int,
    *,
    retries: int = config.HF_REQUEST_RETRIES,
) -> dict:
    params = {
        "dataset": config.HF_DATASET,
        "config": config.HF_CONFIG,
        "split": config.HF_SPLIT,
        "offset": offset,
        "length": length,
    }
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.get(config.HF_ROWS_URL, params=params)
            if response.status_code in _RETRY_STATUSES:
                last_error = httpx.HTTPStatusError(
                    f"{response.status_code} on offset={offset}",
                    request=response.request,
                    response=response,
                )
                logger.warning(
                    "Retryable %s on offset=%s (attempt %s/%s)",
                    response.status_code,
                    offset,
                    attempt,
                    retries,
                )
            else:
                response.raise_for_status()
                payload = response.json()
                if "rows" not in payload:
                    raise ValueError(
                        f"/rows response missing 'rows' at offset={offset}: "
                        f"{list(payload)[:8]}"
                    )
                return payload
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRY_STATUSES:
                raise
            last_error = exc
            logger.warning(
                "Request failed on offset=%s (attempt %s/%s): %s",
                offset,
                attempt,
                retries,
                exc,
            )
        except httpx.RequestError as exc:
            last_error = exc
            logger.warning(
                "Request failed on offset=%s (attempt %s/%s): %s",
                offset,
                attempt,
                retries,
                exc,
            )
        if attempt < retries:
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(
        f"Failed to fetch /rows offset={offset} after {retries} attempts"
    ) from last_error


def _rows_to_frame(pages: list[dict]) -> pd.DataFrame:
    records: list[dict] = []
    for payload in pages:
        for item in payload.get("rows", []):
            row = dict(item["row"])
            row["_row_idx"] = int(item["row_idx"])
            records.append(row)
    if not records:
        raise ValueError("Hugging Face /rows returned no rows")
    df = pd.DataFrame(records)
    df = df.sort_values("_row_idx", kind="mergesort").reset_index(drop=True)
    origin = pd.Timestamp(config.HF_TIMESTAMP_ORIGIN)
    df[config.TIME_COLUMN] = origin + pd.to_timedelta(df["_row_idx"], unit="s")
    return df.drop(columns=["_row_idx"])


def fetch_cicids(
    *,
    client: httpx.Client | None = None,
    max_rows: int = config.HF_DEFAULT_MAX_ROWS,
    sequential: bool = False,
    out_dir: Path | None = None,
    out_name: str = config.HF_OUTPUT_FILENAME,
    page_length: int = config.HF_PAGE_LENGTH,
) -> Path:
    """
    Paginate `/rows`, write one CSV under `out_dir`, return that path.

    `Timestamp` is a proxy from `row_idx` (concatenation order), not
    packet-capture time — this MachineLearningCSV dump has no time column.
    """
    out_dir = out_dir or config.DATA_RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in out_dir.glob("*.csv") if p.name != out_name)
    if existing:
        logger.warning(
            "Other CSVs already in %s (%s). load.py will concatenate them "
            "with the Hugging Face file. Remove them first if you want a "
            "clean CICIDS-only train.",
            out_dir,
            existing,
        )

    own_client = client is None
    if client is None:
        client = httpx.Client(timeout=config.HF_REQUEST_TIMEOUT_S)
    try:
        probe = _fetch_page(client, 0, page_length)
        num_rows_total = int(probe.get("num_rows_total") or 0)
        if num_rows_total <= 0:
            raise ValueError("Hugging Face /rows did not report num_rows_total")

        offsets = page_offsets(
            num_rows_total,
            max_rows,
            page_length=page_length,
            sequential=sequential,
        )
        if not offsets:
            raise ValueError("No /rows offsets to fetch")

        target = num_rows_total if max_rows <= 0 else min(max_rows, num_rows_total)
        n_calls = len(offsets)
        if max_rows <= 0 or target >= num_rows_total:
            logger.warning(
                "Fetching the full %s-row split via /rows needs ~%s HTTP "
                "calls (max %s rows each). This can take hours.",
                num_rows_total,
                math.ceil(num_rows_total / page_length),
                page_length,
            )
        logger.info(
            "Fetching %s/%s rows from %s (%s, %s pages of %s)",
            target,
            num_rows_total,
            config.HF_DATASET,
            "sequential" if sequential else "strided",
            n_calls,
            page_length,
        )

        pages = [probe] if offsets[0] == 0 else []
        for offset in offsets:
            if offset == 0 and pages:
                continue
            logger.info("  GET offset=%s length=%s", offset, page_length)
            pages.append(_fetch_page(client, offset, page_length))
    finally:
        if own_client:
            client.close()

    df = _rows_to_frame(pages)
    if max_rows > 0 and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    out_path = out_dir / out_name
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s rows x %s cols -> %s", df.shape[0], df.shape[1], out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=config.HF_DEFAULT_MAX_ROWS,
        help="row cap (default 20000). 0 = entire train split",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="paginate offset 0, 100, 200, … instead of a strided sample",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="destination directory (default DATA_RAW_DIR)",
    )
    parser.add_argument(
        "--out-name",
        default=config.HF_OUTPUT_FILENAME,
        help="CSV filename inside --out-dir",
    )
    args = parser.parse_args(argv)
    fetch_cicids(
        max_rows=args.max_rows,
        sequential=args.sequential,
        out_dir=args.out_dir,
        out_name=args.out_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
