"""
Load or publish trained artifacts without baking them into the API image.

  python -m src.models.sync --pull --wait
      Download from MODEL_STORAGE_BASE_URL if set, then wait until
      classifier.joblib is on disk (a training Job writing a shared PVC).

  python -m src.models.sync --push
      HTTP PUT each artifact to MODEL_STORAGE_BASE_URL. No-op if unset.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

REQUIRED_NAMES = ("classifier.joblib", "feature_columns.json")
OPTIONAL_NAMES = (
    "classifier_card.json",
    "forecaster.joblib",
    "forecaster_feature_columns.json",
    "forecaster_card.json",
)
ALL_NAMES = REQUIRED_NAMES + OPTIONAL_NAMES


def models_ready(models_dir: Path | None = None) -> bool:
    root = models_dir or config.MODELS_DIR
    return all((root / name).exists() for name in REQUIRED_NAMES)


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("MODEL_STORAGE_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _urlopen(url: str, data: bytes | None = None, method: str = "GET") -> bytes:
    request = urllib.request.Request(
        url, data=data, headers=_auth_headers(), method=method
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def pull(base_url: str, models_dir: Path | None = None) -> None:
    root = models_dir or config.MODELS_DIR
    root.mkdir(parents=True, exist_ok=True)
    base = base_url.rstrip("/")
    for name in ALL_NAMES:
        url = f"{base}/{name}"
        dest = root / name
        try:
            payload = _urlopen(url)
        except urllib.error.URLError as exc:
            if name in REQUIRED_NAMES:
                raise RuntimeError(f"Required artifact missing at {url}") from exc
            logger.warning("Optional artifact not at %s (%s)", url, exc)
            continue
        dest.write_bytes(payload)
        logger.info("Pulled %s (%s bytes)", dest, len(payload))
    if not models_ready(root):
        raise RuntimeError(f"Pull from {base} did not produce required artifacts")


def push(base_url: str, models_dir: Path | None = None) -> None:
    root = models_dir or config.MODELS_DIR
    base = base_url.rstrip("/")
    for name in ALL_NAMES:
        src = root / name
        if not src.exists():
            if name in REQUIRED_NAMES:
                raise FileNotFoundError(f"Cannot push missing {src}")
            continue
        url = f"{base}/{name}"
        _urlopen(url, data=src.read_bytes(), method="PUT")
        logger.info("Pushed %s -> %s", src, url)


def wait_for_models(
    timeout: int,
    models_dir: Path | None = None,
    interval: float = 2.0,
) -> None:
    root = models_dir or config.MODELS_DIR
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if models_ready(root):
            logger.info("Models ready in %s", root)
            return
        time.sleep(interval)
    raise TimeoutError(
        f"Timed out after {timeout}s waiting for {', '.join(REQUIRED_NAMES)} in {root}"
    )


def storage_base_url(cli_value: str | None = None) -> str:
    if cli_value:
        return cli_value.strip()
    return os.environ.get("MODEL_STORAGE_BASE_URL", "").strip()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base = storage_base_url(args.base_url)

    if args.pull and base:
        pull(base)
    elif args.pull and not base:
        logger.info("No MODEL_STORAGE_BASE_URL; skip pull")

    if args.wait and not models_ready():
        wait_for_models(args.timeout)

    if args.push and base:
        push(base)
    elif args.push and not base:
        logger.info("No MODEL_STORAGE_BASE_URL; skip push")

    if args.push and not args.pull and not args.wait:
        return 0
    if models_ready():
        return 0
    logger.error(
        "No classifier artifacts in %s. Train a Job onto the models PVC "
        "or set MODEL_STORAGE_BASE_URL.",
        config.MODELS_DIR,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
