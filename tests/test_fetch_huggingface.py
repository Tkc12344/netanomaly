"""Hugging Face /rows fetcher — mocked httpx, never hits the network."""
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd
import pytest

from src import config
from src.data.fetch_huggingface import fetch_cicids, page_offsets
from src.data.load import load_raw_dataset
from src.data.preprocess import run_preprocessing, standardize_columns

NUM_ROWS_TOTAL = 10_000
PAGE = 100


def _label_for_offset(offset: int) -> str:
    return "BENIGN" if offset == 0 else "DoS Hulk"


def _fake_row(row_idx: int, label: str) -> dict:
    return {
        " Destination Port": 80,
        " Flow Duration": 100 + row_idx,
        "Total Length of Fwd Packets": 12 + (row_idx % 50),
        " Label": label,
    }


def _page(offset: int, length: int) -> dict:
    rows = []
    label = _label_for_offset(offset)
    for i in range(length):
        idx = offset + i
        if idx >= NUM_ROWS_TOTAL:
            break
        rows.append({"row_idx": idx, "row": _fake_row(idx, label), "truncated_cells": []})
    return {
        "features": [],
        "rows": rows,
        "num_rows_total": NUM_ROWS_TOTAL,
        "num_rows_per_page": PAGE,
        "partial": False,
    }


def _mock_client(requested: list[int]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        parsed = urlparse(str(request.url))
        assert parsed.path.endswith("/rows") or parsed.path == "/rows"
        params = parse_qs(parsed.query)
        offset = int(params["offset"][0])
        length = int(params["length"][0])
        assert length == PAGE
        requested.append(offset)
        return httpx.Response(200, json=_page(offset, length))

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="https://datasets-server.huggingface.co")


def test_page_offsets_strided_are_not_only_zero():
    offsets = page_offsets(NUM_ROWS_TOTAL, 300, page_length=PAGE, sequential=False)
    assert offsets[0] == 0
    assert len(offsets) == 3
    assert offsets != [0, 100, 200]
    assert offsets[1] > PAGE


def test_page_offsets_sequential_are_contiguous():
    offsets = page_offsets(NUM_ROWS_TOTAL, 300, page_length=PAGE, sequential=True)
    assert offsets == [0, 100, 200]


def test_page_offsets_full_split():
    offsets = page_offsets(250, 0, page_length=PAGE, sequential=True)
    assert offsets == [0, 100, 200]


def test_fetch_writes_csv_and_uses_strided_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "HF_ROWS_URL", "https://datasets-server.huggingface.co/rows")
    requested: list[int] = []
    path = fetch_cicids(
        client=_mock_client(requested),
        max_rows=300,
        sequential=False,
        out_dir=tmp_path,
    )
    assert path == tmp_path / config.HF_OUTPUT_FILENAME
    assert path.is_file()
    assert requested[0] == 0
    assert 0 in requested
    assert requested != [0]
    assert any(off > PAGE for off in requested)
    assert requested == page_offsets(NUM_ROWS_TOTAL, 300, page_length=PAGE, sequential=False)

    raw = pd.read_csv(path)
    assert config.TIME_COLUMN in raw.columns
    assert len(raw) == 300
    labels = raw[" Label"].astype(str).str.strip() if " Label" in raw.columns else raw["Label"]
    assert set(labels.str.strip()) >= {"BENIGN", "DoS Hulk"}


def test_fetch_sequential_paginates_from_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HF_ROWS_URL", "https://datasets-server.huggingface.co/rows")
    requested: list[int] = []
    fetch_cicids(
        client=_mock_client(requested),
        max_rows=200,
        sequential=True,
        out_dir=tmp_path,
    )
    assert requested == [0, 100]


def test_fetch_csv_loads_and_preprocesses(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "HF_ROWS_URL", "https://datasets-server.huggingface.co/rows")
    fetch_cicids(
        client=_mock_client([]),
        max_rows=200,
        sequential=False,
        out_dir=tmp_path,
    )
    raw = load_raw_dataset(tmp_path)
    stripped = standardize_columns(raw)
    assert "Label" in stripped.columns
    assert "Total_Length_of_Fwd_Packets" in stripped.columns
    clean = run_preprocessing(raw, balance=False)
    assert set(clean[config.LABEL_COLUMN].unique()) <= {0, 1}
    assert set(clean[config.ATTACK_TYPE_COLUMN]) >= {"BENIGN", "DoS Hulk"}
    assert config.TIME_COLUMN in clean.columns
    assert config.SOURCE_FILE_COLUMN in clean.columns
    assert len(clean) == 200


def test_load_error_mentions_hf_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_huggingface"):
        load_raw_dataset(tmp_path)
