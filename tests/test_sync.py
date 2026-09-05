import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from src.models.sync import (
    main,
    models_ready,
    pull,
    push,
    wait_for_models,
)


def _write_required(root: Path) -> None:
    (root / "classifier.joblib").write_bytes(b"clf")
    (root / "feature_columns.json").write_text('["Flow_Duration"]')


def test_models_ready_requires_both_files(tmp_path):
    assert models_ready(tmp_path) is False
    (tmp_path / "classifier.joblib").write_bytes(b"x")
    assert models_ready(tmp_path) is False
    (tmp_path / "feature_columns.json").write_text("[]")
    assert models_ready(tmp_path) is True


def test_wait_for_models_sees_late_write(tmp_path):
    def _writer():
        import time

        time.sleep(0.05)
        _write_required(tmp_path)

    thread = threading.Thread(target=_writer)
    thread.start()
    wait_for_models(timeout=2, models_dir=tmp_path, interval=0.02)
    thread.join()
    assert models_ready(tmp_path)


def test_wait_for_models_times_out(tmp_path):
    with pytest.raises(TimeoutError):
        wait_for_models(timeout=0, models_dir=tmp_path, interval=0.01)


def test_pull_and_push_round_trip(tmp_path):
    store: dict[str, bytes] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.lstrip("/")
            if name not in store:
                self.send_error(404)
                return
            body = store[name]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self):
            length = int(self.headers.get("Content-Length", "0"))
            store[self.path.lstrip("/")] = self.rfile.read(length)
            self.send_response(201)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        _write_required(src)
        (src / "classifier_card.json").write_text('{"version": "1"}')
        push(base, src)
        pull(base, dest)
        assert (dest / "classifier.joblib").read_bytes() == b"clf"
        assert (dest / "classifier_card.json").read_text() == '{"version": "1"}'
        assert models_ready(dest)
    finally:
        server.shutdown()


def test_sync_main_fails_when_empty(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert main([]) == 1


def test_sync_main_ok_when_present(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    _write_required(tmp_path)
    assert main([]) == 0


def test_dockerfile_does_not_copy_models():
    text = Path("Dockerfile").read_text()
    assert "COPY models" not in text
    assert "mkdir -p /app/models" in text
