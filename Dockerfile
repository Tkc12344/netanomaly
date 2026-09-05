# --- build stage: install deps into a venv so the runtime image stays lean ---
FROM python:3.12-slim AS builder

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# API/pipeline only needs the core deps at runtime — skip shap/torch/dev tools
RUN grep -vE '^(shap|torch|pytest|httpx|matplotlib|ruff)' requirements.txt > requirements.runtime.txt \
    && pip install --no-cache-dir -r requirements.runtime.txt

# --- runtime stage ---
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models

COPY src ./src

# Artifacts are never baked in. A PVC, bind mount, or
# `python -m src.models.sync --pull` populates this directory at runtime.
RUN mkdir -p /app/models && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
