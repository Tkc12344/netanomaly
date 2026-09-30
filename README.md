# Netanomaly

Detect anomalies in CICIDS-2017 network flows and forecast the next traffic volume. Flows are preprocessed in time order, a binary classifier is trained against a dummy baseline, a one-step volume forecaster is trained against a persist baseline, and a FastAPI service serves the artifacts. The Docker image is **code only** — models are trained separately and mounted or pulled at runtime.

Design, split rules, and artifact flow: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What it does

| Layer | Behaviour |
|---|---|
| Data | Concatenates every CSV in `data/raw/`. Fetches [`San0160/CICIDS-2017`](https://huggingface.co/datasets/San0160/CICIDS-2017) via Hugging Face `/rows`, or generates synthetic rows for CI. |
| Preprocess | Time sort, leakage-ID drop, binary `Label` plus original `Attack_Type`. The holdout keeps the natural class mix. |
| Classifier | Dummy (most-frequent), scaled logistic regression, Random Forest, Gradient Boosting. Sample weights on the **train** fold only. Per-attack recall on a binary decision. |
| Forecaster | Random Forest regressor, rolling stats grouped by source file, chronological split, persist (last-value) baseline. |
| API | `/classify`, `/forecast`, `/explain`. Missing features are 422. `/ready` is 503 until a classifier is loaded. Console at `/ui`. |
| Deploy | Training Job + PVC (or `MODEL_STORAGE_BASE_URL`). CI publishes to GHCR. |

The Transformer in `src/models/transformer.py` is an experiment. Trees outperform it on this tabular data; the API does not load it.

## Quickstart

Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make hf-data      # strided 20k sample from Hugging Face → data/raw/
make pipeline     # preprocess, train, write models/ + model cards
make api          # uvicorn on :8000
# open http://localhost:8000/ui
```

Offline / CI without Hugging Face: `make synthetic-data` then `make pipeline`.

After the API is up, `/schema` is the request contract. A complete dummy vector:

```bash
FEATURES=$(curl -s localhost:8000/schema | python -c \
  "import json,sys; s=json.load(sys.stdin); print(json.dumps({'features': {c: 1.0 for c in s['classifier_features']}}))")

curl -s -X POST localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d "$FEATURES"
```

Use the same body on `/explain`. Forecast needs `forecaster_features` from `/schema`, not the classifier list.

```bash
make test         # unit + API tests (no network)
make e2e          # generate → pipeline → classify/forecast/explain
```

## Data

Training CSVs come from Hugging Face [`San0160/CICIDS-2017`](https://huggingface.co/datasets/San0160/CICIDS-2017) through datasets-server `/rows` (max **100 rows per request**):

```bash
curl -X GET \
  "https://datasets-server.huggingface.co/rows?dataset=San0160%2FCICIDS-2017&config=default&split=train&offset=0&length=100"

make hf-data                                    # default: strided 20k sample
python -m src.data.fetch_huggingface --sequential --max-rows 500
HF_MAX_ROWS=0 python -m src.data.fetch_huggingface   # full split (~28k HTTP calls)
```

- The split is **2,830,743** rows. `offset=0` is Monday BENIGN only. The default fetch **strides** 100-row pages across the week so attacks are present.
- `--sequential` paginates `0, 100, 200, …` like the curl above.
- `--max-rows 0` walks the full split. `--max-rows N` caps the sample.
- This dump has no capture `Timestamp`. The fetcher assigns a **proxy** from each `row_idx` (concatenation order, not packet time) so rolling features have a defined order.
- `load.py` concatenates every `*.csv` in `data/raw/`. Remove leftover files before a clean CICIDS-only train.
- `pytest` never hits Hugging Face.

On a cluster Job, set `DATA_SOURCE=huggingface` (and optionally `HF_MAX_ROWS`) so an empty `data/raw` fetches instead of generating synthetic rows.

## API

Base URL: `http://localhost:8000`. Interactive docs: `/docs`.

| Method | Path | Behaviour |
|---|---|---|
| GET | `/` | Service name, version, path hints. |
| GET | `/ui` | Operator console (classify / forecast / explain). |
| GET | `/health` | Liveness. Always 200. `ready` is whether a classifier is loaded. |
| GET | `/ready` | 503 until `classifier.joblib` + feature columns load. Kubernetes readiness probe. |
| GET | `/schema` | Required feature names for `/classify` and `/forecast`. |
| POST | `/classify` | Binary anomaly + probability + `model_version`. 422 if any trained column is missing or non-finite. |
| POST | `/forecast` | Next-step `Total_Length_of_Fwd_Packets`. Same completeness rule. |
| POST | `/explain` | Same body as `/classify`. Coefficients for a linear pipeline; SHAP for trees when `shap` is installed. |

Request body:

```json
{ "features": { "Flow_Duration": 1.0, "…every column from /schema…": 0.0 } }
```

Optional `API_KEY`: when set, POST routes need `Authorization: Bearer <key>` or `X-API-Key: <key>`. `/health`, `/ready`, and `/schema` stay public so probes still work.

## Configuration

Copy `.env.example` to `.env` before `docker compose` (Compose requires the file). Local `make` commands use defaults if unset.

| Variable | Default | Meaning |
|---|---|---|
| `DATA_RAW_DIR` | `data/raw` | Input CSVs |
| `DATA_PROCESSED_DIR` | `data/processed` | Optional preprocess dump |
| `MODELS_DIR` | `models` | Artifacts (`classifier.joblib`, cards, column lists) |
| `DATA_SOURCE` | `synthetic` | `huggingface` fetches `/rows` when `data/raw` is empty |
| `HF_MAX_ROWS` | `20000` | Hugging Face row cap (`0` = full split) |
| `N_ESTIMATORS` | `200` | Trees for RF classifier / forecaster |
| `API_KEY` | empty | Optional POST auth |
| `MODEL_STORAGE_BASE_URL` | empty | HTTP prefix for artifact sync |
| `MODEL_STORAGE_TOKEN` | empty | Bearer token for that sync |
| `TRAIN_ROWS` / `TRAIN_FILES` | `8000` / `2` | Synthetic size inside the k8s Job |

Column policy (leakage IDs, protected columns, forecast target) lives in `src/config.py`.

## Make targets

| Target | What it runs |
|---|---|
| `make hf-data` | Hugging Face `/rows` → `data/raw/` |
| `make synthetic-data` | CICIDS-shaped synthetic CSVs |
| `make pipeline` | Load, preprocess, train, write `models/` |
| `make api` | `uvicorn` on port 8000 |
| `make test` / `make e2e` / `make lint` | pytest / e2e / ruff |
| `make docker-build` | Image `netanomaly-api:local` |
| `make docker-train` | Compose trainer → `./models` |
| `make docker-up` | Compose API, `./models` mounted read-only |
| `make k8s-train` / `make k8s-apply` | Job then Deployment (`IMAGE=…`) |

## Layout

```
src/data/          load, HF /rows fetch, preprocess, synthetic generator
src/features/      grouped rolling stats, forecast target shift
src/models/        classifier, forecaster, transformer, explain, sync, train job
src/api/           FastAPI service
tests/             preprocess, splits, API 422/503/200, HF fetcher (mocked), e2e
k8s/               PVC, train Job, Deployment, Service, HPA
terraform/         EKS starting point (not apply-ready)
docs/ARCHITECTURE.md
```

## Docker

```bash
cp .env.example .env     # Compose reads .env; leave MODEL_STORAGE_* commented
make docker-build
make docker-train        # python -m src.models.train_job → ./models
docker compose up api    # mounts ./models read-only
```

The image does not `COPY models`. An empty `models/` directory means `/ready` fails, which is intentional.

## Kubernetes

Images come from GHCR. CI tag: `ghcr.io/<owner>/<repo>/netanomaly-api:<sha>`. Manifests still say `ghcr.io/OWNER/netanomaly-api:latest` — pass `IMAGE` (or edit the placeholder) before apply.

```bash
make k8s-train IMAGE=ghcr.io/<owner>/<repo>/netanomaly-api:<tag>
make k8s-apply IMAGE=ghcr.io/<owner>/<repo>/netanomaly-api:<tag>
```

1. `netanomaly-train` writes joblib + cards onto the `netanomaly-models` PVC.
2. The API init container runs `python -m src.models.sync --pull --wait` so the pod does not become ready without a classifier.
3. `/ready` is the readiness probe; `/health` is liveness.
4. Pods drop all capabilities, run as non-root, and the API root filesystem is read-only (1Gi memory limit).

On kind / minikube (one node) ReadWriteOnce is shared by both replicas. On multi-node, set `MODEL_STORAGE_BASE_URL` or use ReadWriteMany. Optional secrets: copy `k8s/secret.example.yaml` → `k8s/secret.yaml` and do not commit it.

## CI/CD and Terraform

`.github/workflows/ci-cd.yml` on `main`: ruff → pytest (tiny model in a temp dir, no Hugging Face) → push `ghcr.io/<owner>/<repo>/netanomaly-api` → `kubectl apply` when `KUBE_CONFIG` is a base64 kubeconfig. Without that secret, deploy skips apply and still passes. Retrain with `make k8s-train`, not on every push.

`terraform/` is an EKS starting point, not apply-ready. Fill VPC/subnets and a remote backend first. The registry is GHCR, not ECR.

## Limitations

- The forecaster follows trend and misses sudden spikes.
- The small Transformer loses to trees on this tabular data.
- Zero-filling inf/NaN is a known bias in the cleaned features.
- A 200-tree forest on full CICIDS can be large. Saved forests are pinned to one thread for serving; the API limit is 1Gi. Measure RSS before calling it near-real-time.
- Synthetic data exercises the pipeline and CI. Report CICIDS numbers from `make hf-data`, not from the generator.
- `joblib` artifacts are pickle-based. Treat `models/` as trusted output of this repo’s trainer, not as an untrusted upload.
- Optional `API_KEY` is not an identity provider. There is no rate limit or mTLS.

## Suggested path

1. Local: `make hf-data && make pipeline && make test && make api`
2. Docker: `make docker-build && make docker-train && docker compose up api`
3. Kubernetes on kind, then a real cluster
4. Green GitHub Actions; add `KUBE_CONFIG` when you want deploy
5. External secrets instead of a plain `Secret`
6. Terraform for EKS if you want AWS; keep GHCR as the registry
