# Netanomaly

Network anomaly detection and traffic forecasting, implemented from Akshay Tiwari's MSc thesis *Network Anomaly Detection and Traffic Forecasting Using Machine Learning and Sequence Models* (CSUN, 2026).

The repo is a working system, not a notebook dump: CICIDS-shaped flows go through a time-aware preprocess, a binary classifier with a dummy baseline, a one-step volume forecaster with a persist baseline, a FastAPI service, and a Kubernetes path that **does not bake models into the image**.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design: data flow, split rules, API contract, and how artifacts move from a training Job onto a PVC.

## What you get

| Layer | What it does |
|---|---|
| Data | Merges every CSV in `data/raw/`. Synthetic CICIDS-shaped generator if you do not have the real set yet. |
| Preprocess | Time sort, leakage-ID drop, binary label + original `Attack_Type`, natural class mix on the holdout. |
| Classifier | Dummy (most-frequent) + scaled logistic regression + Random Forest + Gradient Boosting. Train-fold sample weights only. Per-attack recall on a binary decision. |
| Forecaster | Random Forest regressor, grouped rolling stats, chronological split, persist (last-value) baseline. |
| API | `/classify`, `/forecast`, `/explain`. Complete feature vectors only. `/ready` is 503 until a classifier is loaded. |
| Deploy | Image is code-only. Models come from a training Job + PVC, or `MODEL_STORAGE_BASE_URL`. CI publishes to GHCR. |

The optional Transformer (`src/models/transformer.py`) is kept to reproduce the thesis finding that it loses to trees on this tabular data. It is not the served model.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make synthetic-data          # CICIDS-shaped CSVs in data/raw/
make pipeline                # preprocess, train, write models/ + model cards
make api                     # uvicorn on :8000
```

After training, `models/feature_columns.json` is the request schema. A one-key body is rejected (422):

```bash
python - <<'PY'
import json
from pathlib import Path
cols = json.loads(Path("models/feature_columns.json").read_text())
print(json.dumps({"features": {c: 1.0 for c in cols}}))
PY

curl -s -X POST localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d @-   # paste the JSON from the snippet above
```

Same vector on `/explain` returns per-feature contributions (coefficients for a linear pipeline; SHAP for trees when `shap` is installed).

Or: `make install`, `make test`.

## CICIDS2017

This repo cannot download CICIDS2017 (registration wall). To use the real set:

1. Get the MachineLearningCSV flows from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Remove synthetic CSVs from `data/raw/`.
3. Drop the real CSVs there.
4. Run `python -m src.pipeline` again. `load.py` concatenates whatever it finds.

Expect a `Label` column with `BENIGN` for normal traffic. Rename via `src/config.py` if your headers differ.

## API

| Method | Path | Behaviour |
|---|---|---|
| GET | `/health` | Liveness. Always 200. `ready` says whether a classifier is loaded. |
| GET | `/ready` | 503 until `classifier.joblib` + feature columns are loaded. Kubernetes readiness uses this. |
| POST | `/classify` | Binary anomaly + probability + `model_version`. 422 if any trained column is missing. |
| POST | `/forecast` | Next-step `Total_Length_of_Fwd_Packets`. Same completeness rule. |
| POST | `/explain` | Same body as `/classify`. |

## Project layout

```
src/data/          load, preprocess, synthetic generator
src/features/      grouped rolling stats, forecast target shift
src/models/        classifier, forecaster, transformer, SHAP, sync, train job
src/api/           FastAPI service
tests/             preprocess, splits, API 422/503/200, artifact sync
k8s/               PVC, train Job, Deployment, Service, HPA
docs/ARCHITECTURE.md
```

## Docker

```bash
make docker-build
make docker-train            # python -m src.models.train_job → ./models
docker compose up api        # mounts ./models read-only
```

The image does not `COPY models`. An empty `models/` directory means `/ready` fails, which is intentional.

## Kubernetes

Images come from GHCR. Replace `OWNER` or pass `IMAGE=...`.

```bash
make k8s-train IMAGE=ghcr.io/<org>/netanomaly/netanomaly-api:<tag>
make k8s-apply IMAGE=ghcr.io/<org>/netanomaly/netanomaly-api:<tag>
```

1. `netanomaly-train` writes joblib + cards onto the `netanomaly-models` PVC.
2. The API init container runs `python -m src.models.sync --pull --wait` so the pod does not start without a classifier.
3. `/ready` is the readiness probe; `/health` is liveness.

On kind / minikube (one node) ReadWriteOnce is shared by both replicas. On multi-node, set `MODEL_STORAGE_BASE_URL` or use ReadWriteMany. Secrets are optional (`k8s/secret.example.yaml` → `secret.yaml`, never commit it).

## CI/CD and Terraform

`.github/workflows/ci-cd.yml`: lint → pytest (trains a tiny model in a temp dir) → push `ghcr.io/<owner>/<repo>/netanomaly-api` on `main` → `kubectl apply` when `KUBE_CONFIG` is a base64 kubeconfig. Without that secret, deploy skips apply and still passes. Retrain with `make k8s-train`, not on every push.

`terraform/` is an EKS starting point, not apply-ready. Fill VPC/subnets and a remote backend first. The registry is GHCR, not ECR.

## Thesis map

| Thesis | This repo |
|---|---|
| 3.2 Data collection | `src/data/load.py` |
| 3.3 / 4.4 Preprocess | Time parse + sort, inf/NaN → 0, duplicate drop, `Attack_Type` kept, leakage IDs dropped. Holdout is **not** undersampled. |
| 3.4 Features | Grouped rolling mean/std; forecast target shifted inside each source file. |
| 3.5.1 / 4.5 Classification | Dummy + scaled LR + RF + GB; stratified split; train-fold weights. |
| 3.5.2 / 4.8 Forecasting | RF regressor; chronological split; persist MAE/RMSE. |
| 3.5.3 / 4.9 Transformer | Optional, not served. |
| 3.5.4 / 4.7 Explain | `/explain` + `src/models/explain.py`. |
| 3.6 Evaluation | Precision / recall / F1 / AUC-PR + prevalence + per-attack recall; MAE / RMSE vs persist. |
| 3.7 / 4.10 Deploy | Latency helper, FastAPI, Job + PVC, GHCR. |

## Limitations

- The forecaster follows trend and misses sudden spikes (thesis Sec 5.5.3 / 6.4).
- The small Transformer loses to trees on this tabular data (Sec 4.9 / 6.6).
- Zero-filling inf/NaN is a documented bias (Sec 3.3.3).
- A 200-tree forest on full CICIDS may not fit the 512Mi API limit; measure before you serve it.
- Synthetic data is for exercising the pipeline. Report CICIDS numbers from the real CSVs.

## Suggested path

1. Local pipeline (`make synthetic-data && make pipeline && make test`)
2. API (`make api`)
3. Docker (`make docker-build && make docker-train && docker compose up api`)
4. Kubernetes on kind, then a real cluster
5. Green GitHub Actions + `KUBE_CONFIG` for deploy
6. External secrets instead of a plain `Secret`
7. Terraform for EKS if you want AWS; keep GHCR as the registry
