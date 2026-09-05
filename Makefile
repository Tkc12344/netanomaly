.PHONY: install synthetic-data pipeline api test docker-build docker-up docker-train lint k8s-apply k8s-train

install:
	pip install -r requirements.txt

synthetic-data:
	python -m src.data.generate_synthetic --rows 20000 --files 3

pipeline:
	python -m src.pipeline

api:
	uvicorn src.api.main:app --reload --port 8000

test:
	pytest tests/ -q

lint:
	ruff check src tests

docker-build:
	docker build -t netanomaly-api:local .

docker-up:
	docker compose up api

docker-train:
	docker compose --profile train run --rm trainer

# Replace OWNER in the manifests (or set IMAGE) before these talk to a cluster.
IMAGE ?= ghcr.io/OWNER/netanomaly-api:latest

k8s-apply:
	kubectl apply -f k8s/configmap.yaml -f k8s/pvc.yaml
	sed "s|ghcr.io/OWNER/netanomaly-api:latest|$(IMAGE)|g" k8s/deployment.yaml | kubectl apply -f -
	kubectl apply -f k8s/service.yaml -f k8s/hpa.yaml

k8s-train:
	kubectl apply -f k8s/configmap.yaml -f k8s/pvc.yaml
	kubectl delete job netanomaly-train --ignore-not-found
	sed "s|ghcr.io/OWNER/netanomaly-api:latest|$(IMAGE)|g" k8s/train-job.yaml | kubectl apply -f -
	kubectl wait --for=condition=complete job/netanomaly-train --timeout=1800s
