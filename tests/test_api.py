def _complete(columns, fill=1.0):
    return {col: float(fill) for col in columns}


def test_health_is_liveness_without_models(empty_api_client):
    resp = empty_api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ready"] is False
    assert body["classifier_loaded"] is False


def test_ready_and_classify_are_503_without_models(empty_api_client):
    ready = empty_api_client.get("/ready")
    assert ready.status_code == 503
    classify = empty_api_client.post("/classify", json={"features": {"Flow_Duration": 100}})
    assert classify.status_code == 503
    forecast = empty_api_client.post("/forecast", json={"features": {"Flow_Duration": 100}})
    assert forecast.status_code == 503


def test_classify_422_when_features_missing(api_client, trained_artifacts):
    resp = api_client.post("/classify", json={"features": {"Flow_Duration": 100}})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["message"] == "Missing required features"
    expected = set(trained_artifacts["classifier_cols"]) - {"Flow_Duration"}
    assert set(detail["missing"]) == expected


def test_classify_422_on_non_finite(api_client, trained_artifacts):
    features = _complete(trained_artifacts["classifier_cols"])
    features[trained_artifacts["classifier_cols"][0]] = "inf"
    resp = api_client.post("/classify", json={"features": features})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    if isinstance(detail, dict):
        assert detail["message"] == "Non-finite feature values"
    else:
        assert resp.status_code == 422


def test_classify_200_with_complete_features(api_client, trained_artifacts):
    features = _complete(trained_artifacts["classifier_cols"])
    resp = api_client.post("/classify", json={"features": features})
    assert resp.status_code == 200
    body = resp.json()
    assert "is_anomalous" in body
    assert body["model_version"]
    assert body["model"] == "logistic_regression"


def test_ready_200_when_classifier_loaded(api_client):
    resp = api_client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert resp.json()["classifier_loaded"] is True


def test_forecast_422_and_200(api_client, trained_artifacts):
    incomplete = api_client.post("/forecast", json={"features": {}})
    assert incomplete.status_code == 422
    features = _complete(trained_artifacts["forecaster_cols"])
    ok = api_client.post("/forecast", json={"features": features})
    assert ok.status_code == 200
    body = ok.json()
    assert "predicted_traffic_volume" in body
    assert body["model_version"]


def test_explain_same_vector_as_classify(api_client, trained_artifacts):
    features = _complete(trained_artifacts["classifier_cols"])
    resp = api_client.post("/explain", json={"features": features})
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "coefficients"
    assert body["contributions"]
    assert body["model_version"]
    names = {row["feature"] for row in body["contributions"]}
    assert names.issubset(set(trained_artifacts["classifier_cols"]))


def test_model_card_is_written(trained_artifacts, model_dir):
    from src import config
    from src.models.artifacts import read_model_card

    card = read_model_card(config.CLASSIFIER_CARD_PATH)
    assert card is not None
    assert card["task"] == "classification"
    assert card["model_name"] == "logistic_regression"
    assert card["feature_columns"] == trained_artifacts["classifier_cols"]
    assert card["metrics"]["model"] == "logistic_regression"
