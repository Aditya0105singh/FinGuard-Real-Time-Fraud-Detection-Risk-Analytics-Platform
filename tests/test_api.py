"""API contract tests for the FinGuard FastAPI service."""

VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


class TestHealth:
    def test_health_returns_200(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_health_payload(self, api_client):
        body = api_client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_version"] == "1.0"


class TestPredict:
    def test_predict_returns_200(self, api_client, valid_payload):
        resp = api_client.post("/predict", json=valid_payload)
        assert resp.status_code == 200

    def test_predict_response_shape(self, api_client, valid_payload):
        body = api_client.post("/predict", json=valid_payload).json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["risk_level"] in VALID_RISK_LEVELS
        assert 0.0 <= body["confidence"] <= 1.0
        assert isinstance(body["recommendation"], str) and body["recommendation"]

    def test_predict_includes_shap_factors(self, api_client, valid_payload):
        body = api_client.post("/predict", json=valid_payload).json()
        factors = body["top_factors"]
        assert factors is not None and 1 <= len(factors) <= 5
        for factor in factors:
            assert factor["direction"] in {"increases_risk", "decreases_risk"}
            assert isinstance(factor["contribution"], float)

    def test_predict_rejects_wrong_pca_length(self, api_client, valid_payload):
        valid_payload["v1_to_v10"] = [0.1, 0.2, 0.3]  # only 3 values
        resp = api_client.post("/predict", json=valid_payload)
        assert resp.status_code == 422

    def test_predict_rejects_negative_amount(self, api_client, valid_payload):
        valid_payload["amount"] = -50.0
        resp = api_client.post("/predict", json=valid_payload)
        assert resp.status_code == 422

    def test_predict_rejects_invalid_hour(self, api_client, valid_payload):
        valid_payload["transaction_hour"] = 24
        resp = api_client.post("/predict", json=valid_payload)
        assert resp.status_code == 422

    def test_predict_rejects_missing_field(self, api_client, valid_payload):
        del valid_payload["amount"]
        resp = api_client.post("/predict", json=valid_payload)
        assert resp.status_code == 422


class TestBatchPredict:
    def test_batch_predict(self, api_client, valid_payload):
        resp = api_client.post(
            "/batch-predict", json={"transactions": [valid_payload] * 3}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["predictions"]) == 3

    def test_batch_predict_rejects_empty_list(self, api_client):
        resp = api_client.post("/batch-predict", json={"transactions": []})
        assert resp.status_code == 422


class TestStats:
    def test_stats_returns_training_metrics(self, api_client):
        resp = api_client.get("/stats")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("accuracy", "precision", "recall", "f1",
                    "auc_roc", "average_precision"):
            assert 0.0 <= body[key] <= 1.0
        assert body["model_version"] == "1.0"
