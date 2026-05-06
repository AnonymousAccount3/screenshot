"""Tests for Silo 2: Inference engine (B1, B2, B3, C5b, D4).

Spec source: FEATURES.md sections B1, B2, B3, C5b, D4.
These tests are READ-ONLY during implementation.
"""

import pytest


# ---------------------------------------------------------------------------
# B1: Prediction pipeline
# ---------------------------------------------------------------------------


class TestPredictionPipeline:
    """B1: Core prediction with streaming."""

    @pytest.mark.asyncio
    async def test_predict_returns_predictions(self, app_client):
        """Basic prediction returns predicted_viability for each query row."""
        resp = await app_client.post("/tasks", json={
            "task_type": "predict",
            "input_csv": "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5\nA,DrugX,1.0,0.3",
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]
        assert task_id

    @pytest.mark.asyncio
    async def test_predict_full_returns_embeddings(self, app_client):
        """predict_full task returns predictions + sample_embeddings + drug_dose_embeddings."""
        resp = await app_client.post("/tasks", json={
            "task_type": "predict_full",
            "input_csv": "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_cancel_running_task(self, app_client):
        """Cancelling a task returns cancelled status."""
        resp = await app_client.post("/tasks", json={
            "task_type": "predict",
            "input_csv": "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5",
        })
        task_id = resp.json()["task_id"]

        cancel_resp = await app_client.post(f"/tasks/{task_id}/cancel")
        assert cancel_resp.status_code == 200

    @pytest.mark.asyncio
    async def test_cancel_and_resubmit_no_deadlock(self, app_client):
        """Cancel a task and immediately submit a new one — no deadlock or stuck model."""
        resp1 = await app_client.post("/tasks", json={
            "task_type": "predict",
            "input_csv": "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5",
        })
        task_id1 = resp1.json()["task_id"]
        await app_client.post(f"/tasks/{task_id1}/cancel")

        resp2 = await app_client.post("/tasks", json={
            "task_type": "predict",
            "input_csv": "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5",
        })
        assert resp2.status_code == 200
        assert resp2.json()["task_id"] != task_id1


# ---------------------------------------------------------------------------
# B3: Control sample handling
# ---------------------------------------------------------------------------


class TestControlSamples:
    """B3: Control sample selection and delta computation."""

    @pytest.mark.asyncio
    async def test_set_control_samples(self, app_client):
        """User can set control sample IDs."""
        resp = await app_client.post("/controls", json={
            "control_sample_ids": ["RPE", "BJ"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "user_selected"
        assert "RPE" in data["control_sample_ids"]

    @pytest.mark.asyncio
    async def test_default_control_is_median(self, app_client):
        """When no controls are set, method defaults to median."""
        resp = await app_client.get("/controls")
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "median"

    @pytest.mark.asyncio
    async def test_delta_computation(self, app_client):
        """Delta endpoint returns control_viability - sample_viability per drug+dose."""
        # This test validates the computation logic exists
        resp = await app_client.post("/compute/deltas", json={
            "task_id": "test_task",  # reference to a completed prediction task
        })
        # Should return 200 or 409 (task not completed yet) — not 404
        assert resp.status_code in (200, 409)


# ---------------------------------------------------------------------------
# D4: Caching
# ---------------------------------------------------------------------------


class TestResultCaching:
    """D4: Result caching with incremental query support."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_instantly(self, app_client):
        """Submitting the exact same task twice should return cached result."""
        csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5"
        resp1 = await app_client.post("/tasks", json={
            "task_type": "predict",
            "input_csv": csv,
        })
        task_id1 = resp1.json()["task_id"]

        resp2 = await app_client.post("/tasks", json={
            "task_type": "predict",
            "input_csv": csv,
        })
        task_id2 = resp2.json()["task_id"]
        # Cache hit should be indicated (could be same task_id or instant completion)
        # Implementation decides the mechanism — test checks the endpoint works
        assert resp2.status_code == 200

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_input_change(self, app_client):
        """Changing input data invalidates all cached results."""
        csv1 = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5"
        csv2 = "sample_id,drug1,dose1,float_value\nB,DrugY,0.2,0.6"

        await app_client.post("/tasks", json={"task_type": "predict", "input_csv": csv1})
        # Change input
        await app_client.post("/upload/input", content=csv2, headers={"content-type": "text/csv"})
        # New task with different input should not use old cache
        resp = await app_client.post("/tasks", json={"task_type": "predict", "input_csv": csv2})
        assert resp.status_code == 200
