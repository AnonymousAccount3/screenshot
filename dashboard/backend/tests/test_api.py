"""Tests for the REST API endpoints.

These tests use a mock inference worker — no model required.
"""

from __future__ import annotations

import pytest
from app.schemas import TaskType


@pytest.mark.asyncio
async def test_health(app_client):
    resp = await app_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_create_task(app_client):
    resp = await app_client.post("/tasks", json={
        "task_type": "predict",
        "input_csv": "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["task_type"] == "predict"
    assert data["status"] in ("pending", "running")


@pytest.mark.asyncio
async def test_get_task(app_client):
    # Submit a task first
    resp = await app_client.post("/tasks", json={
        "task_type": "predict",
        "input_csv": "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5",
    })
    task_id = resp.json()["task_id"]

    # Fetch it
    resp = await app_client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["task_id"] == task_id


@pytest.mark.asyncio
async def test_get_task_not_found(app_client):
    resp = await app_client.get("/tasks/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_task(app_client):
    resp = await app_client.post("/tasks", json={
        "task_type": "predict_uncertainty",
        "input_csv": "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5",
    })
    task_id = resp.json()["task_id"]

    resp = await app_client.post(f"/tasks/{task_id}/cancel")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_cancel_nonexistent_task(app_client):
    resp = await app_client.post("/tasks/nonexistent/cancel")
    assert resp.status_code == 404
