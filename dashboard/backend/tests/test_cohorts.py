"""Tests for Silo 1: Pre-built cohort inputs (A5).

Spec source: FEATURES.md section A5.
These tests are READ-ONLY during implementation.
"""

import pytest


class TestCohorts:
    """A5: Pre-built cohort inputs."""

    @pytest.mark.asyncio
    async def test_list_cohorts(self, app_client):
        """GET /cohorts returns available pre-built cohorts."""
        resp = await app_client.get("/cohorts")
        assert resp.status_code == 200
        data = resp.json()
        assert "cohorts" in data
        assert isinstance(data["cohorts"], list)

    @pytest.mark.asyncio
    async def test_cohort_has_metadata(self, app_client):
        """Each cohort has name, description, n_samples, n_drugs."""
        resp = await app_client.get("/cohorts")
        cohorts = resp.json()["cohorts"]
        if len(cohorts) > 0:
            cohort = cohorts[0]
            assert "name" in cohort
            assert "description" in cohort
            assert "n_samples" in cohort
            assert "n_drugs" in cohort

    @pytest.mark.asyncio
    async def test_cohort_max_20_samples(self, app_client):
        """No cohort has more than 20 samples."""
        resp = await app_client.get("/cohorts")
        cohorts = resp.json()["cohorts"]
        for cohort in cohorts:
            assert cohort["n_samples"] <= 20

    @pytest.mark.asyncio
    async def test_load_cohort_as_input(self, app_client):
        """Loading a cohort sets it as the current input dataset."""
        resp = await app_client.get("/cohorts")
        cohorts = resp.json()["cohorts"]
        if len(cohorts) > 0:
            cohort_name = cohorts[0]["name"]
            load_resp = await app_client.post(f"/cohorts/{cohort_name}/load")
            assert load_resp.status_code == 200
            data = load_resp.json()
            assert data["row_count"] > 0
            assert "column_mapping" in data
