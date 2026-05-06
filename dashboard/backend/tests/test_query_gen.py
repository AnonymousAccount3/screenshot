"""Tests for Silo 1: Query generation (A4).

Spec source: FEATURES.md section A4.
These tests are READ-ONLY during implementation.
"""

import pytest


class TestQueryGeneration:
    """A4: Query generation when no query dataset is provided."""

    @pytest.mark.asyncio
    async def test_generate_dose_grid_single_agent(self, app_client):
        """Generate dose-response grid for single agent (degree=1)."""
        # Upload input first
        input_csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5\nA,DrugY,1.0,0.3"
        await app_client.post("/upload/input", content=input_csv, headers={"content-type": "text/csv"})

        resp = await app_client.post("/query/generate", json={
            "degree": 1,
            "n_points": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        # Should have rows for each drug × each sample × n_points
        assert data["row_count"] > 0
        assert "preview" in data  # First ~20 rows

    @pytest.mark.asyncio
    async def test_generate_grid_covers_all_input_drugs(self, app_client):
        """Generated grid includes all unique drugs from input."""
        input_csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5\nA,DrugY,1.0,0.3\nA,DrugZ,0.5,0.4"
        await app_client.post("/upload/input", content=input_csv, headers={"content-type": "text/csv"})

        resp = await app_client.post("/query/generate", json={"degree": 1, "n_points": 5})
        data = resp.json()
        drugs_in_query = set(row["drug1"] for row in data["preview"])
        assert "DrugX" in drugs_in_query
        assert "DrugY" in drugs_in_query
        assert "DrugZ" in drugs_in_query

    @pytest.mark.asyncio
    async def test_generate_grid_log_spaced_doses(self, app_client):
        """Generated grid has equally-spaced doses in log10 scale."""
        input_csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5"
        await app_client.post("/upload/input", content=input_csv, headers={"content-type": "text/csv"})

        resp = await app_client.post("/query/generate", json={"degree": 1, "n_points": 5})
        data = resp.json()
        # Extract doses for one drug — they should be log-spaced
        doses = sorted(set(row["dose1"] for row in data["preview"] if row.get("drug1") == "DrugX"))
        if len(doses) >= 3:
            import math
            log_doses = [math.log10(d) if d > 0 else d for d in doses]
            spacings = [log_doses[i + 1] - log_doses[i] for i in range(len(log_doses) - 1)]
            # All spacings should be approximately equal
            assert max(spacings) - min(spacings) < 0.01

    @pytest.mark.asyncio
    async def test_generate_random_combinations(self, app_client):
        """Generate random combinations for degree >= 2."""
        input_csv = "sample_id,drug1,dose1,drug2,dose2,float_value\nA,DrugX,0.1,DrugY,0.2,0.5"
        await app_client.post("/upload/input", content=input_csv, headers={"content-type": "text/csv"})

        resp = await app_client.post("/query/generate", json={
            "degree": 2,
            "n_points": 50,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["row_count"] == 50

    @pytest.mark.asyncio
    async def test_degree_auto_inferred(self, app_client):
        """Degree is auto-inferred from max drugs per row in input."""
        input_csv = "sample_id,drug1,dose1,drug2,dose2,float_value\nA,DrugX,0.1,DrugY,0.2,0.5"
        await app_client.post("/upload/input", content=input_csv, headers={"content-type": "text/csv"})

        resp = await app_client.post("/query/generate", json={"n_points": 10})
        data = resp.json()
        assert data["degree"] == 2

    @pytest.mark.asyncio
    async def test_generate_covers_all_samples(self, app_client):
        """Generated query includes rows for all samples in input."""
        input_csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5\nB,DrugX,0.1,0.6"
        await app_client.post("/upload/input", content=input_csv, headers={"content-type": "text/csv"})

        resp = await app_client.post("/query/generate", json={"degree": 1, "n_points": 5})
        data = resp.json()
        samples = set(row["sample_id"] for row in data["preview"])
        assert "A" in samples
        assert "B" in samples
