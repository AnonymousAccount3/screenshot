"""Tests for Silo 1: Data upload and column detection (A1, A2).

Spec source: FEATURES.md sections A1, A2.
These tests are READ-ONLY during implementation.
"""

import pytest


# ---------------------------------------------------------------------------
# A1: File upload
# ---------------------------------------------------------------------------


class TestFileUpload:
    """A1: Drag-and-drop CSV upload."""

    @pytest.mark.asyncio
    async def test_upload_input_csv(self, app_client):
        """Uploading a valid input CSV returns success with row count and columns."""
        csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5\nA,DrugX,1.0,0.3"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["row_count"] == 2
        assert "sample_id" in data["columns"]
        assert "drug1" in data["columns"]

    @pytest.mark.asyncio
    async def test_upload_input_missing_viability_rejected(self, app_client):
        """Input CSV without viability column is rejected with clear error."""
        csv = "sample_id,drug1,dose1\nA,DrugX,0.1"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        assert resp.status_code == 422
        assert "viability" in resp.json()["detail"].lower() or "float_value" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_upload_query_csv_without_viability_accepted(self, app_client):
        """Query CSV without viability column is accepted (viability is optional for queries)."""
        csv = "sample_id,drug1,dose1\nA,DrugX,0.1"
        resp = await app_client.post("/upload/query", content=csv, headers={"content-type": "text/csv"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_upload_query_csv_with_viability_accepted(self, app_client):
        """Query CSV with viability column is also accepted."""
        csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.8"
        resp = await app_client.post("/upload/query", content=csv, headers={"content-type": "text/csv"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_viability"] is True

    @pytest.mark.asyncio
    async def test_upload_replaces_previous(self, app_client):
        """Re-uploading replaces the previous dataset."""
        csv1 = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5"
        csv2 = "sample_id,drug1,dose1,float_value\nB,DrugY,0.2,0.6\nB,DrugY,1.0,0.3"
        await app_client.post("/upload/input", content=csv1, headers={"content-type": "text/csv"})
        resp = await app_client.post("/upload/input", content=csv2, headers={"content-type": "text/csv"})
        assert resp.status_code == 200
        assert resp.json()["row_count"] == 2


# ---------------------------------------------------------------------------
# A2: Column auto-detection
# ---------------------------------------------------------------------------


class TestColumnDetection:
    """A2: Column auto-detection & mapping."""

    @pytest.mark.asyncio
    async def test_detect_standard_columns(self, app_client):
        """Standard column names are auto-detected correctly."""
        csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,0.5"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        mapping = resp.json()["column_mapping"]
        assert mapping["sample_id"] == "sample_id"
        assert mapping["drug1"] == "drug1"
        assert mapping["dose1"] == "dose1"
        assert mapping["viability"] == "float_value"

    @pytest.mark.asyncio
    async def test_detect_nonstandard_column_names(self, app_client):
        """Non-standard column names should still be detected by content heuristics."""
        csv = "cell_line,compound,concentration,viability\nA,DrugX,0.1,0.5"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        mapping = resp.json()["column_mapping"]
        # Should detect that 'cell_line' maps to sample_id, 'compound' to drug1, etc.
        assert mapping["sample_id"] == "cell_line"
        assert mapping["drug1"] == "compound"
        assert mapping["dose1"] == "concentration"
        assert mapping["viability"] == "viability"

    @pytest.mark.asyncio
    async def test_detect_multi_drug_columns(self, app_client):
        """Multi-drug combination columns are detected."""
        csv = "sample_id,drug1,dose1,drug2,dose2,float_value\nA,DrugX,0.1,DrugY,0.2,0.5"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        mapping = resp.json()["column_mapping"]
        assert mapping["drug2"] == "drug2"
        assert mapping["dose2"] == "dose2"
        assert resp.json()["degree"] == 2

    @pytest.mark.asyncio
    async def test_detect_group_column(self, app_client):
        """Columns with 'group' in the name are detected as group labels."""
        csv = "sample_id,drug1,dose1,float_value,treatment_group\nA,DrugX,0.1,0.5,arm1"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        mapping = resp.json()["column_mapping"]
        assert mapping.get("group") == "treatment_group"

    @pytest.mark.asyncio
    async def test_override_column_mapping(self, app_client):
        """User can override auto-detected column mapping."""
        csv = "col_a,col_b,col_c,col_d\nA,DrugX,0.1,0.5"
        await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        resp = await app_client.post("/upload/input/mapping", json={
            "sample_id": "col_a",
            "drug1": "col_b",
            "dose1": "col_c",
            "viability": "col_d",
        })
        assert resp.status_code == 200
        assert resp.json()["column_mapping"]["sample_id"] == "col_a"

    @pytest.mark.asyncio
    async def test_dose_linear_detected_and_logged(self, app_client):
        """Linear doses (all positive) are detected and log10 is applied."""
        csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.001,0.9\nA,DrugX,0.01,0.8\nA,DrugX,0.1,0.5\nA,DrugX,1.0,0.3\nA,DrugX,10.0,0.1"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        data = resp.json()
        assert data["dose_scale"] == "linear"
        assert data["dose_transform_applied"] == "log10"

    @pytest.mark.asyncio
    async def test_dose_log_detected_not_double_logged(self, app_client):
        """Already log-scaled doses (negative values) are NOT double-logged."""
        csv = "sample_id,drug1,dose1,float_value\nA,DrugX,-3,0.9\nA,DrugX,-2,0.8\nA,DrugX,-1,0.5\nA,DrugX,0,0.3\nA,DrugX,1,0.1"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        data = resp.json()
        assert data["dose_scale"] == "log"
        assert data["dose_transform_applied"] == "none"

    @pytest.mark.asyncio
    async def test_dose_clipping_warning(self, app_client):
        """Doses outside model range [-6, 4] trigger a warning."""
        csv = "sample_id,drug1,dose1,float_value\nA,DrugX,-10,0.9\nA,DrugX,0,0.5"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        data = resp.json()
        assert any("clip" in w.lower() for w in data.get("warnings", []))

    @pytest.mark.asyncio
    async def test_viability_out_of_range_warning(self, app_client):
        """Viability values outside [0, 1] trigger a warning."""
        csv = "sample_id,drug1,dose1,float_value\nA,DrugX,0.1,1.5\nA,DrugX,1.0,-0.2"
        resp = await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        data = resp.json()
        assert any("viability" in w.lower() or "range" in w.lower() for w in data.get("warnings", []))
