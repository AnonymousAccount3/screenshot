"""Tests for Silo 1: Drug library integration (A3).

Spec source: FEATURES.md section A3.
These tests are READ-ONLY during implementation.
"""

import pytest


class TestDrugLibraryIntegration:
    """A3: Drug library update and mapping display."""

    @pytest.mark.asyncio
    async def test_drug_mapping_returns_all_drugs(self, app_client):
        """After upload, drug mapping endpoint returns all unique drugs."""
        csv = "sample_id,drug1,dose1,float_value\nA,Doxorubicin,0.1,0.5\nA,Venetoclax,1.0,0.3"
        await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        resp = await app_client.get("/drugs/mapping")
        assert resp.status_code == 200
        drugs = resp.json()["drugs"]
        drug_names = [d["name"] for d in drugs]
        assert "Doxorubicin" in drug_names or any("doxorubicin" in n.lower() for n in drug_names)

    @pytest.mark.asyncio
    async def test_drug_mapping_shows_known_vs_new(self, app_client):
        """Each drug in the mapping has a 'known' boolean field."""
        csv = "sample_id,drug1,dose1,float_value\nA,Doxorubicin,0.1,0.5"
        await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        resp = await app_client.get("/drugs/mapping")
        drugs = resp.json()["drugs"]
        for drug in drugs:
            assert "known" in drug
            assert isinstance(drug["known"], bool)

    @pytest.mark.asyncio
    async def test_drug_mapping_shows_canonical_name(self, app_client):
        """Each drug has a canonical_name field (may be null for unknown drugs)."""
        csv = "sample_id,drug1,dose1,float_value\nA,Doxorubicin,0.1,0.5"
        await app_client.post("/upload/input", content=csv, headers={"content-type": "text/csv"})
        resp = await app_client.get("/drugs/mapping")
        drugs = resp.json()["drugs"]
        for drug in drugs:
            assert "canonical_name" in drug

    @pytest.mark.asyncio
    async def test_drug_mapping_includes_query_drugs(self, app_client):
        """Drugs from both input AND query datasets appear in the mapping."""
        input_csv = "sample_id,drug1,dose1,float_value\nA,Doxorubicin,0.1,0.5"
        query_csv = "sample_id,drug1,dose1\nA,Venetoclax,0.1"
        await app_client.post("/upload/input", content=input_csv, headers={"content-type": "text/csv"})
        await app_client.post("/upload/query", content=query_csv, headers={"content-type": "text/csv"})
        resp = await app_client.get("/drugs/mapping")
        drug_names = [d["name"].lower() for d in resp.json()["drugs"]]
        assert any("venetoclax" in n for n in drug_names)
