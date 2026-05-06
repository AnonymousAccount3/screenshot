from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..column_detection import detect_columns, detect_degree
from ..schemas import CohortInfo, CohortListResponse, UploadResponse
from ..session_store import DatasetMeta, session_store

router = APIRouter()
COHORTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "cohorts"

def _load_cohort_index() -> list[CohortInfo]:
    """Load cohort metadata from the index JSON file."""
    index_path = COHORTS_DIR / "index.json"
    if not index_path.exists():
        return []

    with open(index_path) as f:
        raw = json.load(f)

    cohorts = []
    for entry in raw:
        cohorts.append(CohortInfo(**entry))
    return cohorts

@router.get("/cohorts", response_model=CohortListResponse)
async def list_cohorts() -> CohortListResponse:
    """List available pre-built cohorts with metadata."""
    cohorts = _load_cohort_index()
    return CohortListResponse(cohorts=cohorts)

@router.post("/cohorts/{cohort_name}/load", response_model=UploadResponse)
async def load_cohort(cohort_name: str) -> UploadResponse:
    csv_path = COHORTS_DIR / f"{cohort_name}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Cohort '{cohort_name}' not found.")

    df = pd.read_csv(csv_path)
    mapping = detect_columns(df)
    degree = detect_degree(df, mapping)
    has_viability = mapping.viability is not None

    csv_text = csv_path.read_text()
    data_hash = session_store.compute_hash(csv_text)

    meta = DatasetMeta(
        raw_df=df,
        row_count=len(df),
        columns=list(df.columns),
        column_mapping=mapping,
        degree=degree,
        has_viability=has_viability,
        dose_scale="linear",
        dose_transform_applied="log10",
        warnings=[],
        data_hash=data_hash,
    )
    session_store.input_meta = meta
    session_store.clear_generated_query()

    return UploadResponse(
        row_count=meta.row_count,
        columns=meta.columns,
        column_mapping=mapping.to_dict(),
        degree=degree,
        has_viability=has_viability,
        dose_scale="linear",
        dose_transform_applied="log10",
        warnings=[],
    )
