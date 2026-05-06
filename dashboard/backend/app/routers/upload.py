from __future__ import annotations

from io import BytesIO, StringIO

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from ..column_detection import (
    check_viability_range,
    detect_columns,
    detect_degree,
    detect_dose_scale,
)
from ..schemas import ColumnMappingOverride, UploadResponse
from ..session_store import ColumnMapping, DatasetMeta, session_store

router = APIRouter()

def _parse_upload(body: bytes, content_type: str) -> tuple[pd.DataFrame, str]:
    """Parse uploaded bytes into a DataFrame and CSV text."""
    ct = (content_type or "").lower()
    if "application/octet-stream" in ct or "application/x-feather" in ct:
        df = pd.read_feather(BytesIO(body))
        return df, df.to_csv(index=False)
    if "spreadsheet" in ct or "excel" in ct:
        df = pd.read_excel(BytesIO(body))
        return df, df.to_csv(index=False)
    csv_text = body.decode("utf-8")
    df = pd.read_csv(StringIO(csv_text))
    return df, csv_text

@router.post("/upload/input", response_model=UploadResponse)
async def upload_input(request: Request) -> UploadResponse:
    body = await request.body()
    content_type = request.headers.get("content-type", "")

    try:
        df, csv_text = _parse_upload(body, content_type)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid file format: {exc}")

    if df.empty and len(df.columns) == 0:
        raise HTTPException(status_code=422, detail="CSV is empty")

    mapping = detect_columns(df)
    # (drug/dose) but NOT viability, meaning viability is truly missing.
    has_any_detected = any(
        getattr(mapping, attr) is not None
        for attr in ["sample_id", "drug1", "dose1", "drug2", "dose2"]
    )
    if mapping.viability is None and has_any_detected:
        raise HTTPException(
            status_code=422,
            detail="Input CSV must contain a viability column (e.g., float_value, viability). None detected.",
        )

    degree = detect_degree(df, mapping)
    has_viability = mapping.viability is not None
    warnings_list: list[str] = []
    dose_col = mapping.dose1
    dose_scale = "linear"
    dose_transform = "none"
    if dose_col and dose_col in df.columns:
        dose_scale, dose_transform, dose_warnings = detect_dose_scale(df, dose_col)
        warnings_list.extend(dose_warnings)
    if mapping.viability and mapping.viability in df.columns:
        via_warnings = check_viability_range(df, mapping.viability)
        warnings_list.extend(via_warnings)
    data_hash = session_store.compute_hash(csv_text)
    meta = DatasetMeta(
        raw_df=df,
        row_count=len(df),
        columns=list(df.columns),
        column_mapping=mapping,
        degree=degree,
        has_viability=has_viability,
        dose_scale=dose_scale,
        dose_transform_applied=dose_transform,
        warnings=warnings_list,
        data_hash=data_hash,
    )
    session_store.input_meta = meta
    session_store.clear_generated_query()

    sample_ids = []
    if mapping.sample_id and mapping.sample_id in df.columns:
        sample_ids = sorted(df[mapping.sample_id].dropna().unique().astype(str).tolist())

    return UploadResponse(
        row_count=meta.row_count,
        columns=meta.columns,
        column_mapping=mapping.to_dict(),
        degree=degree,
        has_viability=has_viability,
        dose_scale=dose_scale,
        dose_transform_applied=dose_transform,
        warnings=warnings_list,
        sample_ids=sample_ids,
        csv_text=csv_text,
    )

@router.post("/upload/input/mapping", response_model=UploadResponse)
async def override_input_mapping(override: ColumnMappingOverride) -> UploadResponse:
    if session_store.input_meta is None:
        raise HTTPException(status_code=400, detail="No input dataset uploaded yet.")

    new_mapping = ColumnMapping(
        sample_id=override.sample_id,
        drug1=override.drug1,
        dose1=override.dose1,
        drug2=override.drug2,
        dose2=override.dose2,
        drug3=override.drug3,
        dose3=override.dose3,
        viability=override.viability,
        group=override.group,
    )
    session_store.input_meta.column_mapping = new_mapping
    meta = session_store.input_meta

    return UploadResponse(
        row_count=meta.row_count,
        columns=meta.columns,
        column_mapping=new_mapping.to_dict(),
        degree=meta.degree,
        has_viability=meta.has_viability,
        dose_scale=meta.dose_scale,
        dose_transform_applied=meta.dose_transform_applied,
        warnings=meta.warnings,
    )
