from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from ..schemas import DrugInfo, DrugMappingResponse
from ..session_store import session_store

router = APIRouter()

@router.get("/drugs/mapping", response_model=DrugMappingResponse)
async def get_drug_mapping() -> DrugMappingResponse:
    from ..main import worker
    if worker is None or worker._drug_library is None:
        raise HTTPException(status_code=503, detail="Drug library not loaded yet.")

    drug_library = worker._drug_library
    all_drug_names: set[str] = set()

    for meta in [session_store.input_meta, session_store.query_meta]:
        if meta is None:
            continue
        mapping = meta.column_mapping
        if mapping is None:
            continue
        df = meta.raw_df
        for slot in ["drug1", "drug2", "drug3"]:
            col = getattr(mapping, slot, None)
            if col and col in df.columns:
                names = df[col].dropna().astype(str).str.strip()
                names = names[names != ""]
                all_drug_names.update(names.unique())
    if all_drug_names:
        await asyncio.to_thread(drug_library.update, list(all_drug_names))
    drugs: list[DrugInfo] = []
    for name in sorted(all_drug_names):
        known = name in drug_library.drug_id_remap
        canonical = drug_library.get_canonical_name(name) if known else None
        drugs.append(DrugInfo(name=name, known=known, canonical_name=canonical))

    return DrugMappingResponse(drugs=drugs)
