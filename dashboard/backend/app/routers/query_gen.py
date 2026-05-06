from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ..schemas import QueryGenerateRequest, QueryGenerateResponse
from ..session_store import session_store

router = APIRouter()

@router.post("/query/generate", response_model=QueryGenerateResponse)
async def generate_query(req: QueryGenerateRequest) -> QueryGenerateResponse:
    if session_store.input_meta is None:
        raise HTTPException(status_code=400, detail="No input dataset uploaded. Upload input first.")

    meta = session_store.input_meta
    mapping = meta.column_mapping
    df = meta.raw_df
    degree = req.degree if req.degree is not None else meta.degree
    n_points = req.n_points or 10
    dose_min = req.dose_min if req.dose_min is not None else -6.0  # log10 scale
    dose_max = req.dose_max if req.dose_max is not None else 4.0   # log10 scale
    sample_col = mapping.sample_id
    if sample_col is None or sample_col not in df.columns:
        raise HTTPException(status_code=400, detail="Cannot determine sample_id column.")
    samples = df[sample_col].unique().tolist()
    query_df = _generate_dose_grid(
        df, mapping, samples, n_points, dose_min, dose_max
    )
    degree = 1
    session_store.set_generated_query(query_df)

    preview = query_df.head(20).to_dict(orient="records")

    return QueryGenerateResponse(
        row_count=len(query_df),
        degree=degree,
        preview=preview,
    )

def _generate_dose_grid(
    df: pd.DataFrame,
    mapping,
    samples: list,
    n_points: int,
    dose_min: float,
    dose_max: float,
) -> pd.DataFrame:
    all_drugs: set[str] = set()
    for attr in ["drug1", "drug2", "drug3"]:
        col = getattr(mapping, attr, None)
        if col and col in df.columns:
            all_drugs.update(df[col].dropna().unique())
    drugs = sorted(all_drugs)

    log_doses = np.linspace(dose_min, dose_max, n_points)
    doses = 10.0 ** log_doses

    rows = []
    for sample in samples:
        for drug in drugs:
            for dose in doses:
                rows.append({
                    "sample_id": sample,
                    "drug1": drug,
                    "dose1": round(float(dose), 10),
                })

    return pd.DataFrame(rows)

def _generate_random_combinations(
    df: pd.DataFrame,
    mapping,
    samples: list,
    degree: int,
    n_points: int,
    dose_min: float,
    dose_max: float,
) -> pd.DataFrame:
    all_drugs: set[str] = set()
    for i in range(1, degree + 1):
        col = getattr(mapping, f"drug{i}", None)
        if col and col in df.columns:
            all_drugs.update(df[col].dropna().unique())
    drugs_list = sorted(all_drugs)

    rng = np.random.default_rng(42)
    rows = []
    per_sample = max(1, n_points // len(samples)) if samples else n_points

    for sample in samples:
        for _ in range(per_sample):
            row = {"sample_id": sample}
            chosen_drugs = rng.choice(drugs_list, size=degree, replace=True)
            for j in range(degree):
                log_dose = rng.uniform(dose_min, dose_max)
                row[f"drug{j+1}"] = chosen_drugs[j]
                row[f"dose{j+1}"] = round(float(10.0 ** log_dose), 10)
            rows.append(row)
    rows = rows[:n_points]
    return pd.DataFrame(rows)
