from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from .session_store import ColumnMapping
SAMPLE_ID_PATTERNS = [
    r"^sample[_\s]?id$", r"^cell[_\s]?line$", r"^sample$",
    r"^cell[_\s]?name$", r"^model[_\s]?id$", r"^patient[_\s]?id$",
    r"^cosmic[_\s]?id$", r"^barcode$",
]

DRUG_PATTERNS = [
    r"^drug\d*$", r"^compound\d*$", r"^treatment\d*$", r"^agent\d*$",
    r"^drug[_\s]?name\d*$",
]

DOSE_PATTERNS = [
    r"^dose\d*$", r"^concentration\d*$", r"^conc\d*$",
    r"^dose[_\s]?um\d*$", r"^log[_\s]?dose\d*$",
]

VIABILITY_PATTERNS = [
    r"^float[_\s]?value$", r"^viability$", r"^response$",
    r"^percent[_\s]?viability$", r"^pct[_\s]?viability$",
    r"^growth[_\s]?rate$", r"^survival$",
]

GROUP_PATTERNS = [
    r".*group.*", r".*subtype.*", r".*arm$", r".*cohort.*",
    r".*treatment[_\s]?arm.*",
]

def _match_column(col: str, patterns: list[str]) -> bool:
    """Check if column name matches any regex pattern (case-insensitive)."""
    col_lower = col.lower().strip()
    return any(re.match(p, col_lower) for p in patterns)

def detect_columns(df: pd.DataFrame) -> ColumnMapping:
    mapping = ColumnMapping()
    columns = list(df.columns)
    used: set[str] = set()

    # --- Phase 1: Exact standard names ---
    exact_map = {
        "sample_id": "sample_id",
        "drug1": "drug1",
        "dose1": "dose1",
        "drug2": "drug2",
        "dose2": "dose2",
        "drug3": "drug3",
        "dose3": "dose3",
        "float_value": "viability",
    }
    for col_name, role_attr in exact_map.items():
        if col_name in columns:
            setattr(mapping, role_attr, col_name)
            used.add(col_name)

    # --- Phase 2: Heuristic matching for unfilled roles ---
    remaining = [c for c in columns if c not in used]

    # sample_id
    if mapping.sample_id is None:
        for col in remaining:
            if _match_column(col, SAMPLE_ID_PATTERNS):
                mapping.sample_id = col
                used.add(col)
                break

    # drugs (numbered)
    drug_slots = ["drug1", "drug2", "drug3"]
    for slot in drug_slots:
        if getattr(mapping, slot) is None:
            for col in [c for c in columns if c not in used]:
                if _match_column(col, DRUG_PATTERNS):
                    setattr(mapping, slot, col)
                    used.add(col)
                    break

    # doses (numbered)
    dose_slots = ["dose1", "dose2", "dose3"]
    for slot in dose_slots:
        if getattr(mapping, slot) is None:
            for col in [c for c in columns if c not in used]:
                if _match_column(col, DOSE_PATTERNS):
                    setattr(mapping, slot, col)
                    used.add(col)
                    break

    # viability
    if mapping.viability is None:
        for col in [c for c in columns if c not in used]:
            if _match_column(col, VIABILITY_PATTERNS):
                mapping.viability = col
                used.add(col)
                break

    # group
    for col in [c for c in columns if c not in used]:
        if _match_column(col, GROUP_PATTERNS):
            mapping.group = col
            used.add(col)
            break

    return mapping

def detect_degree(df: pd.DataFrame, mapping: ColumnMapping) -> int:
    degree = 0
    for slot in ["drug1", "drug2", "drug3"]:
        col = getattr(mapping, slot, None)
        if col is not None and col in df.columns:
            non_null = df[col].dropna()
            non_empty = non_null[non_null.astype(str).str.strip() != ""]
            if len(non_empty) > 0:
                degree += 1
    return max(degree, 1)

def detect_dose_scale(
    df: pd.DataFrame,
    dose_col: str,
) -> tuple[str, str, list[str]]:
    warnings_list: list[str] = []
    doses = df[dose_col].dropna().astype(float)

    if len(doses) == 0:
        return "linear", "none", ["No dose values found"]

    has_negative = (doses < 0).any()

    if has_negative:
        scale = "log"
        transform = "none"
        out_of_range = ((doses < -6) | (doses > 4)).sum()
        if out_of_range > 0:
            warnings_list.append(
                f"{out_of_range} dose values outside model range [-6, 4] (log10 scale) and will be clipped."
            )
    else:
        scale = "linear"
        transform = "log10"
        positive_doses = doses[doses > 0]
        if len(positive_doses) > 0:
            log_doses = np.log10(positive_doses)
            out_of_range = ((log_doses < -6) | (log_doses > 4)).sum()
            if out_of_range > 0:
                warnings_list.append(
                    f"{out_of_range} dose values outside [{chr(0x03BC)}M] range [10⁻⁶, 10⁴] and will be clipped."
                )

    return scale, transform, warnings_list

def check_viability_range(
    df: pd.DataFrame,
    viability_col: str,
) -> list[str]:
    warnings_list: list[str] = []
    vals = df[viability_col].dropna().astype(float)

    n_above = (vals > 1.0).sum()
    n_below = (vals < 0.0).sum()

    if n_above > 0 or n_below > 0:
        warnings_list.append(
            f"{n_above + n_below} viability values outside [0, 1] range "
            f"({n_above} above 1.0, {n_below} below 0.0) will be clipped."
        )

    return warnings_list
