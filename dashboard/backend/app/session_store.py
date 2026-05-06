from __future__ import annotations

import hashlib
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd

@dataclass
class ColumnMapping:
    """Maps semantic roles to actual column names in the uploaded CSV."""
    sample_id: Optional[str] = None
    drug1: Optional[str] = None
    dose1: Optional[str] = None
    drug2: Optional[str] = None
    dose2: Optional[str] = None
    drug3: Optional[str] = None
    dose3: Optional[str] = None
    viability: Optional[str] = None
    group: Optional[str] = None

    def to_dict(self) -> dict[str, Optional[str]]:
        """Return mapping as dict, omitting None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def to_rename_dict(self) -> dict[str, str]:
        """Return {original_col: standard_name} for renaming."""
        rename = {}
        standard_names = {
            "sample_id": self.sample_id,
            "drug1": self.drug1,
            "dose1": self.dose1,
            "drug2": self.drug2,
            "dose2": self.dose2,
            "drug3": self.drug3,
            "dose3": self.dose3,
            "float_value": self.viability,
        }
        for standard, original in standard_names.items():
            if original is not None and original != standard:
                rename[original] = standard
        return rename

@dataclass
class DatasetMeta:
    """Metadata for an uploaded dataset."""
    raw_df: pd.DataFrame  # Original CSV as-is
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    column_mapping: Optional[ColumnMapping] = None
    degree: int = 1  # Max drugs per row
    has_viability: bool = False
    dose_scale: str = "linear"  # "linear" or "log"
    dose_transform_applied: str = "none"  # "log10" or "none"
    warnings: list[str] = field(default_factory=list)
    data_hash: str = ""  # MD5 of CSV for cache keying

    def standardized_df(self) -> pd.DataFrame:
        """Return DataFrame with columns renamed to standard names."""
        if self.column_mapping is None:
            return self.raw_df.copy()
        rename = self.column_mapping.to_rename_dict()
        df = self.raw_df.rename(columns=rename)
        if self.dose_scale == "log":
            for dose_col in ["dose1", "dose2", "dose3"]:
                if dose_col in df.columns:
                    df[dose_col] = 10.0 ** df[dose_col].astype(float)

        return df

class SessionStore:

    def __init__(self) -> None:
        self.input_meta: Optional[DatasetMeta] = None
        self.query_meta: Optional[DatasetMeta] = None
        self._generated_query_df: Optional[pd.DataFrame] = None
        self.subsample_n: Optional[int] = 500  # N random rows per sample for inference (default cap)
        self.dose_filter: Optional[list[float]] = None  # Selected doses to keep

    @property
    def input_df(self) -> Optional[pd.DataFrame]:
        """Standardized input DataFrame (columns renamed)."""
        if self.input_meta is None:
            return None
        return self.input_meta.standardized_df()

    @property
    def query_df(self) -> Optional[pd.DataFrame]:
        """Standardized query DataFrame. Generated query takes precedence."""
        if self._generated_query_df is not None:
            return self._generated_query_df
        if self.query_meta is None:
            return None
        return self.query_meta.standardized_df()

    def set_generated_query(self, df: pd.DataFrame) -> None:
        """Store a generated query DataFrame."""
        self._generated_query_df = df

    def clear_generated_query(self) -> None:
        self._generated_query_df = None

    def get_subsample_indices(self) -> Optional[list[int]]:
        """Return original row indices selected by the current subsample setting."""
        if self.subsample_n is None or self.input_meta is None:
            return None
        df = self.input_meta.standardized_df()
        if "sample_id" not in df.columns:
            return None
        indices: list[int] = []
        for _, grp in df.groupby("sample_id", sort=False):
            if len(grp) > self.subsample_n:
                indices.extend(grp.sample(n=self.subsample_n, random_state=42).index.tolist())
            else:
                indices.extend(grp.index.tolist())
        return sorted(indices)

    def subsample_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply per-sample random subsampling if subsample_n is set."""
        if self.subsample_n is None or "sample_id" not in df.columns:
            return df
        groups = []
        for _, grp in df.groupby("sample_id", sort=False):
            if len(grp) > self.subsample_n:
                groups.append(grp.sample(n=self.subsample_n, random_state=42))
            else:
                groups.append(grp)
        return pd.concat(groups, ignore_index=True) if groups else df

    def get_dose_filter_indices(self) -> Optional[list[int]]:
        """Return original row indices selected by the current dose filter."""
        if self.dose_filter is None or self.input_meta is None:
            return None
        df = self.input_meta.standardized_df()
        if "dose1" not in df.columns:
            return None
        dose_set = set(self.dose_filter)
        mask = df["dose1"].astype(float).isin(dose_set)
        return sorted(df.index[mask].tolist())

    def dose_filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only rows matching selected doses."""
        if self.dose_filter is None or "dose1" not in df.columns:
            return df
        dose_set = set(self.dose_filter)
        return df[df["dose1"].astype(float).isin(dose_set)].reset_index(drop=True)

    def get_available_doses(self) -> Optional[list[float]]:
        """Return sorted unique dose1 values from the input data."""
        if self.input_meta is None:
            return None
        df = self.input_meta.standardized_df()
        if "dose1" not in df.columns:
            return None
        doses = sorted(df["dose1"].astype(float).dropna().unique().tolist())
        return doses

    def clear(self) -> None:
        """Reset all stored data."""
        self.input_meta = None
        self.query_meta = None
        self._generated_query_df = None
        self.subsample_n = 2000
        self.dose_filter = None

    def compute_hash(self, csv_text: str) -> str:
        return hashlib.md5(csv_text.encode()).hexdigest()
session_store = SessionStore()
