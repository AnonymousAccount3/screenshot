"""Preprocessor for drug screening data."""

import pandas as pd
import numpy as np
from typing import Optional, List


class Preprocessor:
    """
    Preprocesses raw drug screening DataFrames for model consumption.

    Performs:
    - Drug name tokenization via drug_library
    - Log-scale and normalization of doses to [0, 1]
    - Viability clipping to [0, 1]
    - Addition of missing drug/dose columns with defaults

    Example:
        ```python
        preprocessor = Preprocessor(drug_library, n_drugs=3)
        df_processed = preprocessor.transform(df_raw)

        # After prediction, recover original drug names
        df_results = preprocessor.inverse_transform_drugs(df_results)
        ```
    """

    def __init__(
        self,
        drug_library,
        n_drugs: int = 3,
        min_dose_val: float = -6.0,
        max_dose_val: float = 4.0,
    ):
        """
        Args:
            drug_library: DrugLibrary for tokenization
            n_drugs: Number of drug columns (drug1, drug2, ..., drugN)
            min_dose_val: Minimum log10 dose value for clipping/normalization
            max_dose_val: Maximum log10 dose value for clipping/normalization
        """
        self.drug_library = drug_library
        self.n_drugs = n_drugs
        self.min_dose_val = min_dose_val
        self.max_dose_val = max_dose_val

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a raw DataFrame.

        Args:
            df: Raw DataFrame with drug names, doses, and optionally viability.
                Expected columns: drug1, dose1, [drug2, dose2, ...], [float_value]

        Returns:
            Preprocessed DataFrame with tokenized drugs and normalized doses.
        """
        df = df.copy()

        drug_columns = [f"drug{i+1}" for i in range(self.n_drugs)]
        dose_columns = [f"dose{i+1}" for i in range(self.n_drugs)]

        # Add missing drug/dose columns with defaults
        for c in drug_columns:
            if c not in df.columns:
                df[c] = None

        for c in dose_columns:
            if c not in df.columns:
                df[c] = np.nan

        # Tokenize drugs
        for col in drug_columns:
            df[col] = self.drug_library.batch_encode(df[col].fillna('').tolist())

        # Process doses
        min_dv = self.min_dose_val
        max_dv = self.max_dose_val
        df[dose_columns] = df[dose_columns].fillna(10**min_dv)
        df[dose_columns] = df[dose_columns].clip(lower=10**min_dv)
        df[dose_columns] = np.log10(df[dose_columns])
        df[dose_columns] = df[dose_columns].clip(lower=min_dv, upper=max_dv)
        df[dose_columns] = (df[dose_columns] - min_dv) / (max_dv - min_dv)

        # Clip viability
        if "float_value" in df.columns:
            df["float_value"] = df["float_value"].clip(lower=0.0, upper=1.0)

        return df

    def inverse_transform_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode tokenized drug IDs back to original drug names.

        Use this to recover human-readable drug names before saving results.

        Args:
            df: DataFrame with tokenized drug columns (integer IDs)

        Returns:
            DataFrame with drug columns decoded to original names.
        """
        df = df.copy()

        drug_columns = [f"drug{i+1}" for i in range(self.n_drugs)]

        for col in drug_columns:
            if col in df.columns:
                df[col] = [
                    self.drug_library.decode(int(x)) if pd.notna(x) and int(x) > 0 else ''
                    for x in df[col]
                ]

        return df
