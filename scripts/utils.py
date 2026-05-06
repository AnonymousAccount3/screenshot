import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

def get_dataset_paths(
    data_dir: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Get filtered .feather file paths from data directory.

    Args:
        data_dir: Directory containing .feather files
        include: If provided, only include these datasets (filenames without extension)
        exclude: If provided, exclude these datasets (filenames without extension)

    Returns:
        List of paths to .feather files
    """
    data_path = Path(data_dir)
    feather_files = list(data_path.glob('*.feather'))

    if not feather_files:
        raise ValueError(f"No .feather files found in {data_dir}")

    # Filter based on include/exclude
    filtered_files = []

    for file_path in feather_files:
        filename = file_path.stem  # filename without extension

        # If include is specified, only include those datasets
        if include is not None:
            if filename not in include:
                continue

        # If exclude is specified, skip those datasets
        if exclude is not None:
            if filename in exclude:
                continue

        filtered_files.append(file_path)

    if not filtered_files:
        raise ValueError(
            f"No datasets found after filtering. "
            f"Include: {include}, Exclude: {exclude}"
        )

    # Sort for consistency
    filtered_files = sorted(filtered_files)

    print(f"\nFound {len(filtered_files)} datasets after filtering:")
    for f in filtered_files:
        print(f"  - {f.stem}")

    return [str(f) for f in filtered_files]


def create_morgan_fingerprint_cache(
    df: pd.DataFrame,
    drug_library,
    n_drugs: int = 3,
    radius: int = 2,
    n_bits: int = 512
):
    """
    Create a cache of Morgan fingerprints for all unique drugs in the dataframe.

    Uses drug_library.get_morgan_fingerprints() for SMILES lookup and
    fingerprint generation (including drug alias resolution).

    Args:
        df: Input dataframe with drug name columns (drug1, drug2, ...)
        drug_library: DrugLibrary instance
        n_drugs: Number of drugs
        radius: Morgan fingerprint radius
        n_bits: Number of bits in fingerprint

    Returns:
        Dictionary mapping drug name to Morgan fingerprint array
    """
    # Collect all unique drug names
    unique_drugs = set()
    for i in range(n_drugs):
        drug_col = f'drug{i+1}'
        if drug_col in df.columns:
            unique_drugs.update(df[drug_col].dropna().unique())

    drug_list = sorted(unique_drugs)

    # Generate fingerprints via drug library
    fingerprints = drug_library.get_morgan_fingerprints(
        drug_list, radius=radius, n_bits=n_bits
    )

    # Build cache, replacing NaN rows with zero vectors
    zero_fp = np.zeros(n_bits)
    fingerprint_cache = {}
    n_missing = 0
    for drug_name, fp in zip(drug_list, fingerprints):
        if np.any(np.isnan(fp)):
            fingerprint_cache[drug_name] = zero_fp
            n_missing += 1
        else:
            fingerprint_cache[drug_name] = fp

    # Sentinel entries for missing/null drug names
    fingerprint_cache[None] = zero_fp
    fingerprint_cache[np.nan] = zero_fp

    return fingerprint_cache


def prepare_features_with_morgan(
    df: pd.DataFrame,
    fingerprint_cache: dict,
    n_drugs: int = 3
):
    """
    Prepare feature matrix from dataframe using cached Morgan fingerprints.

    Args:
        df: Input dataframe (with preprocessed doses)
        fingerprint_cache: Dictionary mapping drug name to fingerprint array
        n_drugs: Number of drugs

    Returns:
        X: Feature matrix
        y: Target values
    """
    features_list = []

    for i in range(n_drugs):
        drug_col = f'drug{i+1}'
        dose_col = f'dose{i+1}'

        drug_names = df[drug_col].values

        fingerprints = []
        for drug_name in drug_names:
            if drug_name in fingerprint_cache:
                fingerprints.append(fingerprint_cache[drug_name])
            else:
                fingerprints.append(fingerprint_cache[None])

        fingerprints = np.array(fingerprints)

        doses = df[dose_col].values.reshape(-1, 1)

        drug_features = np.concatenate([fingerprints, doses], axis=1)
        features_list.append(drug_features)

    X = np.concatenate(features_list, axis=1)
    y = df['float_value'].values

    return X, y