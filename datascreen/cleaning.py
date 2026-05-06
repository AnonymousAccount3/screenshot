import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm


def fit_isotonic(
    df: pd.DataFrame,
    result_col: str,
) -> pd.DataFrame:
    """
    Fits isotonic regression on dose1 vs float_value for each group of drugs and samples.

    Parameters:
    - df: Input DataFrame with dose and drug columns.
    - result_col: Name for the column to store isotonic predictions.

    Returns:
    - DataFrame with an additional column containing isotonic regression predictions.
    """
    drug_columns = []
    dose_columns = []
    j = 2
    while f"drug{j}" in df.columns:
        drug_columns.append(f"drug{j}")
        dose_columns.append(f"dose{j}")
        j += 1

    group_cols = ['sample_id', 'drug1']
    group_cols += drug_columns
    group_cols += dose_columns

    grouped = df.groupby(group_cols, dropna=False)
    results = {"id": [], result_col: []}

    for _, group in tqdm(grouped, desc="Fitting isotonic regression", unit="group"):
        try:
            if len(group) > 1:
                x = group.dose1.values
                y = group.float_value.values

                ir = IsotonicRegression(out_of_bounds="clip", increasing=False)
                yp = ir.fit_transform(x, y)

                results["id"].extend(group.index.values)
                results[result_col].extend(yp)
            else:
                results["id"].extend(group.index.values)
                results[result_col].extend(group.float_value.values)
        except Exception as e:
            print(e)
            results["id"].extend(group.index.values)
            results[result_col].extend(np.full((group.shape[0],), np.nan))

    iso_df = pd.DataFrame(results).set_index("id")
    df[result_col] = iso_df.loc[df.index.values, result_col].values
    return df


def remove_noisy_data(
    df: pd.DataFrame,
    result_col: str = "isotonic_pred",
    threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Cleans the dataset by removing groups with high average absolute isotonic regression error.

    The grouping is based on drug and dose columns.

    Parameters:
    - df: DataFrame to clean.
    - result_col: Column name with isotonic regression predictions.
    - threshold: Maximum allowed average absolute error per group.

    Returns:
    - Filtered DataFrame with only groups below the error threshold.
    """
    drug_columns = []
    dose_columns = []
    j = 2
    while f"drug{j}" in df.columns:
        drug_columns.append(f"drug{j}")
        dose_columns.append(f"dose{j}")
        j += 1

    group_cols = ['sample_id', 'drug1']
    group_cols += drug_columns
    group_cols += dose_columns

    df["diff"] = (df["float_value"] - df[result_col]).abs()
    df["avg_diff"] = df.groupby(group_cols, dropna=False)['diff'].transform('mean')

    df = df.loc[df.avg_diff < threshold]
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['diff', 'avg_diff'], inplace=True)
    return df


def preprocess_data(
    input_dir,
    output_dir,
    exclude=[],
    denoising=True,
    noise_threshold=0.2,
    column_mapping=None,
):
    """
    Clean drug response data from multiple studies (isotonic regression only).
    Does NOT tokenize drugs - that happens in Dataset classes.

    Parameters
    ----------
    input_dir : str or Path
        Can be either:
        - A directory containing .feather files (searched recursively)
        - A single .feather file path
    output_dir : str or Path
        Directory where cleaned study data will be saved
    exclude : list
        List of study IDs to exclude from processing
    denoising : bool
        Whether to apply isotonic regression denoising (default: True)
    noise_threshold : float
        Threshold for removing noisy data groups (default: 0.2)
    column_mapping : dict, optional
        Dictionary with column name mappings. Keys: 'sample', 'drug', 'dose', 'value'
        Example: {'sample': 'cell_line', 'drug': ['compound1'], 'dose': ['conc1'], 'value': 'viability'}
    """
    input_path = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default column mapping if not provided
    if column_mapping is None:
        column_mapping = {}

    # Determine if input is a file or directory
    if input_path.is_file():
        # Single file input
        if input_path.suffix != '.feather':
            raise ValueError(f"Input file must be a .feather file, got: {input_path.suffix}")
        study_paths = [input_path]
        print(f"Processing single file: {input_path}")
    elif input_path.is_dir():
        # Directory input - find all .feather files recursively
        study_paths = sorted(
            p
            for p in input_path.rglob("*.feather")
            if p.is_file()
        )
        print(f"Found {len(study_paths)} study files to process in directory")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    if len(study_paths) == 0:
        print("Warning: No .feather files found to process")
        return

    # Process each study
    for study_path in tqdm(study_paths, desc="Processing studies"):
        study_id = study_path.stem  # Use .stem to remove .feather extension

        # Skip excluded studies
        if study_id in exclude:
            print(f"Skipping excluded study: {study_id}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing study: {study_id}")
        print(f"{'='*60}")

        try:
            # Create study output directory
            study_output_dir = output_dir
            study_output_dir.mkdir(parents=True, exist_ok=True)

            # Load data
            print(f"Loading data from {study_path}")
            df = pd.read_feather(study_path)
            print(f"Loaded {len(df)} viability measurements")

            if len(df) == 0:
                print(f"Warning: No data found for study {study_id}, skipping")
                continue

            # Apply column name mapping
            if column_mapping:
                print("Applying column name mapping...")
                df = column_name_mapping(df, **column_mapping)

            # Apply denoising if requested
            if denoising:
                print("Applying isotonic regression denoising...")

                # Fit isotonic regression
                df = fit_isotonic(df, result_col="isotonic_pred")

                # Remove noisy data
                print(f"Removing noisy data (threshold={noise_threshold})...")
                print(f"Dataset shape before cleaning: {df.shape}")
                df = remove_noisy_data(
                    df,
                    result_col="isotonic_pred",
                    threshold=noise_threshold,
                )
                print(f"Dataset shape after cleaning: {df.shape}")

            # Save cleaned dataframe
            cleaned_path = study_output_dir / f'{study_id}.feather'
            df.to_feather(cleaned_path)
            print(f"Saved cleaned data to {cleaned_path}")

            print(f"Successfully processed study {study_id}")

        except Exception as e:
            print(f"Error processing study {study_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("All studies processed!")
    print(f"{'='*60}")


def column_name_mapping(
    df: pd.DataFrame,
    sample: Optional[str] = None,
    drug: Optional[List[str]] = None,
    dose: Optional[List[str]] = None,
    value: Optional[str] = None,
) -> pd.DataFrame:
    """
    Remap column names in DataFrame to match expected format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to rename columns
    sample : str, optional
        Column name to map to 'sample_id'. If None, checks 'sample_id' exists.
    drug : list of str, optional
        Column names to map to 'drug1', 'drug2', etc. If None, checks 'drug1', etc. exist.
    dose : list of str, optional
        Column names to map to 'dose1', 'dose2', etc. If None, checks 'dose1', etc. exist.
    value : str, optional
        Column name to map to 'float_value'. If None, checks 'float_value' exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns

    Raises
    ------
    ValueError
        If expected columns don't exist or if drug/dose lists have different lengths

    Examples
    --------
    >>> # Remap columns
    >>> df = column_name_mapping(df, sample='cell_line', drug=['compound1'],
    ...                          dose=['concentration1'], value='viability')

    >>> # Just validate existing columns
    >>> df = column_name_mapping(df)
    """
    df = df.copy()
    rename_dict = {}

    # Handle sample_id
    if sample is not None:
        if sample not in df.columns:
            raise ValueError(f"Column '{sample}' not found in DataFrame. Available: {list(df.columns)}")
        rename_dict[sample] = 'sample_id'
    else:
        if 'sample_id' not in df.columns:
            raise ValueError(f"Column 'sample_id' not found in DataFrame. Available: {list(df.columns)}")

    # Handle drug columns
    if drug is not None:
        if isinstance(drug, str):
            drug = [drug]

        for i, drug_col in enumerate(drug, start=1):
            if drug_col not in df.columns:
                raise ValueError(f"Drug column '{drug_col}' not found in DataFrame. Available: {list(df.columns)}")
            rename_dict[drug_col] = f'drug{i}'
    else:
        # Check that at least drug1 exists
        if 'drug1' not in df.columns:
            raise ValueError(f"Column 'drug1' not found in DataFrame. Available: {list(df.columns)}")

    # Handle dose columns
    if dose is not None:
        if isinstance(dose, str):
            dose = [dose]

        for i, dose_col in enumerate(dose, start=1):
            if dose_col not in df.columns:
                raise ValueError(f"Dose column '{dose_col}' not found in DataFrame. Available: {list(df.columns)}")
            rename_dict[dose_col] = f'dose{i}'
    else:
        # Check that at least dose1 exists
        if 'dose1' not in df.columns:
            raise ValueError(f"Column 'dose1' not found in DataFrame. Available: {list(df.columns)}")

    # Validate drug and dose have same length if both provided
    if drug is not None and dose is not None:
        if len(drug) != len(dose):
            raise ValueError(f"Number of drug columns ({len(drug)}) must match number of dose columns ({len(dose)})")

    # Handle float_value
    if value is not None:
        if value not in df.columns:
            raise ValueError(f"Column '{value}' not found in DataFrame. Available: {list(df.columns)}")
        rename_dict[value] = 'float_value'
    else:
        if 'float_value' not in df.columns:
            raise ValueError(f"Column 'float_value' not found in DataFrame. Available: {list(df.columns)}")

    # Apply renaming
    if rename_dict:
        df = df.rename(columns=rename_dict)
        print(f"Renamed columns: {rename_dict}")
    else:
        print("No columns renamed. All expected columns already exist.")

    return df