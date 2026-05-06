"""ScreenShot inference class for pretrained drug screening models."""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict
from tqdm import tqdm

from .model import FoundationModel
from .data_loader import TestDataset


class ScreenShot:
    """
    Inference interface for pretrained drug screening models.

    Performs predictions on drug combinations using context from input data.
    All inputs must be preprocessed (drugs tokenized, doses normalized).
    Use Preprocessor for data preparation.

    Example:
        ```python
        from screenshot import ScreenShot, Preprocessor
        from datascreen import DrugLibrary, extract_drug_names

        # Load model
        screenshot = ScreenShot.from_pretrained("path/to/model")

        # Preprocess data separately
        drug_library = DrugLibrary.from_pretrained("path/to/drug_library")
        drug_library.update(extract_drug_names(df))
        preprocessor = Preprocessor(drug_library, n_drugs=3)
        df_processed = preprocessor.transform(df)

        # Predict
        predictions = screenshot.predict(df_input, df_query)

        # Recover drug names for saving
        predictions = preprocessor.inverse_transform_drugs(predictions)
        ```
    """
    def __init__(
        self,
        model: FoundationModel,
        device: str = None,
        batch_size: int = 128,
    ):
        """
        Initialize ScreenShot inference engine.

        Args:
            model: Pretrained FoundationModel
            device: Device to run inference on. If None or 'auto', auto-detects.
            batch_size: Batch size for query processing
        """
        if device is None or device == 'auto':
            from .utils import get_device
            device = get_device()
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.config = model.config

    @torch.no_grad()
    def predict(
        self,
        df_input: pd.DataFrame,
        df_query: pd.DataFrame,
        n_drugs: int = 3,
        sample_id_col: str = "sample_id",
        col_viability: str = "float_value",
        return_embeddings: bool = False,
        show_progress: bool = True,
    ) -> Union[pd.DataFrame, Dict[str, Union[pd.DataFrame, np.ndarray]]]:
        """
        Make predictions on preprocessed data.

        Input DataFrames must already be preprocessed (drugs tokenized as
        integer IDs, doses log-scaled and normalized to [0, 1], viability
        clipped). Use Preprocessor.transform() for raw data.

        Args:
            df_input: Preprocessed input dataframe with context observations
            df_query: Preprocessed query dataframe with combinations to predict
            n_drugs: Number of drugs per combination (default: 3)
            sample_id_col: Name of sample ID column (default: "sample_id")
            col_viability: Name of viability column (default: "float_value")
            return_embeddings: If True, also return sample and drug-dose embeddings
            show_progress: Show progress bar during prediction

        Returns:
            If return_embeddings=False:
                DataFrame with predictions added as 'predicted_viability' column
            If return_embeddings=True:
                Dictionary with:
                    - 'predictions': DataFrame with predictions
                    - 'embeddings': Sample embeddings array (n_queries, hidden_size)
                    - 'drug_dose_embeddings': Drug-dose embeddings array (n_queries, hidden_size)
        """
        # Create test dataset from preprocessed data
        dataset = TestDataset(
            input_df=df_input,
            query_df=df_query,
            n_drugs=n_drugs,
            sample_id_col=sample_id_col,
            col_viability=col_viability,
            batch_size=self.batch_size,
        )

        # Run inference
        all_predictions = []
        all_embeddings = [] if return_embeddings else None
        all_drug_dose_embeddings = [] if return_embeddings else None
        all_indices = []

        iterator = range(len(dataset))
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting", unit="batch")

        for batch_idx in iterator:
            batch = dataset[batch_idx]

            # Move batch to device
            batch_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get predictions
            if return_embeddings:
                encoder_output = self.model.get_sample_embeddings(**batch_device)
                sample_embeddings = encoder_output['sample_embeddings']
                drug_dose_embeddings = encoder_output['drug_dose_embeddings']
                predictions = torch.sigmoid(
                    self.model.viability_predictor(sample_embeddings).squeeze(-1)
                )
                all_embeddings.append(sample_embeddings.cpu().numpy())
                all_drug_dose_embeddings.append(drug_dose_embeddings.cpu().numpy())
            else:
                outputs = self.model(**batch_device)
                predictions = outputs['predictions']

            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_indices.append(batch_device['origin_index'].cpu().numpy())

        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=-1)
        all_indices = np.concatenate(all_indices, axis=-1)

        if return_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=2)
            all_drug_dose_embeddings = np.concatenate(all_drug_dose_embeddings, axis=2)

        # Flatten to match query dataframe
        predictions_flat = all_predictions.flatten()
        indices_flat = all_indices.flatten()

        # Remove NaN padding
        valid_mask = ~np.isnan(indices_flat)
        predictions_flat = predictions_flat[valid_mask]
        indices_flat = indices_flat[valid_mask].astype(int)

        # Create results dataframe
        df_results = df_query.copy()
        df_results['predicted_viability'] = np.nan
        df_results.iloc[indices_flat, df_results.columns.get_loc('predicted_viability')] = predictions_flat

        if return_embeddings:
            hidden_size = all_embeddings.shape[-1]
            embeddings_flat = all_embeddings.reshape(-1, hidden_size)
            embeddings_flat = embeddings_flat[valid_mask]

            unique_indices = np.unique(indices_flat)
            expected_indices = np.arange(len(df_query))
            if not np.array_equal(sorted(unique_indices), expected_indices):
                print(f"Warning: indices_flat does not form a complete set from 0 to {len(df_query)-1}")
                print(f"   Expected {len(expected_indices)} indices, got {len(unique_indices)} unique indices")

            final_embeddings = embeddings_flat[np.argsort(indices_flat)]

            # Drug-dose embeddings (same shape/indexing as sample embeddings)
            dd_hidden_size = all_drug_dose_embeddings.shape[-1]
            dd_flat = all_drug_dose_embeddings.reshape(-1, dd_hidden_size)
            dd_flat = dd_flat[valid_mask]
            final_drug_dose_embeddings = dd_flat[np.argsort(indices_flat)]

            return {
                'predictions': df_results,
                'embeddings': final_embeddings,
                'drug_dose_embeddings': final_drug_dose_embeddings,
            }
        else:
            return df_results

    @torch.no_grad()
    def get_embeddings(
        self,
        df_input: pd.DataFrame,
        df_query: pd.DataFrame,
        n_drugs: int = 3,
        sample_id_col: str = "sample_id",
        col_viability: str = "float_value",
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Get sample embeddings from preprocessed data.

        Args:
            df_input: Preprocessed input dataframe with context observations
            df_query: Preprocessed query dataframe with combinations
            n_drugs: Number of drugs per combination
            sample_id_col: Name of sample ID column
            col_viability: Name of viability column
            show_progress: Show progress bar

        Returns:
            Sample embeddings array of shape (n_queries, hidden_size) aligned with df_query
        """
        result = self.predict(
            df_input=df_input,
            df_query=df_query,
            n_drugs=n_drugs,
            sample_id_col=sample_id_col,
            col_viability=col_viability,
            return_embeddings=True,
            show_progress=show_progress,
        )

        return result['embeddings']

    @torch.no_grad()
    def get_drug_dose_embeddings(
        self,
        df: pd.DataFrame,
        n_drugs: int = 3,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract drug-dose combination embeddings from the model.

        Passes tokenized drug IDs and normalized doses through the
        drug_combination_encoder to get intermediate representations.
        Useful for cold-start clustering (e.g., k-medoids).

        Args:
            df: Preprocessed DataFrame with tokenized drug columns (drug1, drug2, ...)
                and normalized dose columns (dose1, dose2, ...).
            n_drugs: Number of drug columns
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings (n_rows, hidden_size)
        """
        drug_columns = [f"drug{i+1}" for i in range(n_drugs)]
        dose_columns = [f"dose{i+1}" for i in range(n_drugs)]

        all_embeddings = []

        n_batches = (len(df) + batch_size - 1) // batch_size

        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings")

        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            df_chunk = df.iloc[start_idx:end_idx]

            drug_ids = torch.tensor(
                df_chunk[drug_columns].values, dtype=torch.long, device=self.device
            )
            dose_ids = torch.tensor(
                df_chunk[dose_columns].values, dtype=torch.float32, device=self.device
            )
            attention_mask = (drug_ids > 0).long()

            # Get drug embeddings from the embedding layer
            drug_embeds = self.model.drug_embeddings(drug_ids)

            # Pass through drug_combination_encoder
            embeddings = self.model.drugscreen_encoder.drug_combination_encoder(
                drug_embeds=drug_embeds,
                attention_mask=attention_mask,
                dose_levels=dose_ids,
            )

            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    @classmethod
    @torch.no_grad()
    def from_pretrained(
        cls,
        model_path: str,
        device: str = None,
        batch_size: int = 128,
    ) -> 'ScreenShot':
        """
        Load pretrained model and create ScreenShot instance.

        Args:
            model_path: Path to pretrained model directory
            device: Device to run inference on
            batch_size: Batch size for query processing

        Returns:
            ScreenShot instance ready for inference
        """
        model_path = Path(model_path)

        print(f"Loading ScreenShot from: {model_path}")

        # Load model
        model = FoundationModel.from_pretrained(model_path)
        print(f"  Model loaded")

        screenshot = cls(
            model=model,
            device=device,
            batch_size=batch_size,
        )

        print(f"  ScreenShot ready on {screenshot.device}")

        return screenshot

    def save_pretrained(self, save_path: str):
        """
        Save model for later use.

        Args:
            save_path: Directory to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_path)

    def __repr__(self):
        return (
            f"ScreenShot(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  device={self.device},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )
