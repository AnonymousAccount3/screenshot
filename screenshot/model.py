"""Foundation model with integrated drug library."""

from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DrugScreenConfig
from .transformers_utils import (
    DrugCombinationEncoder,
    DoseResponseEncoder,
    SampleEncoder,
)


class DrugScreenEncoder(nn.Module):
    """
    Complete drug screen encoder that processes:
    1. Drug combinations with doses
    2. Dose-response observations
    3. Sample perturbation profiles
    """

    def __init__(self, config: DrugScreenConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Sub-encoders
        self.drug_combination_encoder = DrugCombinationEncoder(config)
        self.dose_response_encoder = DoseResponseEncoder(config)
        self.sample_encoder = SampleEncoder(config)

    def forward(self,
                observation_combo_ids,      # (bs, n_samples, n_perturbations, n_obs)
                drug_embeds,                # (vocab_size, n_drugs, hidden_size)
                dose_levels,                # (vocab_size, n_drugs)
                viability_levels,           # (bs, n_samples, n_perturbations, n_obs)
                attention_mask_drug,        # (vocab_size, n_drugs)
                attention_mask_obs,         # (bs, n_samples, n_perturbations, n_obs)
                attention_mask_perturb,     # (bs, n_samples, n_perturbations)
                query_drug_embeds,          # (bs, n_samples, n_queries, n_drugs, hidden_size)
                query_drug_mask,            # (bs, n_samples, n_queries, n_drugs)
                query_dose_levels,          # (bs, n_samples, n_queries, n_drugs)
                **kwargs):
        """
        Process drug screen data to generate sample and drug-dose embeddings.

        Returns:
            dict with:
                - 'sample_embeddings': (bs, n_samples, n_queries, hidden_size)
                - 'drug_dose_embeddings': (bs, n_samples, n_queries, hidden_size)
        """
        bs, n_samples, n_perturbations, n_obs = observation_combo_ids.shape
        _, _, n_queries, n_drugs, hidden_size = query_drug_embeds.shape
        vocab_size = drug_embeds.shape[0]

        # 1. Encode all drug combinations in vocabulary with their doses
        combo_embeds = self.drug_combination_encoder(
            drug_embeds=drug_embeds.view(-1, n_drugs, hidden_size),
            dose_levels=dose_levels.view(-1, n_drugs),
            attention_mask=attention_mask_drug.view(-1, n_drugs)
        )  # (vocab_size, hidden_size)

        # 2. Encode query drug combinations
        query_combo_embeds = self.drug_combination_encoder(
            drug_embeds=query_drug_embeds.view(-1, n_drugs, hidden_size),
            dose_levels=query_dose_levels.view(-1, n_drugs),
            attention_mask=query_drug_mask.view(-1, n_drugs)
        )  # (bs * n_samples * n_queries, hidden_size)

        query_combo_embeds = query_combo_embeds.view(
            bs, n_samples, n_queries, hidden_size
        )

        # 3. Look up observation combinations
        observation_embeds = F.embedding(
            observation_combo_ids.view(-1, n_obs),
            combo_embeds,
            padding_idx=0
        )  # (bs * n_samples * n_perturbations, n_obs, hidden_size)

        # 4. Expand query combinations for each perturbation
        query_embeds = query_combo_embeds.unsqueeze(2).expand(
            -1, -1, n_perturbations, -1, -1
        ).contiguous()  # (bs, n_samples, n_perturbations, n_queries, hidden_size)

        query_embeds = query_embeds.view(
            -1, n_queries, hidden_size
        )  # (bs * n_samples * n_perturbations, n_queries, hidden_size)

        # 5. Encode dose-response using cross-attention (no self-attention needed)
        dose_response_embeds = self.dose_response_encoder(
            query_embeds=query_embeds,
            observation_embeds=observation_embeds,
            viability_levels=viability_levels.view(-1, n_obs),
            query_mask=None,  # No self-attention, so no query mask needed
            observation_mask=attention_mask_obs.view(-1, n_obs)
        )  # (bs * n_samples * n_perturbations, n_queries, hidden_size)

        dose_response_embeds = dose_response_embeds.view(
            bs, n_samples, n_perturbations, n_queries, hidden_size
        )

        # Reorganize: (bs, n_samples, n_queries, n_perturbations, hidden_size)
        dose_response_embeds = dose_response_embeds.permute(0, 1, 3, 2, 4).contiguous()

        # 6. Encode sample by aggregating across perturbations
        attention_mask_perturb_expanded = attention_mask_perturb.unsqueeze(2).expand(
            -1, -1, n_queries, -1
        ).contiguous()

        sample_embeds = self.sample_encoder(
            perturbation_embeds=dose_response_embeds.view(-1, n_perturbations, hidden_size),
            attention_mask=attention_mask_perturb_expanded.view(-1, n_perturbations)
        )  # (bs * n_samples * n_queries, hidden_size)

        sample_embeds = sample_embeds.view(
            bs, n_samples, n_queries, hidden_size
        )

        return {
            'sample_embeddings': sample_embeds,
            'drug_dose_embeddings': query_combo_embeds,
        }


class FoundationModel(nn.Module):
    """
    Foundation model for drug screening prediction.
    """

    def __init__(self, config):
        """
        Initialize model.

        Args:
            config: DrugScreenConfig
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Drug embeddings (vocab_size + n_missing_tokens for unknown drugs)
        embedding_size = config.vocab_size + getattr(config, 'n_missing_tokens', 0)
        self.drug_embeddings = nn.Embedding(
            embedding_size,
            config.hidden_size,
            padding_idx=config.padding_idx
        )

        # Main encoder
        self.drugscreen_encoder = DrugScreenEncoder(config)

        # Prediction head
        self.viability_predictor = nn.Linear(config.hidden_size, 1)

        # Initialize weights for all modules
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights of a module."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_sample_embeddings(self,
                              sample_perturb_ids,
                              drug_ids,
                              dose_levels,
                              viability_levels,
                              attention_mask_drug,
                              attention_mask_obs,
                              attention_mask_perturbs,
                              perturb_drug_ids,
                              perturb_mask,
                              label_dose,
                              **kwargs):
        """
        Generate sample embeddings from drug screen data.

        Args:
            sample_perturb_ids: (bs, n_samples, n_perturbations, n_obs) - observation combo IDs
            drug_ids: (vocab_size, n_drugs)
            dose_levels: (vocab_size, n_drugs)
            viability_levels: (bs, n_samples, n_perturbations, n_obs)
            attention_mask_drug: (vocab_size, n_drugs)
            attention_mask_obs: (bs, n_samples, n_perturbations, n_obs)
            attention_mask_perturbs: (bs, n_samples, n_perturbations)
            perturb_drug_ids: (bs, n_samples, n_queries, n_drugs) - query drug IDs
            perturb_mask: (bs, n_samples, n_queries, n_drugs)
            label_dose: (bs, n_samples, n_queries, n_drugs) - query doses

        Returns:
            dict with:
                - 'sample_embeddings': (bs, n_samples, n_queries, hidden_size)
                - 'drug_dose_embeddings': (bs, n_samples, n_queries, hidden_size)
        """
        # Embed drugs
        drug_embeds = self.drug_embeddings(drug_ids)  # (vocab_size, n_drugs, hidden_size)
        query_drug_embeds = self.drug_embeddings(perturb_drug_ids)  # (bs, n_samples, n_queries, n_drugs, hidden_size)

        # Normalize dose levels (already normalized to [0, 1])
        dose_levels_normalized = dose_levels.to(drug_embeds.dtype)
        query_dose_normalized = label_dose.to(query_drug_embeds.dtype)

        # Encode
        encoder_output = self.drugscreen_encoder(
            observation_combo_ids=sample_perturb_ids,
            drug_embeds=drug_embeds,
            dose_levels=dose_levels_normalized,
            viability_levels=viability_levels,
            attention_mask_drug=attention_mask_drug,
            attention_mask_obs=attention_mask_obs,
            attention_mask_perturb=attention_mask_perturbs,
            query_drug_embeds=query_drug_embeds,
            query_drug_mask=perturb_mask,
            query_dose_levels=query_dose_normalized,
        )

        return encoder_output

    def forward(self,
                sample_perturb_ids,
                drug_ids,
                dose_levels,
                viability_levels,
                attention_mask_drug,
                attention_mask_obs,
                attention_mask_perturbs,
                perturb_drug_ids,
                perturb_mask,
                label_dose,
                label_viability,
                **kwargs):
        """
        Forward pass with loss computation.

        Args:
            sample_perturb_ids: (bs, n_samples, n_perturbations, n_obs)
            drug_ids: (vocab_size, n_drugs)
            dose_levels: (vocab_size, n_drugs)
            viability_levels: (bs, n_samples, n_perturbations, n_obs)
            attention_mask_drug: (vocab_size, n_drugs)
            attention_mask_obs: (bs, n_samples, n_perturbations, n_obs)
            attention_mask_perturbs: (bs, n_samples, n_perturbations)
            perturb_drug_ids: (bs, n_samples, n_queries, n_drugs)
            perturb_mask: (bs, n_samples, n_queries, n_drugs)
            label_dose: (bs, n_samples, n_queries, n_drugs)
            label_viability: (bs, n_samples, n_queries)

        Returns:
            output_dict: Dictionary containing loss and predictions
        """
        # Get embeddings
        encoder_output = self.get_sample_embeddings(
            sample_perturb_ids=sample_perturb_ids,
            drug_ids=drug_ids,
            dose_levels=dose_levels,
            viability_levels=viability_levels,
            attention_mask_drug=attention_mask_drug,
            attention_mask_obs=attention_mask_obs,
            attention_mask_perturbs=attention_mask_perturbs,
            perturb_drug_ids=perturb_drug_ids,
            perturb_mask=perturb_mask,
            label_dose=label_dose,
        )
        sample_embeddings = encoder_output['sample_embeddings']

        # Predict viability
        predictions = torch.sigmoid(
            self.viability_predictor(sample_embeddings).squeeze(-1)
        )  # (bs, n_samples, n_queries)

        # Compute loss (MAE)
        loss = F.l1_loss(predictions, label_viability)

        return {
            'loss': loss,
            'predictions': predictions,
            'sample_embeddings': sample_embeddings,
            'drug_dose_embeddings': encoder_output['drug_dose_embeddings'],
        }

    def save_pretrained(self, save_path: str):
        """
        Save model checkpoint.

        Args:
            save_path: Directory to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        checkpoint = {
            'model_state_dict': self.state_dict(),
        }

        model_path = save_path / 'model.pt'
        torch.save(checkpoint, model_path)

        # Save config
        self.config.save_pretrained(save_path)

        print(f"✓ Model saved to: {save_path}")

    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        Load model from checkpoint.

        Args:
            model_path: Directory containing model files

        Returns:
            Loaded model
        """
        model_path = Path(model_path)

        print(f"Loading model from: {model_path}")

        # Load config
        config = DrugScreenConfig.from_pretrained(model_path)

        # Create model
        model = cls(config)

        # Load weights
        checkpoint_path = model_path / 'model.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"✓ Model loaded successfully")

        return model