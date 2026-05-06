import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DrugScreenConfig


class LearnedFourierEncoding(nn.Module):
    """Learnable Fourier features for encoding continuous values."""

    def __init__(self, config: DrugScreenConfig):
        super().__init__()
        self.embedding_dim = config.hidden_size
        self.num_frequencies = config.fourier_frequencies
        self.max_value = config.fourier_max_value

        # Learnable frequencies
        self.frequencies = nn.Parameter(
            torch.randn(self.num_frequencies) * 10.0
        )

        # Project Fourier features to embedding dimension
        self.proj = nn.Linear(2 * self.num_frequencies, self.embedding_dim)

    def forward(self, values):
        """
        Args:
            values: Tensor of shape (..., seq_len)
        Returns:
            embeddings: Tensor of shape (..., seq_len, embedding_dim)
        """
        # Normalize values
        normalized = values.float().unsqueeze(-1) / self.max_value  # (..., seq_len, 1)
        freqs = self.frequencies.view(1, 1, -1)  # (1, 1, num_freq)
        angles = normalized * freqs  # (..., seq_len, num_freq)

        # Concatenate sin and cos
        fourier_feats = torch.cat([
            torch.sin(angles),
            torch.cos(angles)
        ], dim=-1)  # (..., seq_len, 2*num_freq)

        # Project to final embedding
        embeddings = self.proj(fourier_feats)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, config: DrugScreenConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_attention_heads})"
            )

        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout_prob)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _split_heads(self, x):
        """Split hidden dimension into multiple heads."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_size)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) or (batch, seq_len, seq_len) or (batch, 1, seq_len, seq_len)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        residual = hidden_states
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        Q = self._split_heads(self.query(hidden_states))
        K = self._split_heads(self.key(hidden_states))
        V = self._split_heads(self.value(hidden_states))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Handle different mask shapes
            if attention_mask.dim() == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                attention_mask = attention_mask.unsqueeze(1)
            # else: already (batch, 1, seq_len, seq_len) or (batch, num_heads, seq_len, seq_len)

            # Convert boolean mask to additive mask
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.float()

            # Create additive mask (0 for attend, -inf for don't attend)
            attention_mask = (1.0 - attention_mask) * -10000.0
            scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Merge heads
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.output_proj(context)
        output = self.output_dropout(output)
        output = self.layer_norm(output + residual)

        return output


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention mechanism."""

    def __init__(self, config: DrugScreenConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_attention_heads})"
            )

        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout_prob)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _split_heads(self, x):
        """Split hidden dimension into multiple heads."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_value_states,
                query_mask=None, key_value_mask=None):
        """
        Args:
            query_states: (batch, query_len, hidden_size)
            key_value_states: (batch, kv_len, hidden_size)
            query_mask: (batch, query_len) or (batch, query_len, query_len)
            key_value_mask: (batch, kv_len)
        Returns:
            output: (batch, query_len, hidden_size)
        """
        residual = query_states
        batch_size, query_len, _ = query_states.size()
        _, kv_len, _ = key_value_states.size()

        # Project queries from query_states, keys/values from key_value_states
        Q = self._split_heads(self.query(query_states))
        K = self._split_heads(self.key(key_value_states))
        V = self._split_heads(self.value(key_value_states))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)

        # Apply key-value mask
        if key_value_mask is not None:
            # (batch, kv_len) -> (batch, 1, 1, kv_len)
            if key_value_mask.dim() == 2:
                mask = key_value_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = key_value_mask.unsqueeze(1)

            # Convert boolean mask to additive mask
            if mask.dtype == torch.bool:
                mask = mask.float()

            mask = (1.0 - mask) * -10000.0
            scores = scores + mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Merge heads
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, query_len, self.hidden_size)

        # Output projection
        output = self.output_proj(context)
        output = self.output_dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: DrugScreenConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

        if config.activation == 'gelu':
            self.activation = nn.GELU()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        residual = hidden_states

        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        output = self.layer_norm(hidden_states + residual)
        return output


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention."""

    def __init__(self, config: DrugScreenConfig):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        hidden_states = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states


class DecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and cross-attention."""

    def __init__(self, config: DrugScreenConfig, disable_self_attention: bool = False):
        super().__init__()
        self.disable_self_attention = disable_self_attention

        # Only create self-attention if not disabled
        if not self.disable_self_attention:
            self.self_attention = MultiHeadSelfAttention(config)

        self.cross_attention = MultiHeadCrossAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, query_states, key_value_states,
                query_mask=None, key_value_mask=None):
        """
        Args:
            query_states: (batch, query_len, hidden_size)
            key_value_states: (batch, kv_len, hidden_size)
            query_mask: (batch, query_len) or (batch, query_len, query_len)
            key_value_mask: (batch, kv_len)
        Returns:
            output: (batch, query_len, hidden_size)
        """
        hidden_states = query_states

        # Self-attention on queries (only if not disabled)
        if not self.disable_self_attention:
            hidden_states = self.self_attention(hidden_states, query_mask)

        # Cross-attention between queries and key-values
        hidden_states = self.cross_attention(
            hidden_states, key_value_states,
            query_mask, key_value_mask
        )

        # Feed-forward
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states


class EmbeddingLayer(nn.Module):
    """Embedding layer with optional continuous encoding."""

    def __init__(self, config: DrugScreenConfig, use_token_embeddings: bool = True):
        super().__init__()
        self.use_token_embeddings = use_token_embeddings

        # Only create token embeddings if needed
        if use_token_embeddings:
            self.token_embeddings = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.padding_idx
            )
        else:
            self.token_embeddings = None

        self.continuous_embeddings = LearnedFourierEncoding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids=None, continuous_ids=None, inputs_embeds=None):
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            continuous_ids: (batch, seq_len) continuous values (e.g., doses)
            inputs_embeds: (batch, seq_len, hidden_size) pre-computed embeddings
        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            if self.token_embeddings is None:
                raise ValueError("token_embeddings not initialized, cannot use input_ids")
            inputs_embeds = self.token_embeddings(input_ids)

        embeddings = inputs_embeds

        # Add continuous embeddings if provided
        if continuous_ids is not None:
            continuous_embeds = self.continuous_embeddings(continuous_ids)
            embeddings = embeddings + continuous_embeds

        embeddings = self.layer_norm(embeddings)
        return embeddings


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, config: DrugScreenConfig, use_token_embeddings: bool = True):
        super().__init__()
        self.config = config
        self.embeddings = EmbeddingLayer(config, use_token_embeddings=use_token_embeddings)
        self.layers = nn.ModuleList([
            EncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids=None, continuous_ids=None, inputs_embeds=None,
                attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            continuous_ids: (batch, seq_len)
            inputs_embeds: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        hidden_states = self.embeddings(input_ids, continuous_ids, inputs_embeds)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers with cross-attention."""

    def __init__(self, config: DrugScreenConfig, disable_self_attention: bool = False,
                 use_token_embeddings: bool = True):
        super().__init__()
        self.config = config
        self.disable_self_attention = disable_self_attention
        self.embeddings = EmbeddingLayer(config, use_token_embeddings=use_token_embeddings)
        self.layers = nn.ModuleList([
            DecoderLayer(config, disable_self_attention=disable_self_attention)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, query_ids=None, query_continuous_ids=None, query_embeds=None,
                key_value_states=None, query_mask=None, key_value_mask=None):
        """
        Args:
            query_ids: (batch, query_len)
            query_continuous_ids: (batch, query_len)
            query_embeds: (batch, query_len, hidden_size)
            key_value_states: (batch, kv_len, hidden_size)
            query_mask: (batch, query_len) or (batch, query_len, query_len)
            key_value_mask: (batch, kv_len)
        Returns:
            hidden_states: (batch, query_len, hidden_size)
        """
        hidden_states = self.embeddings(query_ids, query_continuous_ids, query_embeds)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, key_value_states,
                query_mask, key_value_mask
            )

        return hidden_states


def masked_average(x, mask=None):
    """Compute masked average over sequence dimension."""
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).to(x.dtype)
        return (x * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1.0)
    else:
        return x.mean(1)


class DrugCombinationEncoder(nn.Module):
    """Encoder for drug combinations with dose information."""

    def __init__(self, config: DrugScreenConfig):
        super().__init__()
        # No token embeddings needed - we always pass drug_embeds directly
        self.encoder = TransformerEncoder(config, use_token_embeddings=False)
        self.hidden_size = config.hidden_size

    def forward(self, drug_embeds, dose_levels, attention_mask=None):
        """
        Args:
            drug_embeds: (batch, num_drugs, hidden_size)
            dose_levels: (batch, num_drugs) normalized doses
            attention_mask: (batch, num_drugs)
        Returns:
            combination_embedding: (batch, hidden_size)
        """
        hidden_states = self.encoder(
            inputs_embeds=drug_embeds,
            continuous_ids=dose_levels,
            attention_mask=attention_mask
        )

        # Average over drugs
        combination_embedding = masked_average(hidden_states, attention_mask)
        return combination_embedding


class DoseResponseEncoder(nn.Module):
    """Encoder for dose-response observations using cross-attention."""

    def __init__(self, config: DrugScreenConfig):
        super().__init__()
        # Disable self-attention since we only want cross-attention to observations
        # No token embeddings needed - we always pass embeds directly
        self.decoder = TransformerDecoder(
            config,
            disable_self_attention=True,
            use_token_embeddings=False
        )
        self.hidden_size = config.hidden_size

    def forward(self, query_embeds, observation_embeds, viability_levels,
                query_mask=None, observation_mask=None):
        """
        Args:
            query_embeds: (batch, num_queries, hidden_size)
            observation_embeds: (batch, num_obs, hidden_size)
            viability_levels: (batch, num_obs) normalized viabilities
            query_mask: (batch, num_queries) or (batch, num_queries, num_queries) - not used when self-attention disabled
            observation_mask: (batch, num_obs)
        Returns:
            query_outputs: (batch, num_queries, hidden_size)
        """
        # Add viability information to observations
        observation_embeds_with_viability = self.decoder.embeddings(
            inputs_embeds=observation_embeds,
            continuous_ids=viability_levels
        )

        # Cross-attend from queries to observations (no self-attention)
        query_outputs = self.decoder(
            query_embeds=query_embeds,
            key_value_states=observation_embeds_with_viability,
            query_mask=None,  # No need for query mask since self-attention is disabled
            key_value_mask=observation_mask
        )

        return query_outputs


class SampleEncoder(nn.Module):
    """Encoder for sample perturbation responses."""

    def __init__(self, config: DrugScreenConfig):
        super().__init__()
        # No token embeddings needed - we always pass perturbation_embeds directly
        self.encoder = TransformerEncoder(config, use_token_embeddings=False)
        self.hidden_size = config.hidden_size

    def forward(self, perturbation_embeds, attention_mask=None):
        """
        Args:
            perturbation_embeds: (batch, num_perturbations, hidden_size)
            attention_mask: (batch, num_perturbations)
        Returns:
            sample_embedding: (batch, hidden_size)
        """
        hidden_states = self.encoder(
            inputs_embeds=perturbation_embeds,
            attention_mask=attention_mask
        )

        # Use first token as sample representation
        sample_embedding = hidden_states[:, 0]
        return sample_embedding