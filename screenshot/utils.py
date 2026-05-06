from typing import Dict, Any


def get_device() -> str:
    """Auto-detect the best available device.

    Returns 'cuda' if available, else 'mps' if available, else 'cpu'.
    """
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class DrugScreenConfig:
    """
    Configuration class for DrugScreenFoundationModel.

    Args:
        vocab_size (int): Size of drug vocabulary. Default: 10000
        hidden_size (int): Dimensionality of embeddings and hidden states. Default: 768
        num_hidden_layers (int): Number of transformer layers. Default: 12
        num_attention_heads (int): Number of attention heads. Default: 12
        intermediate_size (int): Dimensionality of feed-forward layer. Default: 3072
        dropout_prob (float): Dropout probability. Default: 0.1
        attention_dropout_prob (float): Attention dropout probability. Default: 0.1
        layer_norm_eps (float): Epsilon for layer normalization. Default: 1e-12
        initializer_range (float): Standard deviation for weight initialization. Default: 0.02
        n_drugs (int): Maximum number of drugs in a combination. Default: 3
        padding_idx (int): Index for padding token. Default: 0
        fourier_frequencies (int): Number of Fourier frequencies for continuous encoding. Default: 64
        fourier_max_value (float): Maximum value for Fourier encoding normalization. Default: 1.0
        activation (str): Activation function ('gelu' or 'relu'). Default: 'gelu'
    """

    def __init__(
        self,
        # Vocabulary
        vocab_size: int = 10000,
        padding_idx: int = 0,

        # Architecture
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,

        # Regularization
        dropout_prob: float = 0.,
        attention_dropout_prob: float = 0.,
        layer_norm_eps: float = 1e-12,

        # Initialization
        initializer_range: float = 0.02,

        # Drug screening specific
        n_drugs: int = 3,

        # Fourier encoding
        fourier_frequencies: int = 64,
        fourier_max_value: float = 1.0,

        # Missing drug tokens
        n_missing_tokens: int = 0,
        missing_drug_alpha_max: float = 0.2,

        # Other
        activation: str = 'gelu',

        **kwargs
    ):
        # Vocabulary settings
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        # Architecture settings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        # Regularization
        self.dropout_prob = dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.layer_norm_eps = layer_norm_eps

        # Initialization
        self.initializer_range = initializer_range

        # Drug screening specific
        self.n_drugs = n_drugs

        # Fourier encoding
        self.fourier_frequencies = fourier_frequencies
        self.fourier_max_value = fourier_max_value

        # Missing drug tokens
        self.n_missing_tokens = n_missing_tokens
        self.missing_drug_alpha_max = missing_drug_alpha_max

        # Other
        self.activation = activation

        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        if self.activation not in ['gelu', 'relu']:
            raise ValueError(
                f"activation must be 'gelu' or 'relu', got {self.activation}"
            )

        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DrugScreenConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def save_pretrained(self, save_path: str):
        """Save config to JSON file."""
        import json
        import os

        os.makedirs(save_path, exist_ok=True)
        config_file = os.path.join(save_path, 'config.json')

        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, load_path: str) -> 'DrugScreenConfig':
        """Load config from JSON file."""
        import json
        import os

        config_file = os.path.join(load_path, 'config.json')

        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"