"""
OptiFormer Model Configurations

Three sizes: nano (smoke test), small (validation), base (production)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class OptiFormerConfig:
    """Configuration for OptiFormer model."""

    # Model identity
    size: str = "nano"

    # Architecture
    vocab_size: int = 1200
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: Optional[int] = None  # If None, equals num_attention_heads
    max_position_embeddings: int = 512

    # Positional encoding
    rope_theta: float = 10000.0

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0

    # Activation
    hidden_act: str = "silu"

    # Layer normalization
    rms_norm_eps: float = 1e-6

    # Other
    tie_word_embeddings: bool = False
    use_cache: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def nano(cls) -> 'OptiFormerConfig':
        """~8M params - for smoke testing"""
        return cls(
            size="nano",
            vocab_size=1200,
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=512,
            dropout=0.1,
        )

    @classmethod
    def small(cls) -> 'OptiFormerConfig':
        """~25M params - for validation"""
        return cls(
            size="small",
            vocab_size=1500,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=6,
            num_attention_heads=6,
            max_position_embeddings=1024,
            dropout=0.1,
        )

    @classmethod
    def base(cls) -> 'OptiFormerConfig':
        """~50M params - for production"""
        return cls(
            size="base",
            vocab_size=2000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=8,
            num_attention_heads=8,
            max_position_embeddings=2048,
            dropout=0.1,
        )

    @classmethod
    def large(cls) -> 'OptiFormerConfig':
        """~150M params - for full scale"""
        return cls(
            size="large",
            vocab_size=3000,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=4096,
            dropout=0.1,
        )

    @classmethod
    def from_size(cls, size: str) -> 'OptiFormerConfig':
        """Create config from size name."""
        size_map = {
            'nano': cls.nano,
            'small': cls.small,
            'base': cls.base,
            'large': cls.large,
        }
        if size not in size_map:
            raise ValueError(f"Unknown size: {size}. Available: {list(size_map.keys())}")
        return size_map[size]()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OptiFormerConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'size': self.size,
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'max_position_embeddings': self.max_position_embeddings,
            'rope_theta': self.rope_theta,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'hidden_act': self.hidden_act,
            'rms_norm_eps': self.rms_norm_eps,
            'tie_word_embeddings': self.tie_word_embeddings,
            'use_cache': self.use_cache,
        }

    def to_llama_config(self):
        """Convert to HuggingFace LlamaConfig."""
        try:
            from transformers import LlamaConfig
            return LlamaConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                max_position_embeddings=self.max_position_embeddings,
                rope_theta=self.rope_theta,
                hidden_act=self.hidden_act,
                rms_norm_eps=self.rms_norm_eps,
                tie_word_embeddings=self.tie_word_embeddings,
                use_cache=self.use_cache,
                # Note: Llama doesn't have dropout in config in the same way
                # We'll set it during training
            )
        except ImportError:
            raise ImportError("transformers library required for LlamaConfig conversion")

    def count_parameters(self) -> int:
        """
        Estimate parameter count (approximate).

        Based on Llama architecture:
        - Embedding: vocab_size * hidden_size
        - Each layer:
          - Self-attention: 4 * hidden_size^2 (Q, K, V, O projections)
          - MLP: 3 * hidden_size * intermediate_size (gate, up, down)
          - Layer norms: 2 * hidden_size
        - Output head: vocab_size * hidden_size (if not tied)
        """
        # Embedding
        params = self.vocab_size * self.hidden_size

        # Per-layer
        per_layer = (
            4 * self.hidden_size * self.hidden_size +  # Attention
            3 * self.hidden_size * self.intermediate_size +  # MLP
            2 * self.hidden_size  # Layer norms
        )
        params += self.num_hidden_layers * per_layer

        # Output head (if not tied)
        if not self.tie_word_embeddings:
            params += self.vocab_size * self.hidden_size

        # Final layer norm
        params += self.hidden_size

        return params

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"OptiFormerConfig(\n"
            f"  size='{self.size}',\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  intermediate_size={self.intermediate_size},\n"
            f"  num_hidden_layers={self.num_hidden_layers},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  max_position_embeddings={self.max_position_embeddings},\n"
            f"  estimated_params={params:,} (~{params/1e6:.1f}M)\n"
            f")"
        )
