"""
OptiFormer: Transformer for Hyperparameter Optimization

Wraps LlamaForCausalLM with optimization-specific functionality.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json

from .config import OptiFormerConfig


class OptiFormer(nn.Module):
    """
    OptiFormer model for hyperparameter optimization.

    Uses decoder-only transformer with RoPE embeddings.
    Trained to predict next token in optimization trajectory.
    """

    def __init__(self, config: OptiFormerConfig):
        super().__init__()
        self.config = config

        # Import transformers here to avoid import errors during module load
        try:
            from transformers import LlamaForCausalLM
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        # Create underlying Llama model
        llama_config = config.to_llama_config()
        self.model = LlamaForCausalLM(llama_config)

        # Apply dropout if specified
        if config.dropout > 0:
            self._apply_dropout(config.dropout)

    def _apply_dropout(self, dropout: float):
        """Apply dropout to the model."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    @property
    def n_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            labels: (batch, seq_len) labels for loss computation

        Returns:
            ModelOutput with loss and logits
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> int:
        """
        Generate the next token given context.

        Args:
            input_ids: (1, seq_len) context tokens
            temperature: Sampling temperature
            top_k: Top-k sampling (0 to disable)
            top_p: Nucleus sampling threshold

        Returns:
            Next token ID
        """
        self.eval()
        outputs = self.model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last position

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][-1]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        return next_token

    @torch.no_grad()
    def generate_sequence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        stop_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Generate a sequence of tokens.

        Args:
            input_ids: (1, seq_len) context tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            stop_tokens: Token IDs that trigger generation stop

        Returns:
            (1, seq_len + new_tokens) generated sequence
        """
        self.eval()
        stop_tokens = stop_tokens or []

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Generate next token
            next_token = self.generate_next_token(
                generated,
                temperature=temperature,
                top_k=top_k,
            )

            # Append to sequence
            generated = torch.cat([
                generated,
                torch.tensor([[next_token]], device=generated.device)
            ], dim=1)

            # Check for stop token
            if next_token in stop_tokens:
                break

        return generated

    @torch.no_grad()
    def get_token_probabilities(
        self,
        input_ids: torch.Tensor,
        target_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Get probability distribution over next token.

        Args:
            input_ids: (1, seq_len) context tokens
            target_tokens: If provided, return probs only for these tokens

        Returns:
            Probability distribution
        """
        self.eval()
        outputs = self.model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        if target_tokens is not None:
            return probs[target_tokens]
        return probs

    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'config': self.config.to_dict(),
            'model_state_dict': self.model.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'OptiFormer':
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on

        Returns:
            Loaded OptiFormer model
        """
        checkpoint = torch.load(path, map_location=device)
        config = OptiFormerConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)

    def save_pretrained(self, path: str):
        """
        Save model in HuggingFace format.

        Args:
            path: Directory to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(path)

        # Save our config
        with open(path / 'optiformer_config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cuda') -> 'OptiFormer':
        """
        Load model from HuggingFace format.

        Args:
            path: Directory containing model
            device: Device to load model on

        Returns:
            Loaded OptiFormer model
        """
        from transformers import LlamaForCausalLM

        path = Path(path)

        # Load our config
        with open(path / 'optiformer_config.json', 'r') as f:
            config_dict = json.load(f)
        config = OptiFormerConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load weights
        model.model = LlamaForCausalLM.from_pretrained(path)

        return model.to(device)

    def freeze_embeddings(self):
        """Freeze embedding layer weights."""
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = False

    def unfreeze_embeddings(self):
        """Unfreeze embedding layer weights."""
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = True

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embedding vectors for input tokens."""
        return self.model.model.embed_tokens(input_ids)


def create_model(
    config: Optional[OptiFormerConfig] = None,
    size: str = "nano",
    vocab_size: Optional[int] = None,
) -> OptiFormer:
    """
    Create OptiFormer model.

    Args:
        config: Model configuration (if provided, other args ignored)
        size: Model size (nano, small, base, large)
        vocab_size: Override vocab size

    Returns:
        OptiFormer model
    """
    if config is None:
        config = OptiFormerConfig.from_size(size)

    if vocab_size is not None:
        config.vocab_size = vocab_size

    return OptiFormer(config)
