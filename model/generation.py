"""
Inference-time Generation Utilities

Provides utilities for using the trained OptiFormer model
to suggest hyperparameter configurations.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .optiformer import OptiFormer
from ..data.tokenizer import SequenceTokenizer, Trial


@dataclass
class GenerationConfig:
    """Configuration for hyperparameter generation."""
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_retries: int = 3


class HyperparameterGenerator:
    """
    Generator for suggesting hyperparameter configurations.

    Uses trained OptiFormer model to predict optimal hyperparameters
    given optimization history.
    """

    def __init__(
        self,
        model: OptiFormer,
        tokenizer: SequenceTokenizer,
        device: str = 'cuda',
        config: GenerationConfig = None,
    ):
        """
        Initialize the generator.

        Args:
            model: Trained OptiFormer model
            tokenizer: Sequence tokenizer
            device: Device to run inference on
            config: Generation configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or GenerationConfig()

        self.model.eval()
        self.model.to(device)

        # Cache param order
        self.param_order = tokenizer.param_order

    @torch.no_grad()
    def suggest(
        self,
        history: List[Trial],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Suggest next hyperparameter configuration.

        Args:
            history: List of previous trials
            temperature: Override generation temperature

        Returns:
            Dict of parameter name -> value
        """
        temperature = temperature or self.config.temperature

        # Generate parameters one at a time
        params = {}
        for param_name in self.param_order:
            # Encode context including params generated so far
            tokens = self.tokenizer.encode_for_inference(history, params)
            input_ids = torch.tensor([tokens], device=self.device)

            # Get probability distribution over next token
            probs = self.model.get_token_probabilities(input_ids)

            # Get valid tokens for this parameter
            valid_tokens = self.tokenizer.get_valid_tokens_for_param(param_name)

            # Mask invalid tokens
            mask = torch.zeros_like(probs)
            mask[valid_tokens] = 1
            masked_probs = probs * mask
            masked_probs = masked_probs / masked_probs.sum()

            # Apply temperature
            if temperature != 1.0:
                logits = torch.log(masked_probs + 1e-10) / temperature
                masked_probs = torch.softmax(logits, dim=-1)

            # Sample
            next_token = torch.multinomial(masked_probs, num_samples=1).item()

            # Decode value
            value = self.tokenizer.decode_predicted_value(next_token, param_name)
            params[param_name] = value

        return params

    @torch.no_grad()
    def suggest_batch(
        self,
        history: List[Trial],
        n_suggestions: int = 5,
        temperature: Optional[float] = None,
        diverse: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Suggest multiple configurations.

        Args:
            history: List of previous trials
            n_suggestions: Number of suggestions to generate
            temperature: Generation temperature
            diverse: If True, increase temperature for later suggestions

        Returns:
            List of parameter dictionaries
        """
        suggestions = []

        for i in range(n_suggestions):
            # Optionally increase temperature for diversity
            t = temperature or self.config.temperature
            if diverse:
                t = t * (1 + 0.2 * i)

            suggestion = self.suggest(history, temperature=t)
            suggestions.append(suggestion)

        return suggestions

    @torch.no_grad()
    def get_param_distribution(
        self,
        history: List[Trial],
        param_name: str,
        params_so_far: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Get probability distribution for a specific parameter.

        Args:
            history: Optimization history
            param_name: Parameter to get distribution for
            params_so_far: Parameters already selected

        Returns:
            Array of probabilities for each possible value
        """
        params_so_far = params_so_far or {}

        # Build context up to this parameter
        tokens = self.tokenizer.encode_for_inference(history, params_so_far)
        input_ids = torch.tensor([tokens], device=self.device)

        # Get probabilities
        probs = self.model.get_token_probabilities(input_ids)

        # Get valid tokens
        valid_tokens = self.tokenizer.get_valid_tokens_for_param(param_name)

        # Extract probabilities for valid tokens
        valid_probs = probs[valid_tokens].cpu().numpy()
        valid_probs = valid_probs / valid_probs.sum()

        return valid_probs

    def score_configuration(
        self,
        history: List[Trial],
        config: Dict[str, Any],
    ) -> float:
        """
        Score a configuration based on model likelihood.

        Args:
            history: Optimization history
            config: Configuration to score

        Returns:
            Log probability of configuration
        """
        log_prob = 0.0

        params_so_far = {}
        for param_name in self.param_order:
            # Get distribution
            dist = self.get_param_distribution(history, param_name, params_so_far)

            # Get value and find its index
            value = config[param_name]
            spec = self.tokenizer.param_specs[param_name]

            if spec.is_categorical:
                idx = spec.categories.index(value)
            else:
                # Find closest bin
                token = self.tokenizer.numerical.encode(
                    value, spec.low, spec.high, spec.log_scale
                )
                idx = token  # Index in valid tokens

            # Add log probability
            log_prob += np.log(dist[idx] + 1e-10)

            params_so_far[param_name] = value

        return log_prob


class EnsembleGenerator:
    """
    Ensemble of multiple generation strategies.

    Combines model-based suggestions with random exploration
    for better coverage of the search space.
    """

    def __init__(
        self,
        model: OptiFormer,
        tokenizer: SequenceTokenizer,
        exploration_rate: float = 0.1,
        device: str = 'cuda',
    ):
        """
        Initialize ensemble generator.

        Args:
            model: Trained OptiFormer model
            tokenizer: Sequence tokenizer
            exploration_rate: Fraction of random suggestions
            device: Device for inference
        """
        self.generator = HyperparameterGenerator(model, tokenizer, device)
        self.tokenizer = tokenizer
        self.exploration_rate = exploration_rate

    def suggest(self, history: List[Trial]) -> Dict[str, Any]:
        """
        Suggest configuration using ensemble strategy.

        With probability exploration_rate, returns random config.
        Otherwise uses model-based suggestion.
        """
        if np.random.random() < self.exploration_rate:
            return self._random_suggestion()
        return self.generator.suggest(history)

    def _random_suggestion(self) -> Dict[str, Any]:
        """Generate random configuration."""
        params = {}
        for param_name, spec in self.tokenizer.param_specs.items():
            if spec.is_categorical:
                params[param_name] = np.random.choice(spec.categories)
            elif spec.is_integer:
                params[param_name] = np.random.randint(int(spec.low), int(spec.high) + 1)
            elif spec.log_scale:
                log_val = np.random.uniform(np.log(spec.low), np.log(spec.high))
                params[param_name] = np.exp(log_val)
            else:
                params[param_name] = np.random.uniform(spec.low, spec.high)
        return params
