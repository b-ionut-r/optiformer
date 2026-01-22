"""
Optuna Sampler using Trained OptiFormer Model

Provides Optuna-compatible sampler interface for OptiFormer.
"""

import optuna
from optuna.samplers import BaseSampler
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
    BaseDistribution,
)
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
import torch
from torch.cuda.amp import autocast
import numpy as np
from typing import Dict, Any, Optional, Sequence, List

from model import OptiFormer
from data.tokenizer import SequenceTokenizer, Trial


class OptiFormerSampler(BaseSampler):
    """
    Optuna sampler using trained OptiFormer model.

    Usage:
        model = OptiFormer.load('model.pt')
        tokenizer = SequenceTokenizer(...)
        sampler = OptiFormerSampler(model, tokenizer)
        study = optuna.create_study(sampler=sampler)
    """

    def __init__(
        self,
        model: OptiFormer,
        tokenizer: SequenceTokenizer,
        device: str = 'cuda',
        temperature: float = 0.8,
        fallback_sampler: Optional[BaseSampler] = None,
        min_trials_for_model: int = 3,
        exploration_rate: float = 0.1,
        use_amp: bool = True,
    ):
        """
        Initialize OptiFormer sampler.

        Args:
            model: Trained OptiFormer model
            tokenizer: Sequence tokenizer
            device: Device for inference
            temperature: Sampling temperature
            fallback_sampler: Sampler for cold start
            min_trials_for_model: Minimum trials before using model
            exploration_rate: Probability of random exploration
            use_amp: Use automatic mixed precision for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature
        self.fallback = fallback_sampler or optuna.samplers.RandomSampler()
        self.min_trials = min_trials_for_model
        self.exploration_rate = exploration_rate
        self.use_amp = use_amp and torch.cuda.is_available()

        self.model.eval()
        self.model.to(self.device)

        # Parameter order from tokenizer
        self.param_order = tokenizer.param_order

    def infer_relative_search_space(
        self,
        study: Study,
        trial: optuna.Trial,
    ) -> Dict[str, BaseDistribution]:
        """Return the search space."""
        return optuna.search_space.intersection_search_space(study)

    def sample_relative(
        self,
        study: Study,
        trial: optuna.Trial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        """Sample using OptiFormer or fallback."""

        # Use fallback for cold start
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(completed_trials) < self.min_trials:
            return {}  # Will use sample_independent

        # Explore randomly with some probability
        if np.random.random() < self.exploration_rate:
            return {}  # Will use sample_independent (random)

        try:
            # Build history from completed trials
            history = self._build_history(completed_trials, search_space)

            if len(history) == 0:
                return {}

            # Generate parameters
            params = {}
            params_so_far = {}

            for name in self.param_order:
                if name not in search_space:
                    continue

                dist = search_space[name]
                value = self._sample_param(history, name, dist, params_so_far)
                params[name] = value
                params_so_far[name] = value

            return params

        except Exception as e:
            # Fallback on any error
            print(f"OptiFormer sampling error: {e}")
            return {}

    def sample_independent(
        self,
        study: Study,
        trial: optuna.Trial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Fallback to random for independent sampling."""
        return self.fallback.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _build_history(
        self,
        trials: List[FrozenTrial],
        search_space: Dict[str, BaseDistribution]
    ) -> List[Trial]:
        """Convert Optuna trials to our Trial format."""
        history = []

        for t in trials:
            if t.state != TrialState.COMPLETE or t.value is None:
                continue

            params = {}
            for name in self.param_order:
                if name in t.params:
                    params[name] = t.params[name]

            if len(params) == len([n for n in self.param_order if n in search_space]):
                history.append(Trial(params=params, score=t.value))

        # Normalize scores to [0, 1]
        if history:
            scores = [t.score for t in history]
            min_s, max_s = min(scores), max(scores)
            if max_s - min_s > 1e-10:
                for t in history:
                    t.score = (t.score - min_s) / (max_s - min_s)
            else:
                for t in history:
                    t.score = 0.5

        return history

    @torch.no_grad()
    def _sample_param(
        self,
        history: List[Trial],
        param_name: str,
        distribution: BaseDistribution,
        params_so_far: Dict[str, Any],
    ) -> Any:
        """Sample a single parameter using the model."""

        # Encode context
        tokens = self.tokenizer.encode_for_inference(history, params_so_far)
        input_ids = torch.tensor([tokens], device=self.device)

        # Get probability distribution over next token with optional AMP
        if self.use_amp:
            with autocast(dtype=torch.float16):
                probs = self.model.get_token_probabilities(input_ids)
        else:
            probs = self.model.get_token_probabilities(input_ids)

        spec = self.tokenizer.param_specs.get(param_name)

        if isinstance(distribution, FloatDistribution):
            # Get valid numerical tokens
            valid_tokens = list(range(self.tokenizer.num_bins))

            # Extract probabilities for numerical tokens
            valid_probs = probs[valid_tokens].cpu().numpy()
            valid_probs = valid_probs / (valid_probs.sum() + 1e-10)

            # Apply temperature
            if self.temperature != 1.0:
                logits = np.log(valid_probs + 1e-10) / self.temperature
                valid_probs = np.exp(logits) / np.exp(logits).sum()

            # Sample token
            token_idx = np.random.choice(len(valid_tokens), p=valid_probs)
            token_id = valid_tokens[token_idx]

            # Decode to value
            if spec:
                value = self.tokenizer.numerical.decode(
                    token_id, spec.low, spec.high, spec.log_scale
                )
            else:
                value = self.tokenizer.numerical.decode(
                    token_id, distribution.low, distribution.high,
                    distribution.log
                )

            return np.clip(value, distribution.low, distribution.high)

        elif isinstance(distribution, IntDistribution):
            valid_tokens = list(range(self.tokenizer.num_bins))
            valid_probs = probs[valid_tokens].cpu().numpy()
            valid_probs = valid_probs / (valid_probs.sum() + 1e-10)

            if self.temperature != 1.0:
                logits = np.log(valid_probs + 1e-10) / self.temperature
                valid_probs = np.exp(logits) / np.exp(logits).sum()

            token_idx = np.random.choice(len(valid_tokens), p=valid_probs)
            token_id = valid_tokens[token_idx]

            if spec:
                value = self.tokenizer.numerical.decode(
                    token_id, spec.low, spec.high, False
                )
            else:
                value = self.tokenizer.numerical.decode(
                    token_id, float(distribution.low), float(distribution.high), False
                )

            return int(np.clip(round(value), distribution.low, distribution.high))

        elif isinstance(distribution, CategoricalDistribution):
            if spec and spec.is_categorical:
                valid_tokens = self.tokenizer.get_valid_tokens_for_param(param_name)
                valid_probs = probs[valid_tokens].cpu().numpy()
                valid_probs = valid_probs / (valid_probs.sum() + 1e-10)

                if self.temperature != 1.0:
                    logits = np.log(valid_probs + 1e-10) / self.temperature
                    valid_probs = np.exp(logits) / np.exp(logits).sum()

                token_idx = np.random.choice(len(valid_tokens), p=valid_probs)
                token_id = valid_tokens[token_idx]

                value = self.tokenizer.decode_predicted_value(token_id, param_name)
                if value in distribution.choices:
                    return value

            # Fallback to random choice
            return np.random.choice(distribution.choices)

        # Default fallback
        return self.fallback.sample_independent(None, None, param_name, distribution)


class HybridSampler(BaseSampler):
    """
    Hybrid sampler combining OptiFormer warm-start with TPE refinement.

    Uses OptiFormer for early trials, then switches to TPE.
    """

    def __init__(
        self,
        model: OptiFormer,
        tokenizer: SequenceTokenizer,
        device: str = 'cuda',
        optiformer_trials: int = 10,
        temperature: float = 0.8,
    ):
        """
        Initialize hybrid sampler.

        Args:
            model: Trained OptiFormer model
            tokenizer: Sequence tokenizer
            device: Device for inference
            optiformer_trials: Number of trials to use OptiFormer
            temperature: Sampling temperature
        """
        self.optiformer = OptiFormerSampler(
            model, tokenizer, device, temperature,
            min_trials_for_model=0,
            exploration_rate=0.0,
        )
        self.tpe = optuna.samplers.TPESampler()
        self.optiformer_trials = optiformer_trials

    def infer_relative_search_space(
        self,
        study: Study,
        trial: optuna.Trial,
    ) -> Dict[str, BaseDistribution]:
        """Return the search space."""
        return optuna.search_space.intersection_search_space(study)

    def sample_relative(
        self,
        study: Study,
        trial: optuna.Trial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        """Sample using appropriate sampler based on trial count."""
        n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])

        if n_complete < self.optiformer_trials:
            return self.optiformer.sample_relative(study, trial, search_space)
        else:
            return self.tpe.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: optuna.Trial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Fallback sampling."""
        n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])

        if n_complete < self.optiformer_trials:
            return self.optiformer.sample_independent(
                study, trial, param_name, param_distribution
            )
        else:
            return self.tpe.sample_independent(
                study, trial, param_name, param_distribution
            )
