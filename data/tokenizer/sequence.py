"""
Full Trajectory Sequence Tokenizer

Converts optimization trajectories to token sequences and back.

Sequence Format:
<BOS> <TARGET_REGRET_X>
<TRIAL_SEP> <PARAM_p1> [value1] <PARAM_p2> [value2] ... <SCORE> [score]
<TRIAL_SEP> <PARAM_p1> [value3] <PARAM_p2> [value4] ... <SCORE> [score]
...
<EOS>

For inference (predicting next config):
<BOS> <TARGET_REGRET_0>
<TRIAL_SEP> ... history ...
<TRIAL_SEP> <PARAM_p1>  <- model predicts from here
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import json

from .numerical import NumericalTokenizer, NumericalTokenizerConfig
from .vocabulary import Vocabulary, VocabularyConfig


@dataclass
class ParameterSpec:
    """Specification for a single hyperparameter."""
    name: str
    low: float
    high: float
    log_scale: bool = False
    is_categorical: bool = False
    is_integer: bool = False
    categories: Optional[List[str]] = None

    def __post_init__(self):
        if self.is_categorical and self.categories is None:
            raise ValueError(f"Categorical parameter '{self.name}' must have categories")


@dataclass
class Trial:
    """A single trial in an optimization trajectory."""
    params: Dict[str, Any]  # param_name -> value
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {'params': self.params, 'score': self.score}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Trial':
        return cls(params=d['params'], score=d['score'])


class SequenceTokenizer:
    """
    Tokenizer for full optimization trajectories.

    Converts sequences of (hyperparameters, score) trials into token IDs
    and back for training and inference.
    """

    def __init__(
        self,
        param_specs: List[ParameterSpec],
        num_bins: int = 1000,
        score_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialize the sequence tokenizer.

        Args:
            param_specs: List of parameter specifications
            num_bins: Number of bins for numerical quantization
            score_range: (min, max) range for score values
        """
        self.param_specs = {p.name: p for p in param_specs}
        self.param_order = [p.name for p in param_specs]  # Fixed order!
        self.num_bins = num_bins
        self.score_range = score_range

        # Initialize numerical tokenizer
        self.numerical = NumericalTokenizer(NumericalTokenizerConfig(num_bins=num_bins))

        # Initialize vocabulary
        self.vocab = Vocabulary(VocabularyConfig(num_bins=num_bins))

        # Register parameters
        for spec in param_specs:
            self.vocab.register_parameter(spec.name)
            if spec.is_categorical:
                self.vocab.register_categorical(spec.name, spec.categories)

    @property
    def vocab_size(self) -> int:
        """Get total vocabulary size."""
        return self.vocab.vocab_size

    def encode_trajectory(
        self,
        trials: List[Trial],
        target_regret: int = 0  # 0 = optimal, 1 = good, 2 = bad
    ) -> List[int]:
        """
        Encode a full trajectory to token IDs.

        Args:
            trials: List of Trial objects
            target_regret: Target regret level (0=optimal, 1=good, 2=bad)

        Returns:
            List of token IDs
        """
        tokens = []

        # BOS and target regret
        tokens.append(self.vocab.bos_token_id)
        tokens.append(self.vocab.get_target_regret_token(target_regret))

        for trial in trials:
            tokens.append(self.vocab.trial_sep_token_id)

            # Parameters in fixed order
            for param_name in self.param_order:
                spec = self.param_specs[param_name]
                tokens.append(self.vocab.get_param_token(param_name))

                if spec.is_categorical:
                    cat_value = trial.params[param_name]
                    tokens.append(self.vocab.get_categorical_token(param_name, cat_value))
                else:
                    value = trial.params[param_name]
                    token_id = self.numerical.encode(
                        value, spec.low, spec.high, spec.log_scale
                    )
                    tokens.append(token_id)

            # Score
            tokens.append(self.vocab.score_token_id)
            # Normalize score to [0, 1] assuming lower is better
            score_normalized = (trial.score - self.score_range[0]) / (
                self.score_range[1] - self.score_range[0]
            )
            score_normalized = np.clip(score_normalized, 0.0, 1.0)
            score_token = self.numerical.encode(score_normalized, 0.0, 1.0, False)
            tokens.append(score_token)

        tokens.append(self.vocab.eos_token_id)
        return tokens

    def decode_trajectory(self, token_ids: List[int]) -> List[Trial]:
        """
        Decode token IDs back to trials.

        Args:
            token_ids: List of token IDs

        Returns:
            List of Trial objects
        """
        trials = []
        current_params = {}
        current_param_name = None
        in_score = False
        expecting_value = False

        for tid in token_ids:
            # Skip BOS, EOS, and target regret tokens
            if tid == self.vocab.bos_token_id or tid == self.vocab.eos_token_id:
                continue
            if tid in [self.vocab.get_target_regret_token(i) for i in range(3)]:
                continue

            if tid == self.vocab.trial_sep_token_id:
                current_params = {}
                in_score = False
                current_param_name = None
                expecting_value = False

            elif self.vocab.is_param_token(tid):
                _, param_name = self.vocab.decode_token(tid)
                current_param_name = param_name
                in_score = False
                expecting_value = True

            elif tid == self.vocab.score_token_id:
                in_score = True
                current_param_name = None
                expecting_value = True

            elif expecting_value:
                if in_score:
                    # Decode score
                    if self.vocab.is_numerical_token(tid):
                        score_normalized = self.numerical.decode(tid, 0.0, 1.0, False)
                        score = score_normalized * (self.score_range[1] - self.score_range[0]) + self.score_range[0]
                        trials.append(Trial(params=current_params.copy(), score=score))
                    expecting_value = False
                elif current_param_name:
                    spec = self.param_specs[current_param_name]
                    if spec.is_categorical and self.vocab.is_categorical_token(tid):
                        _, (_, cat_value) = self.vocab.decode_token(tid)
                        current_params[current_param_name] = cat_value
                    elif self.vocab.is_numerical_token(tid):
                        value = self.numerical.decode(
                            tid, spec.low, spec.high, spec.log_scale
                        )
                        if spec.is_integer:
                            value = int(round(value))
                        current_params[current_param_name] = value
                    expecting_value = False

        return trials

    def encode_for_inference(
        self,
        history: List[Trial],
        params_so_far: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """
        Encode history and prepare for predicting next parameters.

        Returns tokens for the context, stopping before the first parameter
        value that needs to be predicted.

        Args:
            history: List of completed trials
            params_so_far: Dict of parameters already sampled in current trial

        Returns:
            List of token IDs
        """
        tokens = []
        tokens.append(self.vocab.bos_token_id)
        tokens.append(self.vocab.get_target_regret_token(0))  # We want optimal

        # Encode history
        for trial in history:
            tokens.append(self.vocab.trial_sep_token_id)
            for param_name in self.param_order:
                spec = self.param_specs[param_name]
                tokens.append(self.vocab.get_param_token(param_name))

                if spec.is_categorical:
                    tokens.append(self.vocab.get_categorical_token(
                        param_name, trial.params[param_name]
                    ))
                else:
                    tokens.append(self.numerical.encode(
                        trial.params[param_name], spec.low, spec.high, spec.log_scale
                    ))

            tokens.append(self.vocab.score_token_id)
            score_normalized = (trial.score - self.score_range[0]) / (
                self.score_range[1] - self.score_range[0]
            )
            tokens.append(self.numerical.encode(
                np.clip(score_normalized, 0, 1), 0.0, 1.0, False
            ))

        # Start new trial
        tokens.append(self.vocab.trial_sep_token_id)

        # Add params we've already sampled (if any)
        params_so_far = params_so_far or {}
        for param_name in self.param_order:
            tokens.append(self.vocab.get_param_token(param_name))
            if param_name in params_so_far:
                spec = self.param_specs[param_name]
                if spec.is_categorical:
                    tokens.append(self.vocab.get_categorical_token(
                        param_name, params_so_far[param_name]
                    ))
                else:
                    tokens.append(self.numerical.encode(
                        params_so_far[param_name], spec.low, spec.high, spec.log_scale
                    ))
            else:
                # Stop here - model predicts this value
                break

        return tokens

    def decode_predicted_value(
        self,
        token_id: int,
        param_name: str
    ) -> Any:
        """
        Decode a predicted token ID to a parameter value.

        Args:
            token_id: Predicted token ID
            param_name: Name of the parameter being predicted

        Returns:
            Decoded parameter value
        """
        spec = self.param_specs[param_name]

        if spec.is_categorical:
            if self.vocab.is_categorical_token(token_id):
                _, (_, cat_value) = self.vocab.decode_token(token_id)
                return cat_value
            else:
                # Fallback to first category
                return spec.categories[0]
        else:
            if self.vocab.is_numerical_token(token_id):
                value = self.numerical.decode(
                    token_id, spec.low, spec.high, spec.log_scale
                )
                if spec.is_integer:
                    value = int(np.clip(round(value), spec.low, spec.high))
                return value
            else:
                # Fallback to midpoint
                return (spec.low + spec.high) / 2

    def get_valid_tokens_for_param(self, param_name: str) -> List[int]:
        """Get list of valid token IDs for a parameter."""
        spec = self.param_specs[param_name]
        if spec.is_categorical:
            return [
                self.vocab.get_categorical_token(param_name, cat)
                for cat in spec.categories
            ]
        else:
            # All numerical tokens are valid
            return list(range(self.num_bins))

    def save(self, path: str):
        """Save tokenizer configuration to file."""
        config = {
            'param_specs': [
                {
                    'name': spec.name,
                    'low': spec.low,
                    'high': spec.high,
                    'log_scale': spec.log_scale,
                    'is_categorical': spec.is_categorical,
                    'is_integer': spec.is_integer,
                    'categories': spec.categories,
                }
                for spec in self.param_specs.values()
            ],
            'num_bins': self.num_bins,
            'score_range': self.score_range,
            'vocab': self.vocab.to_dict(),
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SequenceTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            config = json.load(f)

        param_specs = [
            ParameterSpec(**spec_dict)
            for spec_dict in config['param_specs']
        ]

        tokenizer = cls(
            param_specs=param_specs,
            num_bins=config['num_bins'],
            score_range=tuple(config['score_range']),
        )

        return tokenizer
