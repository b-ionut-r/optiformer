"""
Vocabulary Management

Manages the complete token vocabulary including:
- Numerical tokens (0 to num_bins-1)
- Special tokens (BOS, EOS, etc.)
- Parameter name tokens
- Categorical value tokens
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum


class SpecialTokens(IntEnum):
    """Special token IDs."""
    PAD = 0
    BOS = 1
    EOS = 2
    TRIAL_SEP = 3
    SCORE = 4
    TARGET_REGRET_0 = 5
    TARGET_REGRET_1 = 6
    TARGET_REGRET_2 = 7


@dataclass
class VocabularyConfig:
    """Configuration for vocabulary."""
    num_bins: int = 1000
    num_special_tokens: int = 8
    max_params: int = 50
    max_categorical_tokens: int = 200


class Vocabulary:
    """
    Manages the complete token vocabulary for OptiFormer.

    Token ID layout:
    - [0, num_bins-1]: Numerical value tokens
    - [num_bins, num_bins + num_special - 1]: Special tokens
    - [special_end, special_end + max_params - 1]: Parameter name tokens
    - [param_end, ...]: Categorical value tokens
    """

    def __init__(self, config: VocabularyConfig = None):
        self.config = config or VocabularyConfig()

        # Calculate token ranges
        self.num_bins = self.config.num_bins
        self.special_start = self.num_bins
        self.special_end = self.special_start + self.config.num_special_tokens
        self.param_start = self.special_end
        self.param_end = self.param_start + self.config.max_params
        self.categorical_start = self.param_end

        # Token mappings
        self._special_tokens: Dict[str, int] = {}
        self._special_ids: Dict[int, str] = {}
        self._param_tokens: Dict[str, int] = {}
        self._param_ids: Dict[int, str] = {}
        self._categorical_tokens: Dict[str, Dict[str, int]] = {}
        self._categorical_ids: Dict[int, tuple] = {}  # id -> (param_name, value)

        self._next_categorical_id = self.categorical_start

        # Initialize special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize standard special tokens."""
        special_names = [
            "<PAD>", "<BOS>", "<EOS>", "<TRIAL_SEP>", "<SCORE>",
            "<TARGET_REGRET_0>", "<TARGET_REGRET_1>", "<TARGET_REGRET_2>"
        ]
        for i, name in enumerate(special_names):
            token_id = self.special_start + i
            self._special_tokens[name] = token_id
            self._special_ids[token_id] = name

    def register_parameter(self, param_name: str) -> int:
        """
        Register a parameter name and return its token ID.

        Args:
            param_name: Name of the parameter

        Returns:
            Token ID for the parameter name
        """
        if param_name in self._param_tokens:
            return self._param_tokens[param_name]

        # Assign next available param token ID
        param_idx = len(self._param_tokens)
        if param_idx >= self.config.max_params:
            raise ValueError(f"Maximum number of parameters ({self.config.max_params}) exceeded")

        token_id = self.param_start + param_idx
        self._param_tokens[param_name] = token_id
        self._param_ids[token_id] = param_name

        return token_id

    def register_categorical(self, param_name: str, categories: List[str]) -> Dict[str, int]:
        """
        Register categorical values for a parameter.

        Args:
            param_name: Name of the parameter
            categories: List of possible category values

        Returns:
            Dict mapping category values to token IDs
        """
        if param_name not in self._categorical_tokens:
            self._categorical_tokens[param_name] = {}

        result = {}
        for category in categories:
            if category in self._categorical_tokens[param_name]:
                result[category] = self._categorical_tokens[param_name][category]
            else:
                token_id = self._next_categorical_id
                self._categorical_tokens[param_name][category] = token_id
                self._categorical_ids[token_id] = (param_name, category)
                self._next_categorical_id += 1
                result[category] = token_id

        return result

    def get_special_token(self, name: str) -> int:
        """Get token ID for a special token by name."""
        if name not in self._special_tokens:
            raise KeyError(f"Unknown special token: {name}")
        return self._special_tokens[name]

    def get_param_token(self, param_name: str) -> int:
        """Get token ID for a parameter name."""
        if param_name not in self._param_tokens:
            raise KeyError(f"Parameter '{param_name}' not registered")
        return self._param_tokens[param_name]

    def get_categorical_token(self, param_name: str, value: str) -> int:
        """Get token ID for a categorical value."""
        if param_name not in self._categorical_tokens:
            raise KeyError(f"Parameter '{param_name}' has no registered categories")
        if value not in self._categorical_tokens[param_name]:
            raise KeyError(f"Category '{value}' not registered for '{param_name}'")
        return self._categorical_tokens[param_name][value]

    def decode_token(self, token_id: int) -> tuple:
        """
        Decode a token ID to its type and value.

        Returns:
            Tuple of (token_type, value) where token_type is one of:
            - 'numerical': value is the token_id (0 to num_bins-1)
            - 'special': value is the special token name
            - 'param': value is the parameter name
            - 'categorical': value is (param_name, category_value)
        """
        if token_id < self.num_bins:
            return ('numerical', token_id)
        elif token_id in self._special_ids:
            return ('special', self._special_ids[token_id])
        elif token_id in self._param_ids:
            return ('param', self._param_ids[token_id])
        elif token_id in self._categorical_ids:
            return ('categorical', self._categorical_ids[token_id])
        else:
            raise ValueError(f"Unknown token ID: {token_id}")

    def is_numerical_token(self, token_id: int) -> bool:
        """Check if token ID represents a numerical value."""
        return 0 <= token_id < self.num_bins

    def is_special_token(self, token_id: int) -> bool:
        """Check if token ID is a special token."""
        return token_id in self._special_ids

    def is_param_token(self, token_id: int) -> bool:
        """Check if token ID is a parameter name token."""
        return token_id in self._param_ids

    def is_categorical_token(self, token_id: int) -> bool:
        """Check if token ID is a categorical value token."""
        return token_id in self._categorical_ids

    @property
    def vocab_size(self) -> int:
        """Get total vocabulary size."""
        return self._next_categorical_id

    @property
    def pad_token_id(self) -> int:
        return self._special_tokens["<PAD>"]

    @property
    def bos_token_id(self) -> int:
        return self._special_tokens["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return self._special_tokens["<EOS>"]

    @property
    def trial_sep_token_id(self) -> int:
        return self._special_tokens["<TRIAL_SEP>"]

    @property
    def score_token_id(self) -> int:
        return self._special_tokens["<SCORE>"]

    def get_target_regret_token(self, level: int) -> int:
        """Get token ID for target regret level (0, 1, or 2)."""
        if level not in [0, 1, 2]:
            raise ValueError(f"Invalid target regret level: {level}")
        return self._special_tokens[f"<TARGET_REGRET_{level}>"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize vocabulary to dict."""
        return {
            'config': {
                'num_bins': self.config.num_bins,
                'num_special_tokens': self.config.num_special_tokens,
                'max_params': self.config.max_params,
                'max_categorical_tokens': self.config.max_categorical_tokens,
            },
            'param_tokens': self._param_tokens,
            'categorical_tokens': self._categorical_tokens,
            'next_categorical_id': self._next_categorical_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vocabulary':
        """Deserialize vocabulary from dict."""
        config = VocabularyConfig(**data['config'])
        vocab = cls(config)

        # Restore parameter tokens
        for param_name in data['param_tokens']:
            vocab.register_parameter(param_name)

        # Restore categorical tokens
        for param_name, categories in data['categorical_tokens'].items():
            vocab.register_categorical(param_name, list(categories.keys()))

        return vocab
