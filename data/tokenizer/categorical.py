"""
Categorical Tokenizer

Handles mapping of categorical (string) values to token IDs and back.
Each parameter can have its own set of categories.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class CategoricalTokenizerConfig:
    """Configuration for categorical tokenizer."""
    max_categories_per_param: int = 50
    unknown_token: str = "<UNK>"


class CategoricalTokenizer:
    """
    Tokenizer for categorical (string-valued) parameters.

    Maintains a mapping from category values to token IDs for each parameter.
    """

    def __init__(self, config: CategoricalTokenizerConfig = None):
        self.config = config or CategoricalTokenizerConfig()
        # param_name -> {category_value -> token_id}
        self._param_to_category_ids: Dict[str, Dict[str, int]] = {}
        # param_name -> {token_id -> category_value}
        self._param_to_id_categories: Dict[str, Dict[int, str]] = {}
        # Track next available token ID
        self._next_token_id: int = 0

    def register_parameter(
        self,
        param_name: str,
        categories: List[str],
        start_token_id: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Register a categorical parameter with its possible values.

        Args:
            param_name: Name of the parameter
            categories: List of possible category values
            start_token_id: Starting token ID (if None, uses internal counter)

        Returns:
            Dict mapping category values to token IDs
        """
        if len(categories) > self.config.max_categories_per_param:
            raise ValueError(
                f"Parameter '{param_name}' has {len(categories)} categories, "
                f"max allowed is {self.config.max_categories_per_param}"
            )

        if start_token_id is not None:
            current_id = start_token_id
        else:
            current_id = self._next_token_id

        category_to_id = {}
        id_to_category = {}

        for category in categories:
            category_to_id[category] = current_id
            id_to_category[current_id] = category
            current_id += 1

        self._param_to_category_ids[param_name] = category_to_id
        self._param_to_id_categories[param_name] = id_to_category

        if start_token_id is None:
            self._next_token_id = current_id

        return category_to_id

    def encode(self, param_name: str, value: str) -> int:
        """
        Encode a categorical value to a token ID.

        Args:
            param_name: Name of the parameter
            value: Category value to encode

        Returns:
            Token ID

        Raises:
            KeyError: If parameter or value is not registered
        """
        if param_name not in self._param_to_category_ids:
            raise KeyError(f"Parameter '{param_name}' not registered")

        category_map = self._param_to_category_ids[param_name]
        if value not in category_map:
            raise KeyError(
                f"Category '{value}' not registered for parameter '{param_name}'. "
                f"Valid categories: {list(category_map.keys())}"
            )

        return category_map[value]

    def decode(self, param_name: str, token_id: int) -> str:
        """
        Decode a token ID back to a categorical value.

        Args:
            param_name: Name of the parameter
            token_id: Token ID to decode

        Returns:
            Category value

        Raises:
            KeyError: If parameter or token_id is not valid
        """
        if param_name not in self._param_to_id_categories:
            raise KeyError(f"Parameter '{param_name}' not registered")

        id_map = self._param_to_id_categories[param_name]
        if token_id not in id_map:
            raise KeyError(
                f"Token ID {token_id} not valid for parameter '{param_name}'. "
                f"Valid IDs: {list(id_map.keys())}"
            )

        return id_map[token_id]

    def get_categories(self, param_name: str) -> List[str]:
        """Get list of categories for a parameter."""
        if param_name not in self._param_to_category_ids:
            raise KeyError(f"Parameter '{param_name}' not registered")
        return list(self._param_to_category_ids[param_name].keys())

    def get_token_ids(self, param_name: str) -> List[int]:
        """Get list of token IDs for a parameter's categories."""
        if param_name not in self._param_to_category_ids:
            raise KeyError(f"Parameter '{param_name}' not registered")
        return list(self._param_to_category_ids[param_name].values())

    def is_valid_token(self, param_name: str, token_id: int) -> bool:
        """Check if a token ID is valid for a parameter."""
        if param_name not in self._param_to_id_categories:
            return False
        return token_id in self._param_to_id_categories[param_name]

    def get_vocab_size(self) -> int:
        """Get total number of categorical tokens registered."""
        total = 0
        for categories in self._param_to_category_ids.values():
            total += len(categories)
        return total

    @property
    def next_token_id(self) -> int:
        """Get the next available token ID."""
        return self._next_token_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tokenizer state to dict."""
        return {
            'param_to_category_ids': self._param_to_category_ids,
            'param_to_id_categories': {
                p: {str(k): v for k, v in m.items()}
                for p, m in self._param_to_id_categories.items()
            },
            'next_token_id': self._next_token_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: CategoricalTokenizerConfig = None) -> 'CategoricalTokenizer':
        """Deserialize tokenizer from dict."""
        tokenizer = cls(config)
        tokenizer._param_to_category_ids = data['param_to_category_ids']
        tokenizer._param_to_id_categories = {
            p: {int(k): v for k, v in m.items()}
            for p, m in data['param_to_id_categories'].items()
        }
        tokenizer._next_token_id = data['next_token_id']
        return tokenizer
