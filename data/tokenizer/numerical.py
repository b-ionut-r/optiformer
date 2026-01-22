"""
Quantized Numerical Tokenizer

Converts continuous float values to discrete tokens and back.
This is the MOST CRITICAL component - errors here silently break everything.

Strategy:
1. Normalize value to [0, 1] based on known range
2. Optionally apply log transform for log-scale parameters
3. Quantize to one of 1000 bins
4. Return token ID (0-999)

Reverse:
1. Token ID -> bin center (0-999 -> 0.0005, 0.0015, ..., 0.9995)
2. Reverse log transform if applicable
3. Denormalize to original range
"""

import numpy as np
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass


@dataclass
class NumericalTokenizerConfig:
    """Configuration for numerical tokenizer."""
    num_bins: int = 1000
    epsilon: float = 1e-10  # For numerical stability


class NumericalTokenizer:
    """
    Converts continuous float values to discrete tokens and back.

    Uses uniform quantization with optional log-space transformation
    for parameters that span multiple orders of magnitude (e.g., learning rate).
    """

    def __init__(self, config: NumericalTokenizerConfig = None):
        self.config = config or NumericalTokenizerConfig()
        self.num_bins = self.config.num_bins
        self.epsilon = self.config.epsilon

    def encode(
        self,
        value: Union[float, np.ndarray],
        low: float,
        high: float,
        log_scale: bool = False
    ) -> Union[int, np.ndarray]:
        """
        Encode a float value to a token ID.

        Args:
            value: The value to encode (scalar or array)
            low: Minimum of the parameter range
            high: Maximum of the parameter range
            log_scale: If True, apply log transform before binning

        Returns:
            Token ID in range [0, num_bins-1]
        """
        # Handle scalar vs array
        is_scalar = np.isscalar(value)
        value = np.atleast_1d(np.asarray(value, dtype=np.float64))

        # Clamp to valid range
        value = np.clip(value, low, high)

        if log_scale:
            # Log transform: map [low, high] to [log(low), log(high)]
            # then normalize to [0, 1]
            log_low = np.log(max(low, self.epsilon))
            log_high = np.log(max(high, self.epsilon))
            log_value = np.log(np.maximum(value, self.epsilon))

            if log_high - log_low > self.epsilon:
                normalized = (log_value - log_low) / (log_high - log_low)
            else:
                normalized = np.zeros_like(value)
        else:
            # Linear normalization
            if high - low > self.epsilon:
                normalized = (value - low) / (high - low)
            else:
                normalized = np.zeros_like(value)

        # Clamp to [0, 1] (safety)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Quantize to bin
        # Maps [0, 1] to [0, num_bins-1]
        token_id = (normalized * (self.num_bins - 1)).astype(np.int32)
        token_id = np.clip(token_id, 0, self.num_bins - 1)

        if is_scalar:
            return int(token_id[0])
        return token_id

    def decode(
        self,
        token_id: Union[int, np.ndarray],
        low: float,
        high: float,
        log_scale: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Decode a token ID back to a float value.

        Returns the center of the bin.
        """
        # Handle scalar vs array
        is_scalar = np.isscalar(token_id)
        token_id = np.atleast_1d(np.asarray(token_id, dtype=np.float64))

        # Token to normalized value (bin center)
        normalized = (token_id + 0.5) / self.num_bins

        if log_scale:
            # Reverse log transform
            log_low = np.log(max(low, self.epsilon))
            log_high = np.log(max(high, self.epsilon))
            log_value = normalized * (log_high - log_low) + log_low
            value = np.exp(log_value)
            # Clamp to valid range
            value = np.clip(value, low, high)
        else:
            # Linear denormalization
            value = normalized * (high - low) + low

        if is_scalar:
            return float(value[0])
        return value

    def roundtrip_error(
        self,
        value: float,
        low: float,
        high: float,
        log_scale: bool = False
    ) -> float:
        """
        Calculate the error introduced by encode->decode roundtrip.

        Returns relative error normalized by the parameter range.
        """
        token = self.encode(value, low, high, log_scale)
        recovered = self.decode(token, low, high, log_scale)

        if high - low > self.epsilon:
            return abs(value - recovered) / (high - low)
        return 0.0

    def encode_batch(
        self,
        values: np.ndarray,
        low: float,
        high: float,
        log_scale: bool = False
    ) -> np.ndarray:
        """Encode a batch of values."""
        return self.encode(values, low, high, log_scale)

    def decode_batch(
        self,
        token_ids: np.ndarray,
        low: float,
        high: float,
        log_scale: bool = False
    ) -> np.ndarray:
        """Decode a batch of token IDs."""
        return self.decode(token_ids, low, high, log_scale)

    def get_bin_edges(
        self,
        low: float,
        high: float,
        log_scale: bool = False
    ) -> np.ndarray:
        """
        Get the bin edges for visualization/debugging.

        Returns array of shape (num_bins + 1,) with bin boundaries.
        """
        edges_normalized = np.linspace(0, 1, self.num_bins + 1)

        if log_scale:
            log_low = np.log(max(low, self.epsilon))
            log_high = np.log(max(high, self.epsilon))
            log_edges = edges_normalized * (log_high - log_low) + log_low
            return np.exp(log_edges)
        else:
            return edges_normalized * (high - low) + low

    def get_bin_centers(
        self,
        low: float,
        high: float,
        log_scale: bool = False
    ) -> np.ndarray:
        """
        Get the bin centers (decoded values for each token ID).

        Returns array of shape (num_bins,).
        """
        token_ids = np.arange(self.num_bins)
        return self.decode(token_ids, low, high, log_scale)


class IntegerTokenizer:
    """
    Tokenizer for integer-valued parameters.

    Handles the additional complexity of mapping continuous bins
    to discrete integer values.
    """

    def __init__(self, num_tokenizer: NumericalTokenizer):
        self.num_tokenizer = num_tokenizer

    def encode(
        self,
        value: int,
        low: int,
        high: int,
        log_scale: bool = False
    ) -> int:
        """Encode an integer value to a token ID."""
        # Treat as continuous then quantize
        return self.num_tokenizer.encode(float(value), float(low), float(high), log_scale)

    def decode(
        self,
        token_id: int,
        low: int,
        high: int,
        log_scale: bool = False
    ) -> int:
        """Decode a token ID back to an integer value."""
        continuous = self.num_tokenizer.decode(token_id, float(low), float(high), log_scale)
        # Round and clamp to valid integer range
        return int(np.clip(round(continuous), low, high))
