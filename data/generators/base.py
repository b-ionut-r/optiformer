"""
Base Class for Function Generators

Defines the interface for generating synthetic objective functions
for training the OptiFormer model.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class FunctionGeneratorConfig:
    """Base configuration for function generators."""
    n_dims: int = 2
    bounds: Tuple[float, float] = (-5.0, 5.0)
    seed: Optional[int] = None


class BaseFunctionGenerator(ABC):
    """
    Abstract base class for synthetic function generators.

    All generators should produce callable objective functions
    that map from R^n -> R, suitable for optimization.
    """

    def __init__(self, config: FunctionGeneratorConfig):
        self.config = config
        self.n_dims = config.n_dims
        self.bounds = config.bounds

        if config.seed is not None:
            np.random.seed(config.seed)

    @abstractmethod
    def generate(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Generate a new random objective function.

        Returns:
            Callable that takes (n_points, n_dims) array and returns (n_points,) array
        """
        pass

    def generate_batch(self, n_functions: int) -> List[Callable]:
        """Generate multiple functions."""
        return [self.generate() for _ in range(n_functions)]

    def evaluate(
        self,
        func: Callable,
        x: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Evaluate function at given points.

        Args:
            func: The objective function
            x: Points to evaluate, shape (n_points, n_dims) or (n_dims,)
            normalize: If True, normalize outputs to [0, 1]

        Returns:
            Function values, shape (n_points,) or scalar
        """
        x = np.atleast_2d(x)
        y = func(x)

        if normalize:
            # Normalize to [0, 1]
            y_min, y_max = y.min(), y.max()
            if y_max - y_min > 1e-10:
                y = (y - y_min) / (y_max - y_min)
            else:
                y = np.full_like(y, 0.5)

        return y

    def sample_random_points(self, n_points: int) -> np.ndarray:
        """Sample random points within bounds."""
        low, high = self.bounds
        return np.random.uniform(low, high, size=(n_points, self.n_dims))

    def get_info(self) -> Dict[str, Any]:
        """Get generator information."""
        return {
            'type': self.__class__.__name__,
            'n_dims': self.n_dims,
            'bounds': self.bounds,
        }


class FunctionWrapper:
    """
    Wrapper for objective functions that adds metadata and normalization.
    """

    def __init__(
        self,
        func: Callable,
        n_dims: int,
        bounds: Tuple[float, float],
        name: str = "unnamed",
        optimal_value: Optional[float] = None,
        optimal_point: Optional[np.ndarray] = None,
    ):
        self._func = func
        self.n_dims = n_dims
        self.bounds = bounds
        self.name = name
        self.optimal_value = optimal_value
        self.optimal_point = optimal_point

        # For normalization
        self._y_min = None
        self._y_max = None
        self._normalize = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function."""
        x = np.atleast_2d(x)
        y = self._func(x)

        if self._normalize and self._y_min is not None:
            y = (y - self._y_min) / (self._y_max - self._y_min + 1e-10)
            y = np.clip(y, 0, 1)

        return y.flatten() if y.shape[0] == 1 else y

    def calibrate_normalization(self, n_samples: int = 10000):
        """
        Calibrate normalization by sampling the function.

        This estimates the output range for proper normalization.
        """
        low, high = self.bounds
        X = np.random.uniform(low, high, size=(n_samples, self.n_dims))
        y = self._func(X)
        self._y_min = y.min()
        self._y_max = y.max()
        self._normalize = True

    def get_regret(self, y: float) -> float:
        """Calculate regret relative to optimal value."""
        if self.optimal_value is not None:
            return y - self.optimal_value
        return y  # Assume optimal is 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dict."""
        return {
            'name': self.name,
            'n_dims': self.n_dims,
            'bounds': self.bounds,
            'optimal_value': self.optimal_value,
            'optimal_point': self.optimal_point.tolist() if self.optimal_point is not None else None,
        }
