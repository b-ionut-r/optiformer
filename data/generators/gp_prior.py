"""
Gaussian Process Prior Function Sampler

Generates random smooth functions by sampling from GP priors.
Different kernels create different smoothness properties.

This provides a distribution over smooth functions for training,
ensuring the model learns to optimize smooth landscapes.
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator

from .base import BaseFunctionGenerator, FunctionGeneratorConfig, FunctionWrapper


@dataclass
class GPConfig(FunctionGeneratorConfig):
    """Configuration for GP prior function generator."""
    kernel: str = "rbf"  # rbf, matern_1.5, matern_2.5
    lengthscale_mean: float = 0.0
    lengthscale_std: float = 1.0
    output_scale: float = 1.0
    n_support_points: int = 500  # Points for function representation
    interpolation_method: str = "rbf"  # rbf, linear


class GPKernels:
    """Collection of GP kernel functions."""

    @staticmethod
    def rbf(X1: np.ndarray, X2: np.ndarray, lengthscale: float) -> np.ndarray:
        """Radial Basis Function (squared exponential) kernel."""
        dists = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-0.5 * dists / (lengthscale ** 2))

    @staticmethod
    def matern_1_5(X1: np.ndarray, X2: np.ndarray, lengthscale: float) -> np.ndarray:
        """Matern 3/2 kernel - less smooth than RBF."""
        dists = cdist(X1, X2, metric='euclidean')
        sqrt3 = np.sqrt(3.0)
        scaled = sqrt3 * dists / lengthscale
        return (1 + scaled) * np.exp(-scaled)

    @staticmethod
    def matern_2_5(X1: np.ndarray, X2: np.ndarray, lengthscale: float) -> np.ndarray:
        """Matern 5/2 kernel - intermediate smoothness."""
        dists = cdist(X1, X2, metric='euclidean')
        sqrt5 = np.sqrt(5.0)
        scaled = sqrt5 * dists / lengthscale
        return (1 + scaled + scaled**2 / 3) * np.exp(-scaled)


class GPPriorSampler(BaseFunctionGenerator):
    """
    Samples random functions from Gaussian Process priors.

    Uses kernel covariance matrices to generate smooth random functions.
    The lengthscale controls how "wiggly" the function is.
    """

    def __init__(self, config: GPConfig = None):
        config = config or GPConfig()
        super().__init__(config)
        self.gp_config = config
        self.kernel_fn = self._get_kernel_fn(config.kernel)

    def _get_kernel_fn(self, kernel_name: str) -> Callable:
        """Get kernel function by name."""
        kernels = {
            'rbf': GPKernels.rbf,
            'matern_1.5': GPKernels.matern_1_5,
            'matern_2.5': GPKernels.matern_2_5,
        }
        if kernel_name not in kernels:
            raise ValueError(f"Unknown kernel: {kernel_name}. Available: {list(kernels.keys())}")
        return kernels[kernel_name]

    def _sample_lengthscale(self) -> float:
        """Sample a random lengthscale from log-normal distribution."""
        return np.exp(np.random.normal(
            self.gp_config.lengthscale_mean,
            self.gp_config.lengthscale_std
        ))

    def generate(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Generate a random function sampled from a GP prior.

        Returns:
            Callable that takes (n_points, n_dims) and returns (n_points,)
        """
        # Sample hyperparameters
        lengthscale = self._sample_lengthscale()

        # Generate support points
        low, high = self.bounds
        n_points = self.gp_config.n_support_points
        X_support = np.random.uniform(low, high, size=(n_points, self.n_dims))

        # Compute kernel matrix
        K = self.kernel_fn(X_support, X_support, lengthscale)
        K = K * self.gp_config.output_scale

        # Add jitter for numerical stability
        K += 1e-6 * np.eye(n_points)

        # Sample from multivariate normal via Cholesky
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Fall back to eigendecomposition if Cholesky fails
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        z = np.random.randn(n_points)
        y_support = L @ z

        # Normalize to [0, 1]
        y_min, y_max = y_support.min(), y_support.max()
        if y_max - y_min > 1e-10:
            y_support = (y_support - y_min) / (y_max - y_min)
        else:
            y_support = np.full_like(y_support, 0.5)

        # Create interpolator
        interpolator = RBFInterpolator(X_support, y_support, kernel='thin_plate_spline')

        def func(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            y = interpolator(x)
            return np.clip(y, 0, 1)

        return func

    def generate_with_metadata(self) -> FunctionWrapper:
        """Generate function with metadata."""
        func = self.generate()
        return FunctionWrapper(
            func=func,
            n_dims=self.n_dims,
            bounds=self.bounds,
            name=f"gp_{self.gp_config.kernel}",
        )


def generate_gp_functions(
    n_functions: int,
    n_dims: int,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    kernels: List[str] = None,
    seed: Optional[int] = None,
) -> List[Callable]:
    """
    Generate multiple GP-sampled functions with varied kernels.

    Args:
        n_functions: Number of functions to generate
        n_dims: Dimensionality of input space
        bounds: (low, high) bounds for each dimension
        kernels: List of kernel names to cycle through
        seed: Random seed

    Returns:
        List of callable functions
    """
    if seed is not None:
        np.random.seed(seed)

    kernels = kernels or ["rbf", "matern_1.5", "matern_2.5"]
    functions = []

    for i in range(n_functions):
        kernel = kernels[i % len(kernels)]
        config = GPConfig(
            n_dims=n_dims,
            bounds=bounds,
            kernel=kernel,
            lengthscale_mean=np.random.uniform(-1, 1),
            lengthscale_std=0.5,
        )
        sampler = GPPriorSampler(config)
        func = sampler.generate()
        functions.append(func)

    return functions


def generate_gp_dataset(
    n_functions: int,
    dims_list: List[int],
    bounds: Tuple[float, float] = (-5.0, 5.0),
    kernels: List[str] = None,
    seed: Optional[int] = None,
) -> List[Tuple[int, Callable]]:
    """
    Generate GP functions with varying dimensionality.

    Args:
        n_functions: Total number of functions to generate
        dims_list: List of possible dimensions
        bounds: Input bounds
        kernels: Kernel types to use
        seed: Random seed

    Returns:
        List of (n_dims, function) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    kernels = kernels or ["rbf", "matern_1.5", "matern_2.5"]
    functions = []
    funcs_per_dim = n_functions // len(dims_list)

    for n_dims in dims_list:
        dim_funcs = generate_gp_functions(
            n_functions=funcs_per_dim,
            n_dims=n_dims,
            bounds=bounds,
            kernels=kernels,
        )
        for f in dim_funcs:
            functions.append((n_dims, f))

    # Add any remaining functions
    remaining = n_functions - len(functions)
    if remaining > 0:
        n_dims = dims_list[-1]
        extra_funcs = generate_gp_functions(
            n_functions=remaining,
            n_dims=n_dims,
            bounds=bounds,
            kernels=kernels,
        )
        for f in extra_funcs:
            functions.append((n_dims, f))

    return functions
