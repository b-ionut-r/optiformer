"""
Standard Optimization Test Functions

Well-known benchmark functions for evaluating optimization algorithms.
All functions are designed for minimization.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class BenchmarkFunction:
    """Container for benchmark function with metadata."""
    name: str
    fn: callable
    bounds: Tuple[float, float]
    optimum: float
    optimal_point: Optional[np.ndarray] = None
    dimensions: int = 2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fn(x)


def sphere(x: np.ndarray) -> np.ndarray:
    """
    Sphere function - simplest convex function.

    f(x) = sum(x_i^2)
    Global minimum: f(0, ..., 0) = 0

    Easy to optimize - useful as sanity check.
    """
    x = np.atleast_2d(x)
    return np.sum(x**2, axis=1)


def rastrigin(x: np.ndarray, A: float = 10.0) -> np.ndarray:
    """
    Rastrigin function - highly multimodal.

    f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    Global minimum: f(0, ..., 0) = 0

    Has many local minima in a regular pattern.
    Tests global optimization capability.
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Rosenbrock function - banana-shaped valley.

    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum: f(1, ..., 1) = 0

    The global minimum lies in a narrow, curved valley.
    """
    x = np.atleast_2d(x)
    return np.sum(
        100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2,
        axis=1
    )


def ackley(x: np.ndarray) -> np.ndarray:
    """
    Ackley function - many local minima.

    Global minimum: f(0, ..., 0) = 0

    Nearly flat outer region with a large hole at the center.
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    sum1 = np.sum(x**2, axis=1)
    sum2 = np.sum(np.cos(2 * np.pi * x), axis=1)
    return (
        -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
        - np.exp(sum2 / n)
        + 20 + np.e
    )


def levy(x: np.ndarray) -> np.ndarray:
    """
    Levy function - complex multimodal.

    Global minimum: f(1, ..., 1) = 0
    """
    x = np.atleast_2d(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[:, 0])**2
    term2 = np.sum(
        (w[:, :-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1)**2),
        axis=1
    )
    term3 = (w[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, -1])**2)
    return term1 + term2 + term3


def branin(x: np.ndarray) -> np.ndarray:
    """
    Branin function (2D only).

    Three global minima:
    f(-pi, 12.275) = f(pi, 2.275) = f(9.42478, 2.475) ≈ 0.397887

    Classic 2D benchmark with multiple global minima.
    """
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]

    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    return (
        a * (x2 - b * x1**2 + c * x1 - r)**2
        + s * (1 - t) * np.cos(x1)
        + s
    )


def griewank(x: np.ndarray) -> np.ndarray:
    """
    Griewank function.

    Global minimum: f(0, ..., 0) = 0

    Many local minima, but easier than Rastrigin.
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    sum_sq = np.sum(x**2, axis=1) / 4000
    prod_cos = np.prod(
        np.cos(x / np.sqrt(np.arange(1, n + 1))),
        axis=1
    )
    return sum_sq - prod_cos + 1


def schwefel(x: np.ndarray) -> np.ndarray:
    """
    Schwefel function.

    Global minimum: f(420.9687, ..., 420.9687) ≈ 0

    Global minimum is far from next best local minima.
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def michalewicz(x: np.ndarray, m: float = 10.0) -> np.ndarray:
    """
    Michalewicz function.

    Has d! local minima where d is the dimension.
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    i = np.arange(1, n + 1)
    return -np.sum(
        np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * m),
        axis=1
    )


# Benchmark registry
BENCHMARKS: Dict[str, BenchmarkFunction] = {
    'sphere': BenchmarkFunction(
        name='sphere',
        fn=sphere,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        optimal_point=None,  # Origin
    ),
    'rastrigin': BenchmarkFunction(
        name='rastrigin',
        fn=rastrigin,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        optimal_point=None,  # Origin
    ),
    'rosenbrock': BenchmarkFunction(
        name='rosenbrock',
        fn=rosenbrock,
        bounds=(-5.0, 10.0),
        optimum=0.0,
        optimal_point=None,  # All ones
    ),
    'ackley': BenchmarkFunction(
        name='ackley',
        fn=ackley,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        optimal_point=None,  # Origin
    ),
    'levy': BenchmarkFunction(
        name='levy',
        fn=levy,
        bounds=(-10.0, 10.0),
        optimum=0.0,
        optimal_point=None,  # All ones
    ),
    'branin': BenchmarkFunction(
        name='branin',
        fn=branin,
        bounds=(-5.0, 15.0),  # Different bounds for each dim typically
        optimum=0.397887,
        dimensions=2,
    ),
    'griewank': BenchmarkFunction(
        name='griewank',
        fn=griewank,
        bounds=(-600.0, 600.0),
        optimum=0.0,
    ),
    'schwefel': BenchmarkFunction(
        name='schwefel',
        fn=schwefel,
        bounds=(-500.0, 500.0),
        optimum=0.0,
    ),
}


def get_benchmark(name: str, n_dims: int = 2) -> BenchmarkFunction:
    """
    Get a benchmark function by name.

    Args:
        name: Benchmark name
        n_dims: Number of dimensions

    Returns:
        BenchmarkFunction instance
    """
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARKS.keys())}")

    bench = BENCHMARKS[name]
    bench.dimensions = n_dims

    # Set optimal point based on function
    if name in ['sphere', 'rastrigin', 'ackley']:
        bench.optimal_point = np.zeros(n_dims)
    elif name in ['rosenbrock', 'levy']:
        bench.optimal_point = np.ones(n_dims)

    return bench


def normalize_function(fn: callable, n_samples: int = 10000, bounds: Tuple[float, float] = (-5, 5), n_dims: int = 2):
    """
    Create a normalized version of a function that maps to [0, 1].

    Useful for consistent evaluation across different functions.
    """
    # Sample to estimate range
    low, high = bounds
    X = np.random.uniform(low, high, size=(n_samples, n_dims))
    y = fn(X)
    y_min, y_max = y.min(), y.max()

    def normalized_fn(x):
        x = np.atleast_2d(x)
        raw = fn(x)
        if y_max - y_min > 1e-10:
            return (raw - y_min) / (y_max - y_min)
        return np.full(x.shape[0], 0.5)

    return normalized_fn, y_min, y_max
