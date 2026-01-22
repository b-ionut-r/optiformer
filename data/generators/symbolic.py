"""
Random Symbolic Expression Generator

Creates random mathematical expressions for non-smooth, jagged landscapes.
These are crucial for bridging the sim-to-real gap, as real hyperparameter
landscapes often have discontinuities and non-smooth regions.
"""

import numpy as np
import random
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass, field

from .base import BaseFunctionGenerator, FunctionGeneratorConfig, FunctionWrapper


@dataclass
class SymbolicConfig(FunctionGeneratorConfig):
    """Configuration for symbolic function generator."""
    operators: List[str] = field(default_factory=lambda: ['+', '-', '*', 'sin', 'cos', 'exp', 'abs'])
    max_depth: int = 5
    min_depth: int = 2
    constant_range: Tuple[float, float] = (-2.0, 2.0)
    variable_prob: float = 0.7  # Probability of choosing variable over constant


class ExpressionNode:
    """Node in expression tree."""

    def __init__(
        self,
        op: Optional[str] = None,
        value: Optional[float] = None,
        var_idx: Optional[int] = None,
        left: Optional['ExpressionNode'] = None,
        right: Optional['ExpressionNode'] = None,
    ):
        self.op = op
        self.value = value
        self.var_idx = var_idx
        self.left = left
        self.right = right

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate expression at given points."""
        if self.value is not None:
            return np.full(x.shape[0], self.value)

        if self.var_idx is not None:
            return x[:, self.var_idx]

        # Evaluate children
        if self.op in ['sin', 'cos', 'exp', 'abs', 'sqrt', 'log']:
            # Unary operators
            child_val = self.left.evaluate(x)
            if self.op == 'sin':
                return np.sin(child_val)
            elif self.op == 'cos':
                return np.cos(child_val)
            elif self.op == 'exp':
                return np.exp(np.clip(child_val, -10, 10))
            elif self.op == 'abs':
                return np.abs(child_val)
            elif self.op == 'sqrt':
                return np.sqrt(np.abs(child_val))
            elif self.op == 'log':
                return np.log(np.abs(child_val) + 1e-10)
        else:
            # Binary operators
            left_val = self.left.evaluate(x)
            right_val = self.right.evaluate(x)

            if self.op == '+':
                return left_val + right_val
            elif self.op == '-':
                return left_val - right_val
            elif self.op == '*':
                return left_val * right_val
            elif self.op == '/':
                return left_val / (np.abs(right_val) + 0.1)
            elif self.op == 'pow':
                return np.power(np.abs(left_val) + 0.1, np.clip(right_val, -2, 2))

        return np.zeros(x.shape[0])

    def __str__(self) -> str:
        """String representation of expression."""
        if self.value is not None:
            return f"{self.value:.3f}"
        if self.var_idx is not None:
            return f"x[{self.var_idx}]"

        if self.op in ['sin', 'cos', 'exp', 'abs', 'sqrt', 'log']:
            return f"{self.op}({self.left})"
        else:
            return f"({self.left} {self.op} {self.right})"


class SymbolicGenerator(BaseFunctionGenerator):
    """Generates random symbolic mathematical expressions."""

    UNARY_OPS = {'sin', 'cos', 'exp', 'abs', 'sqrt', 'log'}
    BINARY_OPS = {'+', '-', '*', '/', 'pow'}

    def __init__(self, config: SymbolicConfig = None):
        config = config or SymbolicConfig()
        super().__init__(config)
        self.sym_config = config

        # Separate operators by arity
        self.unary_ops = [op for op in config.operators if op in self.UNARY_OPS]
        self.binary_ops = [op for op in config.operators if op in self.BINARY_OPS]

    def _generate_tree(self, depth: int = 0) -> ExpressionNode:
        """Recursively generate expression tree."""
        # Terminal condition
        if depth >= self.sym_config.max_depth or (
            depth >= self.sym_config.min_depth and random.random() < 0.3
        ):
            # Return a variable or constant
            if random.random() < self.sym_config.variable_prob:
                var_idx = random.randint(0, self.n_dims - 1)
                return ExpressionNode(var_idx=var_idx)
            else:
                const = random.uniform(*self.sym_config.constant_range)
                return ExpressionNode(value=const)

        # Choose operator type
        if self.unary_ops and self.binary_ops:
            use_unary = random.random() < 0.3
        elif self.unary_ops:
            use_unary = True
        else:
            use_unary = False

        if use_unary:
            op = random.choice(self.unary_ops)
            left = self._generate_tree(depth + 1)
            return ExpressionNode(op=op, left=left)
        else:
            op = random.choice(self.binary_ops) if self.binary_ops else '+'
            left = self._generate_tree(depth + 1)
            right = self._generate_tree(depth + 1)
            return ExpressionNode(op=op, left=left, right=right)

    def generate(self) -> Callable[[np.ndarray], np.ndarray]:
        """Generate a random symbolic function."""
        tree = self._generate_tree()

        def func(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            try:
                y = tree.evaluate(x)
                # Handle infinities and NaN
                y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)

                # Normalize to [0, 1]
                y_min, y_max = y.min(), y.max()
                if y_max - y_min > 1e-10:
                    y = (y - y_min) / (y_max - y_min)
                else:
                    y = np.full_like(y, 0.5)

                return np.clip(y, 0, 1)
            except Exception:
                return np.full(x.shape[0], 0.5)

        return func

    def generate_with_expression(self) -> Tuple[Callable, str]:
        """Generate function and return its string representation."""
        tree = self._generate_tree()

        def func(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            try:
                y = tree.evaluate(x)
                y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
                y_min, y_max = y.min(), y.max()
                if y_max - y_min > 1e-10:
                    y = (y - y_min) / (y_max - y_min)
                else:
                    y = np.full_like(y, 0.5)
                return np.clip(y, 0, 1)
            except Exception:
                return np.full(x.shape[0], 0.5)

        return func, str(tree)


def generate_symbolic_functions(
    n_functions: int,
    n_dims: int,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    config: SymbolicConfig = None,
    seed: Optional[int] = None,
) -> List[Callable]:
    """
    Generate multiple symbolic functions.

    Args:
        n_functions: Number of functions to generate
        n_dims: Dimensionality of input space
        bounds: (low, high) bounds for each dimension
        config: Optional configuration
        seed: Random seed

    Returns:
        List of callable functions
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    config = config or SymbolicConfig()
    config.n_dims = n_dims
    config.bounds = bounds

    functions = []
    gen = SymbolicGenerator(config)

    for _ in range(n_functions):
        func = gen.generate()
        functions.append(func)

    return functions


def generate_symbolic_dataset(
    n_functions: int,
    dims_list: List[int],
    bounds: Tuple[float, float] = (-5.0, 5.0),
    config: SymbolicConfig = None,
    seed: Optional[int] = None,
) -> List[Tuple[int, Callable]]:
    """
    Generate symbolic functions with varying dimensionality.

    Args:
        n_functions: Total number of functions to generate
        dims_list: List of possible dimensions
        bounds: Input bounds
        config: Optional configuration
        seed: Random seed

    Returns:
        List of (n_dims, function) tuples
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    config = config or SymbolicConfig()
    functions = []
    funcs_per_dim = n_functions // len(dims_list)

    for n_dims in dims_list:
        dim_funcs = generate_symbolic_functions(
            n_functions=funcs_per_dim,
            n_dims=n_dims,
            bounds=bounds,
            config=config,
        )
        for f in dim_funcs:
            functions.append((n_dims, f))

    # Add any remaining functions
    remaining = n_functions - len(functions)
    if remaining > 0:
        n_dims = dims_list[-1]
        extra_funcs = generate_symbolic_functions(
            n_functions=remaining,
            n_dims=n_dims,
            bounds=bounds,
            config=config,
        )
        for f in extra_funcs:
            functions.append((n_dims, f))

    return functions
