"""
OptiFormer Data Generators

Provides synthetic function generation and trajectory creation.
"""

from .base import BaseFunctionGenerator, FunctionGeneratorConfig, FunctionWrapper
from .gp_prior import (
    GPPriorSampler,
    GPConfig,
    GPKernels,
    generate_gp_functions,
    generate_gp_dataset,
)
from .symbolic import (
    SymbolicGenerator,
    SymbolicConfig,
    ExpressionNode,
    generate_symbolic_functions,
    generate_symbolic_dataset,
)
from .trajectory import (
    TrajectoryGenerator,
    TrajectoryConfig,
    generate_all_trajectories,
    load_trajectories,
    DataAugmenter,
)

__all__ = [
    # Base
    'BaseFunctionGenerator',
    'FunctionGeneratorConfig',
    'FunctionWrapper',
    # GP Prior
    'GPPriorSampler',
    'GPConfig',
    'GPKernels',
    'generate_gp_functions',
    'generate_gp_dataset',
    # Symbolic
    'SymbolicGenerator',
    'SymbolicConfig',
    'ExpressionNode',
    'generate_symbolic_functions',
    'generate_symbolic_dataset',
    # Trajectory
    'TrajectoryGenerator',
    'TrajectoryConfig',
    'generate_all_trajectories',
    'load_trajectories',
    'DataAugmenter',
]
