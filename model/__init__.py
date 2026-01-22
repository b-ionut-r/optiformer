"""
OptiFormer Model Module

Provides the OptiFormer transformer model for hyperparameter optimization.
"""

from .config import OptiFormerConfig
from .optiformer import OptiFormer, create_model
from .generation import HyperparameterGenerator, GenerationConfig, EnsembleGenerator

__all__ = [
    'OptiFormerConfig',
    'OptiFormer',
    'create_model',
    'HyperparameterGenerator',
    'GenerationConfig',
    'EnsembleGenerator',
]
