"""
OptiFormer Evaluation Module

Benchmarks and evaluation utilities.
"""

from . import synthetic_benchmarks
from . import ml_benchmarks
from .evaluate import (
    EvaluationResult,
    evaluate_on_function,
    run_synthetic_evaluation,
    run_ml_evaluation,
    compute_statistics,
    print_evaluation_summary,
)

__all__ = [
    'synthetic_benchmarks',
    'ml_benchmarks',
    'EvaluationResult',
    'evaluate_on_function',
    'run_synthetic_evaluation',
    'run_ml_evaluation',
    'compute_statistics',
    'print_evaluation_summary',
]
