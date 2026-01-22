"""
OptiFormer Smoke Test Module

End-to-end validation of the OptiFormer pipeline.
"""

from .phase1_tokenizer import test_tokenizer
from .phase2_data import generate_smoke_test_data
from .phase3_training import train_smoke_test_model
from .phase4_synthetic import evaluate_synthetic
from .phase5_realworld import evaluate_realworld
from .run_all import run_smoke_test

__all__ = [
    'test_tokenizer',
    'generate_smoke_test_data',
    'train_smoke_test_model',
    'evaluate_synthetic',
    'evaluate_realworld',
    'run_smoke_test',
]
