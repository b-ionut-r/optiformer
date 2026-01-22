"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def seed():
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_seeds(seed):
    """Set random seeds before each test."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def sample_param_specs():
    """Sample parameter specifications for testing."""
    from data.tokenizer import ParameterSpec

    return [
        ParameterSpec("learning_rate", 1e-5, 1e-1, log_scale=True),
        ParameterSpec("batch_size", 16, 256, is_integer=True),
        ParameterSpec("dropout", 0.0, 0.5),
        ParameterSpec("optimizer", 0, 1, is_categorical=True,
                     categories=["adam", "sgd", "rmsprop"]),
    ]


@pytest.fixture
def sample_trials(sample_param_specs):
    """Sample trials for testing."""
    from data.tokenizer import Trial

    return [
        Trial(
            params={
                "learning_rate": 0.01,
                "batch_size": 64,
                "dropout": 0.2,
                "optimizer": "adam",
            },
            score=0.15,
        ),
        Trial(
            params={
                "learning_rate": 0.001,
                "batch_size": 128,
                "dropout": 0.1,
                "optimizer": "sgd",
            },
            score=0.12,
        ),
        Trial(
            params={
                "learning_rate": 0.0001,
                "batch_size": 32,
                "dropout": 0.3,
                "optimizer": "rmsprop",
            },
            score=0.18,
        ),
    ]
