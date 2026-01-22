"""
OptiFormer Configuration Module

Provides configuration loading and management utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_config_path(name: str = "smoke_test") -> Path:
    """Get path to a named configuration file."""
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config '{name}' not found at {config_path}")

    return config_path


def load_named_config(name: str = "smoke_test") -> Dict[str, Any]:
    """Load a named configuration (smoke_test, base, full_training)."""
    config_path = get_config_path(name)
    return load_config(str(config_path))


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    num_bins: int = 1000
    max_categorical: int = 100
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        'pad': 0,
        'bos': 1,
        'eos': 2,
        'trial_sep': 3,
        'score': 4,
        'target_regret_0': 5,
        'target_regret_1': 6,
        'target_regret_2': 7,
    })
    param_token_start: int = 1100

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TokenizerConfig':
        return cls(
            num_bins=d.get('num_bins', 1000),
            max_categorical=d.get('max_categorical', 100),
            special_tokens=d.get('special_tokens', cls.__dataclass_fields__['special_tokens'].default_factory()),
            param_token_start=d.get('param_token_start', 1100),
        )


@dataclass
class DataConfig:
    """Data generation configuration."""
    gp_functions: int = 2000
    symbolic_functions: int = 2000
    trials_per_function: int = 64
    dimensions: list = field(default_factory=lambda: [2, 3, 5])
    noise_std: float = 0.02

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataConfig':
        synthetic = d.get('synthetic', {})
        return cls(
            gp_functions=synthetic.get('gp_functions', 2000),
            symbolic_functions=synthetic.get('symbolic_functions', 2000),
            trials_per_function=synthetic.get('trials_per_function', 64),
            dimensions=synthetic.get('dimensions', [2, 3, 5]),
            noise_std=synthetic.get('noise_std', 0.02),
        )


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_steps: int = 5000
    warmup_steps: int = 200
    lr_scheduler: str = "cosine"
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        early_stopping = d.get('early_stopping', {})
        return cls(
            batch_size=d.get('batch_size', 128),
            gradient_accumulation_steps=d.get('gradient_accumulation_steps', 1),
            learning_rate=d.get('learning_rate', 3e-4),
            weight_decay=d.get('weight_decay', 0.01),
            max_grad_norm=d.get('max_grad_norm', 1.0),
            max_steps=d.get('max_steps', 5000),
            warmup_steps=d.get('warmup_steps', 200),
            lr_scheduler=d.get('lr_scheduler', 'cosine'),
            fp16=d.get('fp16', True),
            bf16=d.get('bf16', False),
            logging_steps=d.get('logging_steps', 50),
            eval_steps=d.get('eval_steps', 500),
            save_steps=d.get('save_steps', 1000),
            early_stopping_patience=early_stopping.get('patience', 3),
            early_stopping_min_delta=early_stopping.get('min_delta', 0.001),
        )


__all__ = [
    'load_config',
    'get_config_path',
    'load_named_config',
    'TokenizerConfig',
    'DataConfig',
    'TrainingConfig',
]
