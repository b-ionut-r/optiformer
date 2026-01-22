"""
Smoke Test Phase 3: Model Training Validation

Trains a small OptiFormer model and validates that training
works correctly (loss decreases).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import json

from model import OptiFormer, OptiFormerConfig
from data.tokenizer import SequenceTokenizer, ParameterSpec
from data.datasets import TrajectoryDataset
from training import OptiFormerTrainer, TrainingConfig


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory usage information.

    Returns:
        Dict with memory info in MB, or empty dict if CUDA not available.
    """
    if not torch.cuda.is_available():
        return {}

    try:
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        # Get total GPU memory
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024 / 1024  # MB

        return {
            'allocated_mb': round(allocated, 2),
            'reserved_mb': round(reserved, 2),
            'max_allocated_mb': round(max_allocated, 2),
            'total_mb': round(total, 2),
            'utilization_pct': round(100 * max_allocated / total, 1),
        }
    except Exception:
        return {}


def create_default_param_specs(n_dims: int = 20) -> list:
    """
    Create default parameter specifications for optimization.

    IMPORTANT: n_dims must be large enough to cover ALL data sources:
    - Synthetic functions: typically 2-10 dims
    - HPOBench FCNet: 9 params
    - YAHPO XGBoost: 14 params
    - JAHS-Bench: 14 params

    Using 20 dims provides headroom for most benchmarks.
    Unused dimensions are simply not present in shorter trajectories.
    """
    return [
        ParameterSpec(f"x{i}", 0.0, 1.0, log_scale=False)
        for i in range(n_dims)
    ]


def train_smoke_test_model(
    data_dir: Path,
    output_dir: Path,
    max_steps: int = 2000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    n_dims: int = 20,  # Must cover all benchmark dimensions (HPOBench=9, YAHPO=14, JAHS=14)
    device: str = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Train a small model for smoke testing.

    Args:
        data_dir: Directory containing trajectory data
        output_dir: Directory for model outputs
        max_steps: Maximum training steps
        batch_size: Training batch size
        learning_rate: Learning rate
        n_dims: Number of input dimensions
        device: Device to train on

    Returns:
        (success, results_dict)
    """
    print("\nPhase 3: Model Training")
    print("-" * 40)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        'initial_loss': None,
        'final_loss': None,
        'best_val_loss': None,
        'loss_reduction': 0.0,
        'total_steps': 0,
        'errors': [],
        'gpu_memory': {},
    }

    # Reset GPU memory stats for accurate tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        # Create tokenizer
        print("  Creating tokenizer...")
        param_specs = create_default_param_specs(n_dims)
        tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

        # Save tokenizer
        tokenizer.save(str(output_dir / "tokenizer.json"))

        # Create datasets
        print("  Loading datasets...")
        train_path = data_dir / "train_trajectories.jsonl"
        val_path = data_dir / "val_trajectories.jsonl"

        if not train_path.exists():
            results['errors'].append(f"Training data not found: {train_path}")
            return False, results

        train_dataset = TrajectoryDataset(
            data_path=train_path,
            tokenizer=tokenizer,
            max_length=256,
            shuffle_history=True,
        )

        val_dataset = TrajectoryDataset(
            data_path=val_path,
            tokenizer=tokenizer,
            max_length=256,
            shuffle_history=False,
        )

        print(f"    Train samples: {len(train_dataset)}")
        print(f"    Val samples: {len(val_dataset)}")

        if len(train_dataset) == 0:
            results['errors'].append("Empty training dataset")
            return False, results

        # Create model
        print("  Creating model...")
        config = OptiFormerConfig.nano()
        config.vocab_size = tokenizer.vocab_size + 100  # Safety margin
        model = OptiFormer(config)

        print(f"    Model size: {model.n_params:,} parameters")
        print(f"    Vocab size: {config.vocab_size}")

        # Profile GPU memory after model creation
        if torch.cuda.is_available():
            model.to(device)
            mem_after_model = get_gpu_memory_info()
            results['gpu_memory']['after_model_creation'] = mem_after_model
            print(f"    GPU memory (model): {mem_after_model.get('allocated_mb', 0):.1f} MB")

        # Create trainer
        print(f"  Training for {max_steps} steps on {device}...")
        train_config = TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            warmup_steps=min(100, max_steps // 10),
            logging_steps=max(10, max_steps // 50),
            eval_steps=max(100, max_steps // 10),
            save_steps=max(500, max_steps // 4),
            early_stopping_patience=5,
            device=device,
            fp16=torch.cuda.is_available(),
        )

        trainer = OptiFormerTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=train_config,
            output_dir=output_dir,
            pad_token_id=tokenizer.vocab.pad_token_id,
        )

        # Train
        training_results = trainer.train()

        results['initial_loss'] = training_results['initial_train_loss']
        results['final_loss'] = training_results['final_train_loss']
        results['best_val_loss'] = training_results['best_val_loss']
        results['loss_reduction'] = training_results['loss_reduction']
        results['total_steps'] = training_results['total_steps']

        # Profile GPU memory after training
        if torch.cuda.is_available():
            mem_after_training = get_gpu_memory_info()
            results['gpu_memory']['after_training'] = mem_after_training
            results['gpu_memory']['peak_mb'] = mem_after_training.get('max_allocated_mb', 0)
            results['gpu_memory']['utilization_pct'] = mem_after_training.get('utilization_pct', 0)

        print(f"\n  Training Results:")
        print(f"    Initial loss: {results['initial_loss']:.4f}")
        print(f"    Final loss: {results['final_loss']:.4f}")
        print(f"    Best val loss: {results['best_val_loss']:.4f}")
        print(f"    Loss reduction: {results['loss_reduction']:.1%}")

        # Print GPU memory summary
        if results['gpu_memory']:
            peak = results['gpu_memory'].get('peak_mb', 0)
            util = results['gpu_memory'].get('utilization_pct', 0)
            print(f"    Peak GPU memory: {peak:.1f} MB ({util:.1f}% of total)")

        # Check success criteria (matches smoke_test.yaml threshold)
        success = results['loss_reduction'] > 0.6  # At least 60% reduction

        if not success:
            results['errors'].append(f"Insufficient loss reduction: {results['loss_reduction']:.1%}")

        return success, results

    except Exception as e:
        import traceback
        results['errors'].append(str(e))
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False, results


if __name__ == "__main__":
    data_dir = Path("./outputs/smoke_test/data")
    output_dir = Path("./outputs/smoke_test/model")

    success, results = train_smoke_test_model(
        data_dir=data_dir,
        output_dir=output_dir,
        max_steps=500,
        batch_size=32,
    )

    print(f"\nTraining: {'SUCCESS' if success else 'FAILED'}")
    if results['errors']:
        for err in results['errors']:
            print(f"  Error: {err}")

    sys.exit(0 if success else 1)
