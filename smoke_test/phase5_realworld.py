"""
Smoke Test Phase 5: Real-World ML Evaluation

Evaluates the trained model on real ML hyperparameter optimization tasks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from model import OptiFormer
from data.tokenizer import SequenceTokenizer
from samplers import OptiFormerSampler
from evaluation import run_ml_evaluation
from evaluation.ml_benchmarks import MNISTBenchmark, FashionMNISTBenchmark


def evaluate_realworld(
    model_path: Path,
    output_dir: Path,
    n_trials: int = 20,
    n_seeds: int = 2,
    epochs_per_trial: int = 3,
    device: str = None,
    run_mnist: bool = True,
    run_fashion: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate trained model on real ML benchmarks.

    Args:
        model_path: Path to trained model checkpoint
        output_dir: Directory for results
        n_trials: Trials per optimization run
        n_seeds: Number of random seeds
        epochs_per_trial: Epochs for each ML training
        device: Device for training/inference
        run_mnist: Whether to run MNIST benchmark
        run_fashion: Whether to run Fashion-MNIST benchmark

    Returns:
        (success, results_dict)
    """
    print("\nPhase 5: Real-World ML Evaluation")
    print("-" * 40)

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        'benchmarks': {},
        'vs_random_win_rate': 0.0,
        'errors': [],
    }

    try:
        # Load model
        print(f"  Loading model from {model_path}...")
        if not model_path.exists():
            results['errors'].append(f"Model not found: {model_path}")
            return False, results

        model = OptiFormer.load(str(model_path), device=device)
        model.eval()

        all_win_rates = []

        # MNIST evaluation
        if run_mnist:
            print("\n  Evaluating on MNIST MLP...")
            mnist_benchmark = MNISTBenchmark(
                data_dir='./data',
                epochs_per_trial=epochs_per_trial,
                subset_size=5000,  # Small subset for smoke test
                device=device,
            )

            # Create tokenizer from benchmark specs
            tokenizer = SequenceTokenizer(mnist_benchmark.param_specs, num_bins=1000)

            # Create sampler
            sampler = OptiFormerSampler(
                model=model,
                tokenizer=tokenizer,
                device=device,
                temperature=0.8,
                min_trials_for_model=2,
                exploration_rate=0.15,
            )

            # Run evaluation
            mnist_results = run_ml_evaluation(
                optiformer_sampler=sampler,
                benchmark=mnist_benchmark,
                n_trials=n_trials,
                n_seeds=n_seeds,
                output_dir=output_dir,
            )

            mnist_stats = mnist_results['statistics']
            results['benchmarks']['mnist_mlp'] = {
                'optiformer_best': mnist_stats['optiformer_best'],
                'optiformer_mean': mnist_stats['optiformer_mean'],
                'tpe_best': mnist_stats['tpe_best'],
                'tpe_mean': mnist_stats['tpe_mean'],
                'random_best': mnist_stats['random_best'],
                'random_mean': mnist_stats['random_mean'],
                'vs_random_win_rate': mnist_stats['vs_random_win_rate'],
                'vs_tpe_win_rate': mnist_stats['vs_tpe_win_rate'],
            }
            all_win_rates.append(mnist_stats['vs_random_win_rate'])

            print(f"    MNIST Results:")
            print(f"      OptiFormer best error: {mnist_stats['optiformer_best']:.4f}")
            print(f"      TPE best error: {mnist_stats['tpe_best']:.4f}")
            print(f"      Random best error: {mnist_stats['random_best']:.4f}")
            print(f"      vs Random win rate: {mnist_stats['vs_random_win_rate']:.1%}")

        # Fashion-MNIST evaluation
        if run_fashion:
            print("\n  Evaluating on Fashion-MNIST MLP...")
            fashion_benchmark = FashionMNISTBenchmark(
                data_dir='./data',
                epochs_per_trial=epochs_per_trial,
                subset_size=5000,
                device=device,
            )

            tokenizer = SequenceTokenizer(fashion_benchmark.param_specs, num_bins=1000)
            sampler = OptiFormerSampler(
                model=model,
                tokenizer=tokenizer,
                device=device,
                temperature=0.8,
                min_trials_for_model=2,
            )

            fashion_results = run_ml_evaluation(
                optiformer_sampler=sampler,
                benchmark=fashion_benchmark,
                n_trials=n_trials,
                n_seeds=n_seeds,
                output_dir=output_dir,
            )

            fashion_stats = fashion_results['statistics']
            results['benchmarks']['fashion_mlp'] = {
                'optiformer_best': fashion_stats['optiformer_best'],
                'tpe_best': fashion_stats['tpe_best'],
                'random_best': fashion_stats['random_best'],
                'vs_random_win_rate': fashion_stats['vs_random_win_rate'],
            }
            all_win_rates.append(fashion_stats['vs_random_win_rate'])

        # Compute overall win rate
        if all_win_rates:
            results['vs_random_win_rate'] = np.mean(all_win_rates)

        # Determine success
        success = results['vs_random_win_rate'] > 0.4  # Beat random >40% on ML

        if not success:
            results['errors'].append(f"Win rate vs random too low: {results['vs_random_win_rate']:.1%}")

        return success, results

    except Exception as e:
        import traceback
        results['errors'].append(str(e))
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False, results


if __name__ == "__main__":
    model_path = Path("./outputs/smoke_test/model/best.pt")
    output_dir = Path("./outputs/smoke_test/eval_realworld")

    success, results = evaluate_realworld(
        model_path=model_path,
        output_dir=output_dir,
        n_trials=15,
        n_seeds=2,
        epochs_per_trial=2,
        run_mnist=True,
        run_fashion=False,
    )

    print(f"\nReal-world evaluation: {'SUCCESS' if success else 'FAILED'}")
    print(f"  vs Random: {results['vs_random_win_rate']:.1%}")

    sys.exit(0 if success else 1)
