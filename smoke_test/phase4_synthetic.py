"""
Smoke Test Phase 4: Synthetic Benchmark Evaluation

Evaluates the trained model on synthetic optimization benchmarks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from model import OptiFormer
from data.tokenizer import SequenceTokenizer, ParameterSpec
from samplers import OptiFormerSampler
from evaluation import run_synthetic_evaluation, print_evaluation_summary


def create_synthetic_param_specs(n_dims: int = 2, bounds: tuple = (-5.0, 5.0)) -> list:
    """Create parameter specs for synthetic benchmark evaluation."""
    return [
        ParameterSpec(f"x{i}", bounds[0], bounds[1], log_scale=False)
        for i in range(n_dims)
    ]


def evaluate_synthetic(
    model_path: Path,
    output_dir: Path,
    n_dims: int = 2,
    dims_list: list = None,
    n_trials: int = 30,
    n_seeds: int = 3,
    device: str = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate trained model on synthetic benchmarks.

    Args:
        model_path: Path to trained model checkpoint
        output_dir: Directory for results
        n_dims: Dimensionality for benchmarks (used if dims_list is None)
        dims_list: List of dimensions to test (e.g., [2, 3, 5] to match training)
        n_trials: Trials per optimization run
        n_seeds: Number of random seeds
        device: Device for inference

    Returns:
        (success, results_dict)
    """
    print("\nPhase 4: Synthetic Benchmark Evaluation")
    print("-" * 40)

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use dims_list if provided, otherwise single dimension
    if dims_list is None:
        dims_list = [n_dims]

    results = {
        'vs_random_win_rate': 0.0,
        'vs_tpe_win_rate': 0.0,
        'benchmarks': {},
        'by_dimension': {},
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

        benchmarks = ['sphere', 'rastrigin', 'rosenbrock', 'ackley', 'levy']

        # Aggregate wins across all dimensions
        total_vs_random_wins = 0
        total_vs_tpe_wins = 0
        total_comparisons = 0

        for dim in dims_list:
            print(f"\n  Testing {dim}D benchmarks...")

            # Create tokenizer for this dimension
            param_specs = create_synthetic_param_specs(dim)
            tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

            # Create sampler
            sampler = OptiFormerSampler(
                model=model,
                tokenizer=tokenizer,
                device=device,
                temperature=0.8,
                min_trials_for_model=2,
                exploration_rate=0.1,
            )

            # Run evaluation for this dimension
            dim_output_dir = output_dir / f"dim_{dim}"
            dim_output_dir.mkdir(parents=True, exist_ok=True)

            eval_results = run_synthetic_evaluation(
                optiformer_sampler=sampler,
                benchmark_names=benchmarks,
                n_dims=dim,
                n_trials=n_trials,
                n_seeds=n_seeds,
                output_dir=dim_output_dir,
            )

            stats = eval_results['statistics']

            # Store per-dimension results
            dim_vs_random = stats.get('overall', {}).get('vs_random_win_rate', 0)
            dim_vs_tpe = stats.get('overall', {}).get('vs_tpe_win_rate', 0)

            results['by_dimension'][dim] = {
                'vs_random_win_rate': dim_vs_random,
                'vs_tpe_win_rate': dim_vs_tpe,
                'benchmarks': {b: stats.get(b, {}) for b in benchmarks},
            }

            # Accumulate for overall statistics
            n_bench = len(benchmarks) * n_seeds
            total_vs_random_wins += dim_vs_random * n_bench
            total_vs_tpe_wins += dim_vs_tpe * n_bench
            total_comparisons += n_bench

            print(f"    {dim}D vs Random: {dim_vs_random:.1%}, vs TPE: {dim_vs_tpe:.1%}")

        # Compute overall win rates across all dimensions
        if total_comparisons > 0:
            results['vs_random_win_rate'] = total_vs_random_wins / total_comparisons
            results['vs_tpe_win_rate'] = total_vs_tpe_wins / total_comparisons

        # Collect all benchmark results
        for dim_results in results['by_dimension'].values():
            for bench, bench_stats in dim_results['benchmarks'].items():
                if bench not in results['benchmarks']:
                    results['benchmarks'][bench] = []
                results['benchmarks'][bench].append(bench_stats)

        # Print summary
        print(f"\n  Overall Results (across {len(dims_list)} dimensions):")
        print(f"    vs Random: {results['vs_random_win_rate']:.1%}")
        print(f"    vs TPE: {results['vs_tpe_win_rate']:.1%}")

        # Determine success
        success = results['vs_random_win_rate'] > 0.5  # Beat random >50% of time

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
    output_dir = Path("./outputs/smoke_test/eval_synthetic")

    # Test multiple dimensions to match training (use subset for speed)
    success, results = evaluate_synthetic(
        model_path=model_path,
        output_dir=output_dir,
        dims_list=[2, 3, 5],  # Match subset of training dimensions
        n_trials=20,
        n_seeds=2,
    )

    print(f"\nSynthetic evaluation: {'SUCCESS' if success else 'FAILED'}")
    print(f"  vs Random: {results['vs_random_win_rate']:.1%}")
    print(f"  vs TPE: {results['vs_tpe_win_rate']:.1%}")
    if results.get('by_dimension'):
        print("  By dimension:")
        for dim, dim_res in results['by_dimension'].items():
            print(f"    {dim}D: vs Random {dim_res['vs_random_win_rate']:.1%}")

    sys.exit(0 if success else 1)
