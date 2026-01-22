"""
Smoke Test Phase 6: Optuna Integration Test

Validates end-to-end integration with Optuna by running the model
as a sampler in an actual optimization study.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from model import OptiFormer
from data.tokenizer import SequenceTokenizer, ParameterSpec
from samplers import OptiFormerSampler


def create_param_specs_for_optuna(n_dims: int = 3) -> List[ParameterSpec]:
    """Create parameter specs matching Optuna search space."""
    return [
        ParameterSpec(f"x{i}", 0.0, 5.0, log_scale=False)
        for i in range(n_dims)
    ]


def quadratic_objective(trial: optuna.Trial, n_dims: int = 3) -> float:
    """
    Simple quadratic objective: f(x) = sum((x_i - 2)^2)

    Optimum is at x_i = 2 for all i, with f(x*) = 0.
    """
    total = 0.0
    for i in range(n_dims):
        x = trial.suggest_float(f"x{i}", 0.0, 5.0)
        total += (x - 2.0) ** 2
    return total


def test_optuna_integration(
    model_path: Path,
    n_dims: int = 3,
    n_trials: int = 15,
    device: str = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test OptiFormer as an Optuna sampler in a real optimization study.

    Args:
        model_path: Path to trained model
        n_dims: Number of dimensions for test objective
        n_trials: Number of optimization trials to run
        device: Device for inference

    Returns:
        (success, results_dict)
    """
    print("\nPhase 6: Optuna Integration Test")
    print("-" * 40)

    model_path = Path(model_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        'study_completed': False,
        'n_trials': 0,
        'best_value': None,
        'best_params': None,
        'all_suggestions_valid': True,
        'suggestions_adapted': False,
        'trial_history': [],
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

        # Create tokenizer matching Optuna search space
        param_specs = create_param_specs_for_optuna(n_dims)
        tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

        # Create OptiFormer sampler
        print("  Creating OptiFormerSampler...")
        sampler = OptiFormerSampler(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=0.8,
            min_trials_for_model=2,  # Use model after 2 trials
            exploration_rate=0.1,
        )

        # Create Optuna study
        print(f"  Running optimization study ({n_trials} trials)...")
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Run optimization
        study.optimize(
            lambda trial: quadratic_objective(trial, n_dims),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        results['study_completed'] = True
        results['n_trials'] = len(study.trials)
        results['best_value'] = study.best_value
        results['best_params'] = study.best_params

        # Record trial history
        for trial in study.trials:
            results['trial_history'].append({
                'number': trial.number,
                'params': {k: round(v, 4) for k, v in trial.params.items()},
                'value': round(trial.value, 4) if trial.value else None,
            })

        # ========== VALIDITY CHECK ==========
        print("\n  Checking suggestion validity...")

        all_valid = True
        for trial in study.trials:
            for param_name, value in trial.params.items():
                if not (0.0 <= value <= 5.0):
                    all_valid = False
                    results['errors'].append(
                        f"Out of bounds: {param_name}={value} in trial {trial.number}"
                    )

        results['all_suggestions_valid'] = all_valid
        print(f"    All suggestions in bounds: {all_valid}")

        # ========== ADAPTATION CHECK ==========
        print("\n  Checking for adaptation behavior...")

        # Check if model adapts: later trials should generally be better
        if len(study.trials) >= 6:
            early_values = [t.value for t in study.trials[:3]]
            late_values = [t.value for t in study.trials[-3:]]

            early_avg = np.mean(early_values)
            late_avg = np.mean(late_values)

            # Model should improve over time (late trials better than early)
            results['suggestions_adapted'] = late_avg < early_avg

            print(f"    Early trials avg value: {early_avg:.4f}")
            print(f"    Late trials avg value: {late_avg:.4f}")
            print(f"    Shows adaptation: {results['suggestions_adapted']}")
        else:
            results['suggestions_adapted'] = True  # Not enough trials to check

        # ========== OPTIMUM PROXIMITY CHECK ==========
        print("\n  Checking proximity to optimum...")

        optimum = {f"x{i}": 2.0 for i in range(n_dims)}
        best_distance = np.mean([
            abs(study.best_params[f"x{i}"] - 2.0) for i in range(n_dims)
        ])

        results['distance_from_optimum'] = float(best_distance)
        print(f"    Optimum: x_i = 2.0 for all i")
        print(f"    Best found: {study.best_params}")
        print(f"    Best value: {study.best_value:.4f}")
        print(f"    Avg distance from optimum: {best_distance:.4f}")

        # ========== SUMMARY ==========
        print("\n  Study Summary:")
        print(f"    Trials completed: {results['n_trials']}")
        print(f"    Best objective value: {results['best_value']:.4f}")

        # Success criteria:
        # 1. Study completed without errors
        # 2. All suggestions were valid (in bounds)
        # 3. Best value is reasonable (< 10 for this simple objective)
        overall_pass = (
            results['study_completed'] and
            results['all_suggestions_valid'] and
            results['best_value'] is not None and
            results['best_value'] < 10.0
        )

        print(f"\n  Optuna integration test: {'PASS' if overall_pass else 'FAIL'}")

        return overall_pass, results

    except Exception as e:
        import traceback
        results['errors'].append(str(e))
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False, results


def run_comparison_study(
    model_path: Path,
    n_dims: int = 3,
    n_trials: int = 20,
    n_seeds: int = 3,
    device: str = None,
) -> Dict[str, Any]:
    """
    Run comparison between OptiFormer, TPE, and Random samplers.

    Returns summary statistics for each sampler.
    """
    print("\n  Running sampler comparison...")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    comparison_results = {
        'optiformer': [],
        'tpe': [],
        'random': [],
    }

    model = OptiFormer.load(str(model_path), device=device)
    model.eval()

    param_specs = create_param_specs_for_optuna(n_dims)
    tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

    for seed in range(n_seeds):
        # OptiFormer
        of_sampler = OptiFormerSampler(
            model=model, tokenizer=tokenizer, device=device,
            temperature=0.8, min_trials_for_model=2, exploration_rate=0.1,
        )
        of_study = optuna.create_study(direction="minimize", sampler=of_sampler)
        of_study.optimize(
            lambda t: quadratic_objective(t, n_dims),
            n_trials=n_trials, show_progress_bar=False
        )
        comparison_results['optiformer'].append(of_study.best_value)

        # TPE
        tpe_sampler = optuna.samplers.TPESampler(seed=seed)
        tpe_study = optuna.create_study(direction="minimize", sampler=tpe_sampler)
        tpe_study.optimize(
            lambda t: quadratic_objective(t, n_dims),
            n_trials=n_trials, show_progress_bar=False
        )
        comparison_results['tpe'].append(tpe_study.best_value)

        # Random
        rand_sampler = optuna.samplers.RandomSampler(seed=seed)
        rand_study = optuna.create_study(direction="minimize", sampler=rand_sampler)
        rand_study.optimize(
            lambda t: quadratic_objective(t, n_dims),
            n_trials=n_trials, show_progress_bar=False
        )
        comparison_results['random'].append(rand_study.best_value)

    # Compute statistics
    stats = {}
    for name, values in comparison_results.items():
        stats[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'best': float(min(values)),
        }

    print(f"\n    OptiFormer: {stats['optiformer']['mean']:.4f} +/- {stats['optiformer']['std']:.4f}")
    print(f"    TPE:        {stats['tpe']['mean']:.4f} +/- {stats['tpe']['std']:.4f}")
    print(f"    Random:     {stats['random']['mean']:.4f} +/- {stats['random']['std']:.4f}")

    return stats


if __name__ == "__main__":
    model_path = Path("./outputs/smoke_test/model/best.pt")

    success, results = test_optuna_integration(
        model_path=model_path,
        n_dims=3,
        n_trials=15,
    )

    print(f"\nOptuna integration test: {'SUCCESS' if success else 'FAILED'}")

    if success:
        print("\nRunning sampler comparison...")
        comparison = run_comparison_study(
            model_path=model_path,
            n_dims=3,
            n_trials=20,
            n_seeds=3,
        )

    sys.exit(0 if success else 1)
