"""
Full Evaluation Pipeline

Compare OptiFormer against baseline optimizers on benchmarks.
"""

import optuna
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import json
from pathlib import Path
import time

from .synthetic_benchmarks import BENCHMARKS, normalize_function
from samplers import OptiFormerSampler

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class EvaluationResult:
    """Result of a single optimization run."""
    method: str
    benchmark: str
    best_value: float
    values_over_time: List[float]
    n_trials: int
    seed: int
    elapsed_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'benchmark': self.benchmark,
            'best_value': self.best_value,
            'values_over_time': self.values_over_time,
            'n_trials': self.n_trials,
            'seed': self.seed,
            'elapsed_time': self.elapsed_time,
        }


def evaluate_on_function(
    sampler,
    objective_fn: Callable,
    search_space: Dict[str, Dict],
    n_trials: int,
    direction: str = "minimize",
    seed: int = 42,
) -> EvaluationResult:
    """
    Run optimization and collect results.

    Args:
        sampler: Optuna sampler to use
        objective_fn: Objective function
        search_space: Search space definition
        n_trials: Number of trials
        direction: Optimization direction
        seed: Random seed

    Returns:
        EvaluationResult with all metrics
    """
    np.random.seed(seed)

    def objective(trial):
        params = {}
        x_vals = []

        for name, spec in search_space.items():
            if spec['type'] == 'float':
                val = trial.suggest_float(
                    name, spec['low'], spec['high'],
                    log=spec.get('log', False)
                )
            elif spec['type'] == 'int':
                val = trial.suggest_int(name, spec['low'], spec['high'])
            elif spec['type'] == 'categorical':
                val = trial.suggest_categorical(name, spec['choices'])
            else:
                val = trial.suggest_float(name, spec['low'], spec['high'])

            params[name] = val
            x_vals.append(val)

        x = np.array(x_vals)
        return float(objective_fn(x.reshape(1, -1))[0])

    start_time = time.time()
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, catch=(Exception,))
    elapsed = time.time() - start_time

    # Collect best value over time
    values = []
    best_so_far = float('inf') if direction == 'minimize' else float('-inf')

    for trial in study.trials:
        if trial.value is not None:
            if direction == 'minimize':
                best_so_far = min(best_so_far, trial.value)
            else:
                best_so_far = max(best_so_far, trial.value)
        values.append(best_so_far)

    return EvaluationResult(
        method=sampler.__class__.__name__,
        benchmark="",
        best_value=study.best_value if study.best_trial else float('inf'),
        values_over_time=values,
        n_trials=n_trials,
        seed=seed,
        elapsed_time=elapsed,
    )


def run_synthetic_evaluation(
    optiformer_sampler,
    benchmark_names: List[str] = None,
    n_dims: int = 2,
    n_trials: int = 50,
    n_seeds: int = 5,
    output_dir: Path = None,
) -> Dict:
    """
    Evaluate on synthetic benchmarks.

    Args:
        optiformer_sampler: The trained OptiFormer sampler
        benchmark_names: Which benchmarks to run
        n_dims: Dimensionality
        n_trials: Trials per run
        n_seeds: Number of random seeds
        output_dir: Where to save results

    Returns:
        Results dictionary
    """
    benchmark_names = benchmark_names or ['sphere', 'rastrigin', 'rosenbrock', 'ackley', 'levy']

    results = {
        'optiformer': [],
        'tpe': [],
        'random': [],
    }

    for bench_name in benchmark_names:
        if bench_name not in BENCHMARKS:
            print(f"Unknown benchmark: {bench_name}")
            continue

        bench = BENCHMARKS[bench_name]
        fn = bench.fn
        bounds = bench.bounds

        # Create search space
        search_space = {
            f'x{i}': {'type': 'float', 'low': bounds[0], 'high': bounds[1]}
            for i in range(n_dims)
        }

        print(f"\nEvaluating on {bench_name} (d={n_dims})...")

        for seed in tqdm(range(n_seeds), desc=f"Seeds for {bench_name}"):
            # OptiFormer
            result = evaluate_on_function(
                optiformer_sampler,
                fn,
                search_space,
                n_trials,
                seed=seed,
            )
            result.benchmark = bench_name
            results['optiformer'].append(result)

            # TPE baseline
            tpe_sampler = optuna.samplers.TPESampler(seed=seed)
            result = evaluate_on_function(
                tpe_sampler,
                fn,
                search_space,
                n_trials,
                seed=seed,
            )
            result.benchmark = bench_name
            results['tpe'].append(result)

            # Random baseline
            random_sampler = optuna.samplers.RandomSampler(seed=seed)
            result = evaluate_on_function(
                random_sampler,
                fn,
                search_space,
                n_trials,
                seed=seed,
            )
            result.benchmark = bench_name
            results['random'].append(result)

    # Compute statistics
    stats = compute_statistics(results, benchmark_names)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'synthetic_results.json', 'w') as f:
            json.dump({
                'results': {k: [r.to_dict() for r in v] for k, v in results.items()},
                'statistics': stats,
            }, f, indent=2)

    return {'results': results, 'statistics': stats}


def run_ml_evaluation(
    optiformer_sampler,
    benchmark,
    n_trials: int = 30,
    n_seeds: int = 3,
    output_dir: Path = None,
) -> Dict:
    """
    Evaluate on ML benchmark.

    Args:
        optiformer_sampler: The trained OptiFormer sampler
        benchmark: ML benchmark instance (e.g., MNISTBenchmark)
        n_trials: Trials per run
        n_seeds: Number of seeds
        output_dir: Where to save results

    Returns:
        Results dictionary
    """
    results = {
        'optiformer': [],
        'tpe': [],
        'random': [],
    }

    bench_name = benchmark.__class__.__name__
    search_space = benchmark.search_space

    print(f"\nEvaluating on {bench_name}...")

    for seed in tqdm(range(n_seeds), desc="Seeds"):
        np.random.seed(seed)

        # OptiFormer
        def make_objective(sampler_name):
            def objective(trial):
                params = {}
                for name, spec in search_space.items():
                    if spec['type'] == 'float':
                        params[name] = trial.suggest_float(
                            name, spec['low'], spec['high'],
                            log=spec.get('log', False)
                        )
                    elif spec['type'] == 'int':
                        params[name] = trial.suggest_int(name, spec['low'], spec['high'])
                    elif spec['type'] == 'categorical':
                        params[name] = trial.suggest_categorical(name, spec['choices'])
                return benchmark.evaluate(params)
            return objective

        # OptiFormer
        start = time.time()
        study = optuna.create_study(direction="minimize", sampler=optiformer_sampler)
        study.optimize(make_objective("optiformer"), n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - start

        values = []
        best = float('inf')
        for t in study.trials:
            if t.value is not None:
                best = min(best, t.value)
            values.append(best)

        results['optiformer'].append(EvaluationResult(
            method='OptiFormerSampler',
            benchmark=bench_name,
            best_value=study.best_value if study.best_trial else float('inf'),
            values_over_time=values,
            n_trials=n_trials,
            seed=seed,
            elapsed_time=elapsed,
        ))

        # TPE
        start = time.time()
        tpe_sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=tpe_sampler)
        study.optimize(make_objective("tpe"), n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - start

        values = []
        best = float('inf')
        for t in study.trials:
            if t.value is not None:
                best = min(best, t.value)
            values.append(best)

        results['tpe'].append(EvaluationResult(
            method='TPESampler',
            benchmark=bench_name,
            best_value=study.best_value if study.best_trial else float('inf'),
            values_over_time=values,
            n_trials=n_trials,
            seed=seed,
            elapsed_time=elapsed,
        ))

        # Random
        start = time.time()
        random_sampler = optuna.samplers.RandomSampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=random_sampler)
        study.optimize(make_objective("random"), n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - start

        values = []
        best = float('inf')
        for t in study.trials:
            if t.value is not None:
                best = min(best, t.value)
            values.append(best)

        results['random'].append(EvaluationResult(
            method='RandomSampler',
            benchmark=bench_name,
            best_value=study.best_value if study.best_trial else float('inf'),
            values_over_time=values,
            n_trials=n_trials,
            seed=seed,
            elapsed_time=elapsed,
        ))

    # Compute statistics
    stats = compute_ml_statistics(results)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f'{bench_name}_results.json', 'w') as f:
            json.dump({
                'results': {k: [r.to_dict() for r in v] for k, v in results.items()},
                'statistics': stats,
            }, f, indent=2)

    return {'results': results, 'statistics': stats}


def compute_statistics(results: Dict, benchmarks: List[str]) -> Dict:
    """Compute comparison statistics for synthetic benchmarks."""
    stats = {}

    for bench in benchmarks:
        optiformer_vals = [r.best_value for r in results['optiformer'] if r.benchmark == bench]
        tpe_vals = [r.best_value for r in results['tpe'] if r.benchmark == bench]
        random_vals = [r.best_value for r in results['random'] if r.benchmark == bench]

        if not optiformer_vals:
            continue

        stats[bench] = {
            'optiformer_mean': float(np.mean(optiformer_vals)),
            'optiformer_std': float(np.std(optiformer_vals)),
            'tpe_mean': float(np.mean(tpe_vals)),
            'tpe_std': float(np.std(tpe_vals)),
            'random_mean': float(np.mean(random_vals)),
            'random_std': float(np.std(random_vals)),
            'optiformer_vs_random_win_rate': float(np.mean(
                np.array(optiformer_vals) < np.array(random_vals)
            )),
            'optiformer_vs_tpe_win_rate': float(np.mean(
                np.array(optiformer_vals) < np.array(tpe_vals)
            )),
        }

    # Overall statistics
    all_optiformer = [r.best_value for r in results['optiformer']]
    all_tpe = [r.best_value for r in results['tpe']]
    all_random = [r.best_value for r in results['random']]

    stats['overall'] = {
        'vs_random_win_rate': float(np.mean(np.array(all_optiformer) < np.array(all_random))),
        'vs_tpe_win_rate': float(np.mean(np.array(all_optiformer) < np.array(all_tpe))),
    }

    return stats


def compute_ml_statistics(results: Dict) -> Dict:
    """Compute comparison statistics for ML benchmarks."""
    optiformer_vals = [r.best_value for r in results['optiformer']]
    tpe_vals = [r.best_value for r in results['tpe']]
    random_vals = [r.best_value for r in results['random']]

    return {
        'optiformer_mean': float(np.mean(optiformer_vals)),
        'optiformer_std': float(np.std(optiformer_vals)),
        'optiformer_best': float(np.min(optiformer_vals)),
        'tpe_mean': float(np.mean(tpe_vals)),
        'tpe_std': float(np.std(tpe_vals)),
        'tpe_best': float(np.min(tpe_vals)),
        'random_mean': float(np.mean(random_vals)),
        'random_std': float(np.std(random_vals)),
        'random_best': float(np.min(random_vals)),
        'vs_random_win_rate': float(np.mean(np.array(optiformer_vals) < np.array(random_vals))),
        'vs_tpe_win_rate': float(np.mean(np.array(optiformer_vals) < np.array(tpe_vals))),
    }


def print_evaluation_summary(stats: Dict):
    """Print a summary of evaluation results."""
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)

    if 'overall' in stats:
        print(f"\nOverall Performance:")
        print(f"  vs Random win rate: {stats['overall']['vs_random_win_rate']:.1%}")
        print(f"  vs TPE win rate: {stats['overall']['vs_tpe_win_rate']:.1%}")

    for bench, bench_stats in stats.items():
        if bench == 'overall':
            continue
        print(f"\n{bench}:")
        print(f"  OptiFormer: {bench_stats.get('optiformer_mean', 0):.4f} +/- {bench_stats.get('optiformer_std', 0):.4f}")
        print(f"  TPE:        {bench_stats.get('tpe_mean', 0):.4f} +/- {bench_stats.get('tpe_std', 0):.4f}")
        print(f"  Random:     {bench_stats.get('random_mean', 0):.4f} +/- {bench_stats.get('random_std', 0):.4f}")
