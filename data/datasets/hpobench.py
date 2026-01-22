"""
HPOBench Dataset Loader

Loads real-world hyperparameter optimization traces from HPOBench
and other sources for training.

HPOBench provides tabular benchmarks with precomputed results from
real ML training runs (FCNet, XGBoost, etc.)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import warnings

from data.tokenizer import SequenceTokenizer, ParameterSpec, Trial


def try_import_hpobench():
    """Try to import HPOBench, return None if not available."""
    try:
        from hpobench.benchmarks.nas.tabular_benchmarks import FCNetBenchmark
        return True
    except ImportError:
        return False


def normalize_params_to_generic(
    params: dict,
    param_order: list = None,
    param_ranges: dict = None,
) -> dict:
    """
    Normalize parameter names to generic x0, x1, x2, ... format.

    This is CRITICAL for mixing real-world data with synthetic data.
    The model learns on generic parameter positions, so all data must
    use the same naming convention.

    Args:
        params: Dict with original parameter names
        param_order: Optional fixed ordering of parameter names
        param_ranges: Optional dict of {param_name: (min, max)} for normalization

    Returns:
        Dict with parameters renamed to x0, x1, x2, ... and values in [0, 1]
    """
    if param_order is None:
        param_order = sorted(params.keys())

    normalized = {}
    for i, name in enumerate(param_order):
        if name in params:
            value = params[name]
            # Convert categorical to numeric index if string
            if isinstance(value, str):
                # Hash to a float in [0, 1] for categorical values
                # This is a simple approach; could use proper encoding
                value = (hash(value) % 1000) / 1000.0
            else:
                value = float(value)
                # Normalize to [0, 1] if ranges provided
                if param_ranges and name in param_ranges:
                    low, high = param_ranges[name]
                    if high > low:
                        value = (value - low) / (high - low)
                        value = max(0.0, min(1.0, value))  # Clamp

            normalized[f"x{i}"] = float(value)

    return normalized


def get_hpobench_param_ranges(benchmark_name: str) -> dict:
    """Get value ranges for HPOBench benchmark parameters."""
    # Define ranges for normalization based on benchmark specs
    PARAM_RANGES = {
        'activation_fn_1': (0, 1),  # categorical, handled by hash
        'activation_fn_2': (0, 1),
        'batch_size': (8, 64),
        'dropout_1': (0.0, 0.6),
        'dropout_2': (0.0, 0.6),
        'init_lr': (0.0005, 0.1),
        'lr_schedule': (0, 1),  # categorical
        'n_units_1': (16, 512),
        'n_units_2': (16, 512),
    }
    return PARAM_RANGES


class HPOBenchLoader:
    """
    Loader for HPOBench tabular benchmarks.

    HPOBench provides precomputed results for hyperparameter configurations,
    allowing us to generate optimization trajectories without actual training.
    """

    # Available benchmarks and their search spaces
    BENCHMARKS = {
        'fcnet_protein': {
            'class': 'FCNetProteinStructureBenchmark',
            'params': {
                'activation_fn_1': {'type': 'categorical', 'choices': ['tanh', 'relu']},
                'activation_fn_2': {'type': 'categorical', 'choices': ['tanh', 'relu']},
                'batch_size': {'type': 'categorical', 'choices': [8, 16, 32, 64]},
                'dropout_1': {'type': 'categorical', 'choices': [0.0, 0.3, 0.6]},
                'dropout_2': {'type': 'categorical', 'choices': [0.0, 0.3, 0.6]},
                'init_lr': {'type': 'categorical', 'choices': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]},
                'lr_schedule': {'type': 'categorical', 'choices': ['cosine', 'const']},
                'n_units_1': {'type': 'categorical', 'choices': [16, 32, 64, 128, 256, 512]},
                'n_units_2': {'type': 'categorical', 'choices': [16, 32, 64, 128, 256, 512]},
            },
        },
        'fcnet_naval': {
            'class': 'FCNetNavalPropulsionBenchmark',
            'params': {
                'activation_fn_1': {'type': 'categorical', 'choices': ['tanh', 'relu']},
                'activation_fn_2': {'type': 'categorical', 'choices': ['tanh', 'relu']},
                'batch_size': {'type': 'categorical', 'choices': [8, 16, 32, 64]},
                'dropout_1': {'type': 'categorical', 'choices': [0.0, 0.3, 0.6]},
                'dropout_2': {'type': 'categorical', 'choices': [0.0, 0.3, 0.6]},
                'init_lr': {'type': 'categorical', 'choices': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]},
                'lr_schedule': {'type': 'categorical', 'choices': ['cosine', 'const']},
                'n_units_1': {'type': 'categorical', 'choices': [16, 32, 64, 128, 256, 512]},
                'n_units_2': {'type': 'categorical', 'choices': [16, 32, 64, 128, 256, 512]},
            },
        },
        'fcnet_slice': {
            'class': 'FCNetSliceLocalizationBenchmark',
            'params': {
                'activation_fn_1': {'type': 'categorical', 'choices': ['tanh', 'relu']},
                'activation_fn_2': {'type': 'categorical', 'choices': ['tanh', 'relu']},
                'batch_size': {'type': 'categorical', 'choices': [8, 16, 32, 64]},
                'dropout_1': {'type': 'categorical', 'choices': [0.0, 0.3, 0.6]},
                'dropout_2': {'type': 'categorical', 'choices': [0.0, 0.3, 0.6]},
                'init_lr': {'type': 'categorical', 'choices': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]},
                'lr_schedule': {'type': 'categorical', 'choices': ['cosine', 'const']},
                'n_units_1': {'type': 'categorical', 'choices': [16, 32, 64, 128, 256, 512]},
                'n_units_2': {'type': 'categorical', 'choices': [16, 32, 64, 128, 256, 512]},
            },
        },
    }

    def __init__(self, data_dir: str = './hpobench_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._benchmarks = {}

    def _load_benchmark(self, name: str):
        """Load a benchmark by name."""
        if not try_import_hpobench():
            raise ImportError(
                "HPOBench not installed. Install with: pip install hpobench"
            )

        if name not in self.BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {name}")

        if name not in self._benchmarks:
            from hpobench.benchmarks.nas.tabular_benchmarks import (
                FCNetProteinStructureBenchmark,
                FCNetNavalPropulsionBenchmark,
                FCNetSliceLocalizationBenchmark,
            )

            benchmark_classes = {
                'FCNetProteinStructureBenchmark': FCNetProteinStructureBenchmark,
                'FCNetNavalPropulsionBenchmark': FCNetNavalPropulsionBenchmark,
                'FCNetSliceLocalizationBenchmark': FCNetSliceLocalizationBenchmark,
            }

            class_name = self.BENCHMARKS[name]['class']
            self._benchmarks[name] = benchmark_classes[class_name](
                data_dir=str(self.data_dir)
            )

        return self._benchmarks[name]

    def generate_trajectory(
        self,
        benchmark_name: str,
        n_trials: int = 64,
        seed: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate an optimization trajectory using TPE on HPOBench.

        Args:
            benchmark_name: Name of the benchmark
            n_trials: Number of trials
            seed: Random seed

        Returns:
            List of trial dicts with params and score
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        benchmark = self._load_benchmark(benchmark_name)
        search_space = self.BENCHMARKS[benchmark_name]['params']
        trajectory = []

        def objective(trial):
            config = {}
            for name, spec in search_space.items():
                if spec['type'] == 'categorical':
                    config[name] = trial.suggest_categorical(name, spec['choices'])

            # Query benchmark
            result = benchmark.objective_function(config)
            score = result['function_value']

            # Normalize to generic parameter names (x0, x1, ...) AND values to [0, 1]
            # for compatibility with synthetic data
            param_order = list(search_space.keys())
            param_ranges = get_hpobench_param_ranges(benchmark_name)
            normalized_params = normalize_params_to_generic(config, param_order, param_ranges)

            trajectory.append({
                'params': normalized_params,
                'score': float(score),
                'original_params': config,  # Keep original for debugging
            })

            return score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return trajectory

    def generate_trajectories(
        self,
        benchmark_name: str,
        n_trajectories: int,
        n_trials: int = 64,
        output_path: Optional[Path] = None,
    ) -> List[List[Dict]]:
        """Generate multiple trajectories and optionally save to file."""
        from tqdm import tqdm

        trajectories = []
        for i in tqdm(range(n_trajectories), desc=f"Generating {benchmark_name}"):
            traj = self.generate_trajectory(benchmark_name, n_trials, seed=i)
            trajectories.append(traj)

            if output_path:
                with open(output_path, 'a') as f:
                    # Get parameter order for metadata
                    param_order = list(self.BENCHMARKS[benchmark_name]['params'].keys())
                    record = {
                        'source': 'hpobench',
                        'benchmark': benchmark_name,
                        'trajectory_id': i,
                        'n_trials': len(traj),
                        'n_params': len(param_order),
                        'param_mapping': {f'x{j}': name for j, name in enumerate(param_order)},
                        'trajectory': traj,
                    }
                    f.write(json.dumps(record) + '\n')

        return trajectories


class RealWorldTraceCollector:
    """
    Collects real optimization traces from Optuna studies.

    Use this to gather training data from your own hyperparameter
    optimization runs.
    """

    @staticmethod
    def from_optuna_study(
        study,
        normalize_scores: bool = True,
        normalize_param_names: bool = True,
    ) -> List[Trial]:
        """
        Extract trajectory from an Optuna study.

        Args:
            study: Optuna study object
            normalize_scores: Whether to normalize scores to [0, 1]
            normalize_param_names: Whether to convert param names to x0, x1, ...

        Returns:
            List of Trial objects
        """
        trials = []
        all_param_names = set()

        # First pass: collect all parameter names
        for t in study.trials:
            if t.state.name == 'COMPLETE' and t.value is not None:
                all_param_names.update(t.params.keys())

        param_order = sorted(all_param_names)

        # Second pass: extract trials with normalized params
        for t in study.trials:
            if t.state.name == 'COMPLETE' and t.value is not None:
                if normalize_param_names:
                    params = normalize_params_to_generic(dict(t.params), param_order)
                else:
                    params = dict(t.params)
                trials.append(Trial(
                    params=params,
                    score=t.value,
                ))

        if normalize_scores and trials:
            scores = [t.score for t in trials]
            min_s, max_s = min(scores), max(scores)
            if max_s - min_s > 1e-10:
                for t in trials:
                    t.score = (t.score - min_s) / (max_s - min_s)

        return trials

    @staticmethod
    def from_optuna_storage(
        storage_url: str,
        study_name: Optional[str] = None,
        normalize_scores: bool = True,
        normalize_param_names: bool = True,
    ) -> List[List[Trial]]:
        """
        Load trajectories from Optuna storage (SQLite, etc.).

        Args:
            storage_url: Optuna storage URL (e.g., "sqlite:///optuna.db")
            study_name: Optional specific study name
            normalize_scores: Whether to normalize scores
            normalize_param_names: Whether to convert param names to x0, x1, ...

        Returns:
            List of trajectories (one per study)
        """
        import optuna

        storage = optuna.storages.get_storage(storage_url)

        if study_name:
            study_names = [study_name]
        else:
            study_names = [s.study_name for s in storage.get_all_studies()]

        trajectories = []
        for name in study_names:
            study = optuna.load_study(study_name=name, storage=storage_url)
            trajectory = RealWorldTraceCollector.from_optuna_study(
                study, normalize_scores, normalize_param_names
            )
            if trajectory:
                trajectories.append(trajectory)

        return trajectories

    @staticmethod
    def save_trajectory(
        trajectory: List[Trial],
        output_path: Path,
        metadata: Optional[Dict] = None,
    ):
        """Save a trajectory to JSONL file."""
        record = {
            'source': 'real_world',
            'n_trials': len(trajectory),
            'trajectory': [
                {'params': t.params, 'score': t.score}
                for t in trajectory
            ],
        }
        if metadata:
            record.update(metadata)

        with open(output_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


def load_openml_traces(
    task_ids: List[int] = None,
    n_configs_per_task: int = 100,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Load hyperparameter configurations from OpenML.

    OpenML has logged results from many AutoML runs that can be
    used as training data.

    Args:
        task_ids: OpenML task IDs to load
        n_configs_per_task: Number of configurations per task
        output_path: Optional path to save results

    Returns:
        List of trajectory records
    """
    try:
        import openml
    except ImportError:
        raise ImportError("OpenML not installed. Install with: pip install openml")

    task_ids = task_ids or [3, 12, 31, 53, 3917]  # Common benchmarks
    trajectories = []

    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            runs = openml.runs.list_runs(task=[task_id], size=n_configs_per_task)

            trajectory = []
            param_names_seen = set()

            for run_id, run_info in runs.items():
                try:
                    run = openml.runs.get_run(run_id)
                    params = run.parameter_settings
                    # Convert to flat dict
                    param_dict = {p['oml:name']: p['oml:value'] for p in params}

                    # Track all parameter names for consistent ordering
                    param_names_seen.update(param_dict.keys())

                    score = run_info.get('predictive_accuracy', 0.5)

                    trajectory.append({
                        'original_params': param_dict,
                        'score': 1 - score,  # Convert accuracy to error
                    })
                except Exception:
                    continue

            # Normalize all params with consistent ordering
            param_order = sorted(param_names_seen)
            for t in trajectory:
                t['params'] = normalize_params_to_generic(
                    t['original_params'], param_order
                )

            if trajectory:
                record = {
                    'source': 'openml',
                    'task_id': task_id,
                    'n_trials': len(trajectory),
                    'n_params': len(param_order),
                    'param_mapping': {f'x{i}': name for i, name in enumerate(param_order)},
                    'trajectory': trajectory,
                }
                trajectories.append(record)

                if output_path:
                    with open(output_path, 'a') as f:
                        f.write(json.dumps(record) + '\n')

        except Exception as e:
            warnings.warn(f"Failed to load task {task_id}: {e}")

    return trajectories
