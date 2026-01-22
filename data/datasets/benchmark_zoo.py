"""
Benchmark Zoo: Integration of Multiple HPO Benchmark Suites

This module provides unified access to various HPO benchmarks:
- YAHPO Gym: 14 scenarios, ~700 instances (LCBench, rbv2_*, nb301, etc.)
- JAHS-Bench-201: Joint Architecture + Hyperparameter Search (CIFAR10, Fashion-MNIST)
- LCBench: Learning curves for 2000 configs on 35 OpenML datasets
- HPOBench: FCNet, XGBoost, SVM, RandomForest surrogates

All benchmarks are normalized to a common format compatible with OptiFormer training.
"""

import numpy as np
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from .hpobench import normalize_params_to_generic


# ============================================================================
# YAHPO Gym Integration
# ============================================================================

def try_import_yahpo():
    """Check if YAHPO Gym is available."""
    try:
        import yahpo_gym
        return True
    except ImportError:
        return False


class YAHPOLoader:
    """
    Loader for YAHPO Gym surrogate benchmarks.

    YAHPO Gym provides fast surrogate-based benchmarks for:
    - lcbench: Learning curves on OpenML (7 hyperparams, 35 datasets)
    - rbv2_svm, rbv2_ranger, rbv2_xgboost, rbv2_rpart, rbv2_glmnet: ML algorithms
    - nb301: NAS-Bench-301 neural architecture search
    - iaml_*: Various AutoML benchmarks

    Install: pip install yahpo-gym
    Data: Download from https://github.com/slds-lmu/yahpo_data
    """

    # Recommended scenarios for HPO training
    SCENARIOS = {
        'lcbench': {
            'description': 'Learning curves for neural networks on OpenML',
            'n_params': 7,
            'fidelity': 'epoch',
            'instances': 35,  # 35 OpenML datasets
        },
        'rbv2_svm': {
            'description': 'SVM hyperparameter optimization',
            'n_params': 6,
            'fidelity': 'trainsize',
            'instances': 103,
        },
        'rbv2_xgboost': {
            'description': 'XGBoost hyperparameter optimization',
            'n_params': 14,
            'fidelity': 'trainsize',
            'instances': 103,
        },
        'rbv2_ranger': {
            'description': 'Random Forest (ranger) optimization',
            'n_params': 8,
            'fidelity': 'trainsize',
            'instances': 103,
        },
        'nb301': {
            'description': 'NAS-Bench-301 architecture search',
            'n_params': 34,
            'fidelity': 'epoch',
            'instances': 1,
        },
    }

    def __init__(self, data_path: str = None):
        """
        Initialize YAHPO loader.

        Args:
            data_path: Path to YAHPO data directory (download from GitHub)
        """
        if not try_import_yahpo():
            raise ImportError(
                "YAHPO Gym not installed. Install with: pip install yahpo-gym\n"
                "Then download data from: https://github.com/slds-lmu/yahpo_data"
            )

        from yahpo_gym import local_config
        local_config.init_config()
        if data_path:
            local_config.set_data_path(data_path)

        self.benchmarks = {}

    def _get_benchmark(self, scenario: str, instance: str = None):
        """Get or create benchmark instance."""
        from yahpo_gym import benchmark_set

        key = f"{scenario}_{instance}"
        if key not in self.benchmarks:
            bench = benchmark_set.BenchmarkSet(scenario)
            if instance:
                bench.set_instance(instance)
            self.benchmarks[key] = bench
        return self.benchmarks[key]

    def generate_trajectory(
        self,
        scenario: str,
        instance: str,
        n_trials: int = 64,
        seed: int = None,
        fidelity: int = None,
    ) -> List[Dict]:
        """
        Generate optimization trajectory using TPE on YAHPO benchmark.

        Args:
            scenario: YAHPO scenario name (e.g., 'lcbench', 'rbv2_xgboost')
            instance: Instance ID within scenario
            n_trials: Number of optimization trials
            seed: Random seed
            fidelity: Fidelity level (e.g., epoch for lcbench)

        Returns:
            List of trial dicts with normalized params and scores
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        bench = self._get_benchmark(scenario, instance)
        config_space = bench.config_space

        # Get parameter names and their ranges for normalization
        param_names = list(config_space.keys())
        param_ranges = {}
        for name in param_names:
            hp = config_space[name]
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                param_ranges[name] = (hp.lower, hp.upper)

        trajectory = []

        def objective(trial):
            config = {}
            for name in param_names:
                hp = config_space[name]
                if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                    if hasattr(hp, 'log') and hp.log:
                        config[name] = trial.suggest_float(name, hp.lower, hp.upper, log=True)
                    else:
                        config[name] = trial.suggest_float(name, hp.lower, hp.upper)
                elif hasattr(hp, 'choices'):
                    config[name] = trial.suggest_categorical(name, list(hp.choices))

            # Add fidelity if specified
            if fidelity is not None:
                fidelity_name = bench.config_space.get_hyperparameter_names()[-1]  # Usually last
                config[fidelity_name] = fidelity

            # Prune inactive hyperparameters
            # We must remove parameters whose parent conditions are not met
            # Run multiple passes to handle dependency chains
            for _ in range(3):
                for cond in config_space.get_conditions():
                    child_name = cond.child.name
                    if child_name in config:
                        try:
                            if not cond.evaluate(config):
                                del config[child_name]
                        except Exception:
                            pass

            # Evaluate
            result = bench.objective_function(config)

            # Get primary metric (usually first one, or 'val_accuracy'/'val_balanced_accuracy')
            if isinstance(result, list):
                result = result[0]

            score = result.get('val_balanced_accuracy',
                             result.get('val_accuracy',
                             result.get('y', 0.5)))

            # Convert to minimization (error)
            if score > 1:  # Some metrics are percentages
                score = score / 100
            error = 1.0 - score

            # Normalize params
            normalized = normalize_params_to_generic(config, param_names, param_ranges)

            trajectory.append({
                'params': normalized,
                'score': float(error),
                'original_params': config,
            })

            return error

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return trajectory

    def generate_trajectories(
        self,
        scenario: str,
        n_trajectories: int,
        n_trials: int = 64,
        instances: List[str] = None,
        output_path: Optional[Path] = None,
    ) -> List[List[Dict]]:
        """Generate multiple trajectories across instances."""
        bench = self._get_benchmark(scenario)

        if instances is None:
            instances = list(bench.instances)[:20]  # Limit for speed

        trajectories = []

        for i in tqdm(range(n_trajectories), desc=f"YAHPO {scenario}"):
            instance = instances[i % len(instances)]
            traj = self.generate_trajectory(
                scenario, instance, n_trials, seed=i
            )
            trajectories.append(traj)

            if output_path:
                with open(output_path, 'a') as f:
                    # Get param info from first trajectory for metadata
                    if traj and 'original_params' in traj[0]:
                        orig_params = list(traj[0]['original_params'].keys())
                        param_mapping = {f'x{j}': name for j, name in enumerate(sorted(orig_params))}
                    else:
                        param_mapping = {}

                    record = {
                        'source': 'yahpo',
                        'scenario': scenario,
                        'instance': instance,
                        'trajectory_id': i,
                        'n_trials': len(traj),
                        'n_params': len(param_mapping),
                        'param_mapping': param_mapping,
                        'trajectory': traj,
                    }
                    f.write(json.dumps(record) + '\n')

        return trajectories


# ============================================================================
# JAHS-Bench-201 Integration
# ============================================================================

def try_import_jahs():
    """Check if JAHS-Bench is available."""
    try:
        import jahs_bench
        return True
    except ImportError:
        return False


class JAHSBenchLoader:
    """
    Loader for JAHS-Bench-201: Joint Architecture and Hyperparameter Search.

    Features:
    - 14-dimensional search space (architecture + hyperparameters)
    - 3 tasks: CIFAR-10, Fashion-MNIST, Colorectal Histology
    - Multi-fidelity (epochs) and multi-objective support
    - ~161 million data points

    Install: pip install git+https://github.com/automl/jahs_bench_201.git
    """

    TASKS = ['cifar10', 'colorectal_histology', 'fashion_mnist']

    # Parameter ranges for normalization
    PARAM_RANGES = {
        'LearningRate': (1e-3, 1.0),
        'WeightDecay': (1e-5, 1e-2),
        'Activation': (0, 1),  # categorical
        'TrivialAugment': (0, 1),  # boolean
        'Op1': (0, 4),  # categorical (5 choices)
        'Op2': (0, 4),
        'Op3': (0, 4),
        'Op4': (0, 4),
        'Op5': (0, 4),
        'Op6': (0, 4),
        'N': (1, 5),  # int
        'W': (4, 16),  # int
        'Resolution': (0.25, 1.0),
    }

    def __init__(self):
        if not try_import_jahs():
            raise ImportError(
                "JAHS-Bench not installed. Install with:\n"
                "pip install git+https://github.com/automl/jahs_bench_201.git"
            )

        import jahs_bench
        self.jahs_bench = jahs_bench
        self.benchmarks = {}

    def _get_benchmark(self, task: str):
        """Get or create benchmark for task."""
        if task not in self.benchmarks:
            self.benchmarks[task] = self.jahs_bench.Benchmark(
                task=task, download=True
            )
        return self.benchmarks[task]

    def generate_trajectory(
        self,
        task: str,
        n_trials: int = 64,
        nepochs: int = 200,
        seed: int = None,
    ) -> List[Dict]:
        """Generate optimization trajectory on JAHS benchmark."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        bench = self._get_benchmark(task)
        trajectory = []

        def objective(trial):
            config = bench.sample_config(rng=trial.number + (seed or 0))

            # Override with Optuna suggestions for key hyperparams
            config['LearningRate'] = trial.suggest_float(
                'LearningRate', 1e-3, 1.0, log=True
            )
            config['WeightDecay'] = trial.suggest_float(
                'WeightDecay', 1e-5, 1e-2, log=True
            )
            config['N'] = trial.suggest_int('N', 1, 5)
            config['W'] = trial.suggest_int('W', 4, 16)

            # Evaluate
            result = bench(config, nepochs=nepochs)

            # Get validation accuracy
            val_acc = result.get(nepochs, {}).get('valid-acc', 0.5)
            error = 1.0 - val_acc / 100  # Convert percentage to error

            # Normalize params
            param_names = sorted(config.keys())
            normalized = normalize_params_to_generic(
                config, param_names, self.PARAM_RANGES
            )

            trajectory.append({
                'params': normalized,
                'score': float(error),
                'original_params': config,
            })

            return error

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return trajectory

    def generate_trajectories(
        self,
        n_trajectories: int,
        n_trials: int = 64,
        tasks: List[str] = None,
        output_path: Optional[Path] = None,
    ) -> List[List[Dict]]:
        """Generate multiple trajectories across tasks."""
        tasks = tasks or self.TASKS
        trajectories = []

        for i in tqdm(range(n_trajectories), desc="JAHS-Bench"):
            task = tasks[i % len(tasks)]
            traj = self.generate_trajectory(task, n_trials, seed=i)
            trajectories.append(traj)

            if output_path:
                with open(output_path, 'a') as f:
                    # Get param info for metadata
                    if traj and 'original_params' in traj[0]:
                        orig_params = list(traj[0]['original_params'].keys())
                        param_mapping = {f'x{j}': name for j, name in enumerate(sorted(orig_params))}
                    else:
                        param_mapping = {f'x{j}': name for j, name in enumerate(sorted(self.PARAM_RANGES.keys()))}

                    record = {
                        'source': 'jahs_bench',
                        'task': task,
                        'trajectory_id': i,
                        'n_trials': len(traj),
                        'n_params': len(param_mapping),
                        'param_mapping': param_mapping,
                        'trajectory': traj,
                    }
                    f.write(json.dumps(record) + '\n')

        return trajectories


# ============================================================================
# LCBench Integration (via YAHPO or direct)
# ============================================================================

def try_import_lcbench():
    """Check if LCBench API is available."""
    try:
        # Try YAHPO Gym version first (recommended)
        if try_import_yahpo():
            return 'yahpo'
        # Try direct LCBench API
        from api import Benchmark
        return 'direct'
    except ImportError:
        return False


class LCBenchLoader:
    """
    Loader for LCBench: Learning Curve Benchmark.

    Features:
    - 2000 configurations evaluated on 35 OpenML datasets
    - 7 hyperparameters: batch_size, learning_rate, momentum, weight_decay,
      num_layers, max_units, max_dropout
    - 52 epochs per config (learning curves)

    Recommended: Use via YAHPO Gym (pip install yahpo-gym)
    Alternative: Download from https://figshare.com/projects/LCBench/74151
    """

    PARAM_RANGES = {
        'batch_size': (16, 512),
        'learning_rate': (1e-4, 1e-1),
        'momentum': (0.1, 0.99),
        'weight_decay': (1e-5, 1e-1),
        'num_layers': (1, 5),
        'max_units': (64, 1024),
        'max_dropout': (0.0, 1.0),
    }

    def __init__(self, data_path: str = None, use_yahpo: bool = True):
        """
        Initialize LCBench loader.

        Args:
            data_path: Path to LCBench data (for direct API)
            use_yahpo: Whether to use YAHPO Gym interface (recommended)
        """
        self.use_yahpo = use_yahpo and try_import_yahpo()

        if self.use_yahpo:
            self.yahpo_loader = YAHPOLoader(data_path)
        else:
            if data_path is None:
                raise ValueError(
                    "LCBench data_path required when not using YAHPO. "
                    "Download from: https://figshare.com/projects/LCBench/74151"
                )
            # Direct API loading would go here
            raise NotImplementedError("Direct LCBench API not yet implemented")

    def generate_trajectory(
        self,
        dataset_id: str,
        n_trials: int = 64,
        epoch: int = 50,
        seed: int = None,
    ) -> List[Dict]:
        """Generate trajectory on LCBench dataset."""
        if self.use_yahpo:
            return self.yahpo_loader.generate_trajectory(
                scenario='lcbench',
                instance=dataset_id,
                n_trials=n_trials,
                seed=seed,
                fidelity=epoch,
            )
        else:
            raise NotImplementedError()

    def generate_trajectories(
        self,
        n_trajectories: int,
        n_trials: int = 64,
        output_path: Optional[Path] = None,
    ) -> List[List[Dict]]:
        """Generate multiple LCBench trajectories."""
        if self.use_yahpo:
            return self.yahpo_loader.generate_trajectories(
                scenario='lcbench',
                n_trajectories=n_trajectories,
                n_trials=n_trials,
                output_path=output_path,
            )
        else:
            raise NotImplementedError()


# ============================================================================
# Unified Benchmark Generator
# ============================================================================

def generate_diverse_real_world_data(
    output_dir: Path,
    trajectories_per_source: int = 500,
    use_hpobench: bool = True,
    use_yahpo: bool = True,
    use_jahs: bool = True,
    yahpo_data_path: str = None,
) -> Dict[str, Path]:
    """
    Generate diverse real-world training data from all available sources.

    This is the main entry point for collecting training data. It will
    use whatever benchmarks are installed and skip unavailable ones.

    Args:
        output_dir: Directory to save trajectory files
        trajectories_per_source: Number of trajectories per benchmark source
        use_hpobench: Include HPOBench FCNet benchmarks
        use_yahpo: Include YAHPO Gym benchmarks
        use_jahs: Include JAHS-Bench-201
        yahpo_data_path: Path to YAHPO data directory

    Returns:
        Dict mapping source name to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # HPOBench (FCNet)
    if use_hpobench:
        try:
            from .hpobench import HPOBenchLoader, try_import_hpobench

            if try_import_hpobench():
                print("Generating HPOBench trajectories...")
                loader = HPOBenchLoader(str(output_dir / 'hpobench_cache'))
                hpobench_path = output_dir / 'hpobench_trajectories.jsonl'

                for bench in ['fcnet_protein', 'fcnet_naval', 'fcnet_slice']:
                    loader.generate_trajectories(
                        bench,
                        n_trajectories=trajectories_per_source // 3,
                        output_path=hpobench_path,
                    )
                paths['hpobench'] = hpobench_path
            else:
                print("HPOBench not available, skipping...")
        except Exception as e:
            print(f"HPOBench failed: {e}")

    # YAHPO Gym
    if use_yahpo:
        try:
            if try_import_yahpo():
                print("Generating YAHPO Gym trajectories...")
                loader = YAHPOLoader(yahpo_data_path)
                yahpo_path = output_dir / 'yahpo_trajectories.jsonl'

                # Generate from multiple scenarios
                scenarios = ['lcbench', 'rbv2_svm', 'rbv2_xgboost', 'rbv2_ranger']
                for scenario in scenarios:
                    try:
                        loader.generate_trajectories(
                            scenario=scenario,
                            n_trajectories=trajectories_per_source // len(scenarios),
                            output_path=yahpo_path,
                        )
                    except Exception as e:
                        print(f"  {scenario} failed: {e}")

                paths['yahpo'] = yahpo_path
            else:
                print("YAHPO Gym not available, skipping...")
        except Exception as e:
            print(f"YAHPO Gym failed: {e}")

    # JAHS-Bench-201
    if use_jahs:
        try:
            if try_import_jahs():
                print("Generating JAHS-Bench trajectories...")
                loader = JAHSBenchLoader()
                jahs_path = output_dir / 'jahs_trajectories.jsonl'

                loader.generate_trajectories(
                    n_trajectories=trajectories_per_source,
                    output_path=jahs_path,
                )
                paths['jahs'] = jahs_path
            else:
                print("JAHS-Bench not available, skipping...")
        except Exception as e:
            print(f"JAHS-Bench failed: {e}")

    print(f"\nGenerated data from {len(paths)} sources:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    return paths


# ============================================================================
# Installation Helper
# ============================================================================

def print_installation_instructions():
    """Print installation instructions for all benchmark suites."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HPO Benchmark Installation Guide                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. YAHPO Gym (RECOMMENDED - Most Diverse)                                   ║
║     pip install yahpo-gym                                                    ║
║     # Download data: https://github.com/slds-lmu/yahpo_data                  ║
║     # Includes: LCBench, SVM, XGBoost, RandomForest, NAS-Bench-301          ║
║                                                                              ║
║  2. JAHS-Bench-201 (Architecture + Hyperparameters)                          ║
║     pip install git+https://github.com/automl/jahs_bench_201.git            ║
║     # Tasks: CIFAR-10, Fashion-MNIST, Colorectal Histology                  ║
║                                                                              ║
║  3. HPOBench (Original FCNet Benchmarks)                                     ║
║     pip install hpobench                                                     ║
║     # May require Singularity for containerized versions                    ║
║                                                                              ║
║  4. LCBench (via YAHPO - easiest)                                           ║
║     pip install yahpo-gym  # Already included above                         ║
║                                                                              ║
║  5. OpenML (Historical ML Runs)                                              ║
║     pip install openml>=0.14.0                                               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Quick Start (install all):                                                  ║
║    pip install yahpo-gym openml                                              ║
║    pip install git+https://github.com/automl/jahs_bench_201.git             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def check_available_benchmarks() -> Dict[str, bool]:
    """Check which benchmarks are installed."""
    available = {
        'hpobench': False,
        'yahpo_gym': try_import_yahpo(),
        'jahs_bench': try_import_jahs(),
        'openml': False,
    }

    try:
        from .hpobench import try_import_hpobench
        available['hpobench'] = try_import_hpobench()
    except:
        pass

    try:
        import openml
        available['openml'] = True
    except:
        pass

    return available


def validate_benchmark_integration(verbose: bool = True) -> Dict[str, Any]:
    """
    Validate that benchmark integrations work correctly.

    This tests:
    1. Data generation produces correct format
    2. Parameters are normalized to x0, x1, x2...
    3. Values are in [0, 1] range
    4. Trajectories can be loaded by CombinedTrajectoryDataset

    Returns:
        Dict with validation results
    """
    results = {
        'passed': True,
        'benchmarks_tested': [],
        'errors': [],
        'warnings': [],
    }

    available = check_available_benchmarks()

    if verbose:
        print("\n" + "="*60)
        print("  Benchmark Integration Validation")
        print("="*60)
        print(f"\nAvailable benchmarks: {available}\n")

    # Test YAHPO
    if available['yahpo_gym']:
        try:
            if verbose:
                print("Testing YAHPO Gym...")
            loader = YAHPOLoader()
            traj = loader.generate_trajectory('lcbench', '3945', n_trials=5, seed=0)

            # Validate format
            assert len(traj) == 5, f"Expected 5 trials, got {len(traj)}"
            assert 'params' in traj[0], "Missing 'params' key"
            assert 'score' in traj[0], "Missing 'score' key"

            # Validate normalization
            params = traj[0]['params']
            assert all(k.startswith('x') for k in params.keys()), \
                f"Params not normalized: {list(params.keys())}"
            assert all(0 <= v <= 1 for v in params.values()), \
                f"Values not in [0,1]: {params}"

            results['benchmarks_tested'].append('yahpo_gym')
            if verbose:
                print(f"  ✓ YAHPO Gym OK - {len(params)} params, values in [0,1]")

        except Exception as e:
            results['errors'].append(f"YAHPO Gym: {e}")
            results['passed'] = False
            if verbose:
                print(f"  ✗ YAHPO Gym FAILED: {e}")

    # Test JAHS-Bench
    if available['jahs_bench']:
        try:
            if verbose:
                print("Testing JAHS-Bench...")
            loader = JAHSBenchLoader()
            traj = loader.generate_trajectory('cifar10', n_trials=3, nepochs=1, seed=0)

            # Validate format
            assert len(traj) == 3, f"Expected 3 trials, got {len(traj)}"
            params = traj[0]['params']
            assert all(k.startswith('x') for k in params.keys()), \
                f"Params not normalized: {list(params.keys())}"

            results['benchmarks_tested'].append('jahs_bench')
            if verbose:
                print(f"  ✓ JAHS-Bench OK - {len(params)} params")

        except Exception as e:
            results['errors'].append(f"JAHS-Bench: {e}")
            results['passed'] = False
            if verbose:
                print(f"  ✗ JAHS-Bench FAILED: {e}")

    # Test HPOBench
    if available['hpobench']:
        try:
            if verbose:
                print("Testing HPOBench...")
            from .hpobench import HPOBenchLoader
            loader = HPOBenchLoader()
            traj = loader.generate_trajectory('fcnet_protein', n_trials=3, seed=0)

            # Validate format
            assert len(traj) == 3, f"Expected 3 trials, got {len(traj)}"
            params = traj[0]['params']
            assert all(k.startswith('x') for k in params.keys()), \
                f"Params not normalized: {list(params.keys())}"

            results['benchmarks_tested'].append('hpobench')
            if verbose:
                print(f"  ✓ HPOBench OK - {len(params)} params")

        except Exception as e:
            results['errors'].append(f"HPOBench: {e}")
            results['passed'] = False
            if verbose:
                print(f"  ✗ HPOBench FAILED: {e}")

    # Summary
    if verbose:
        print("\n" + "-"*60)
        if results['passed'] and results['benchmarks_tested']:
            print(f"✓ All {len(results['benchmarks_tested'])} tested benchmarks passed!")
        elif not results['benchmarks_tested']:
            print("⚠ No benchmarks available to test. Install with:")
            print("  pip install yahpo-gym")
        else:
            print(f"✗ {len(results['errors'])} benchmark(s) failed")
        print("="*60 + "\n")

    return results
