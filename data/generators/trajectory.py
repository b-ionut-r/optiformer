"""
Teacher Algorithm Trajectory Generator

Runs Optuna TPE on synthetic functions to generate training trajectories.
This is the "algorithm distillation" step - we learn from TPE's behavior.
"""

import optuna
import numpy as np
from typing import List, Callable, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""
    n_trials: int = 64
    noise_std: float = 0.02
    bounds: Tuple[float, float] = (0.0, 1.0)
    sampler: str = "tpe"  # tpe, random, cmaes
    seed: Optional[int] = None


class TrajectoryGenerator:
    """Generates optimization trajectories using teacher algorithms."""

    def __init__(self, config: TrajectoryConfig, n_dims: int):
        self.config = config
        self.n_dims = n_dims

    def _create_sampler(self, seed: int = None) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on config."""
        if self.config.sampler == "tpe":
            return optuna.samplers.TPESampler(seed=seed)
        elif self.config.sampler == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        elif self.config.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=seed)
        else:
            return optuna.samplers.TPESampler(seed=seed)

    def generate_trajectory(
        self,
        objective_fn: Callable,
        seed: int = None
    ) -> List[Dict[str, Any]]:
        """
        Run optimization and record the trajectory.

        Args:
            objective_fn: Function to optimize
            seed: Random seed for reproducibility

        Returns:
            List of dicts: [{"params": {...}, "score": float}, ...]
        """
        trajectory = []
        low, high = self.config.bounds

        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            params = {}
            for i in range(self.n_dims):
                params[f"x{i}"] = trial.suggest_float(f"x{i}", low, high)

            # Evaluate with noise
            x = np.array([params[f"x{i}"] for i in range(self.n_dims)])
            y = objective_fn(x.reshape(1, -1))
            if isinstance(y, np.ndarray):
                y = y[0]
            y = float(y)

            # Add observation noise
            if self.config.noise_std > 0:
                y_noisy = y + np.random.normal(0, self.config.noise_std)
                y_noisy = np.clip(y_noisy, 0, 1)
            else:
                y_noisy = y

            # Record
            trajectory.append({
                "params": params,
                "score": float(y_noisy),
                "true_score": float(y),
            })

            return y_noisy

        # Run optimization
        sampler = self._create_sampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=False,
            catch=(Exception,),
        )

        return trajectory

    def generate_trajectories_batch(
        self,
        functions: List[Callable],
        seeds: Optional[List[int]] = None,
        desc: str = "Generating trajectories"
    ) -> List[List[Dict]]:
        """Generate trajectories for multiple functions."""
        if seeds is None:
            seeds = list(range(len(functions)))

        trajectories = []
        for i, (func, seed) in enumerate(tqdm(
            zip(functions, seeds),
            total=len(functions),
            desc=desc
        )):
            traj = self.generate_trajectory(func, seed=seed)
            trajectories.append(traj)

        return trajectories


def generate_all_trajectories(
    functions: List[Tuple[int, Callable]],
    config: TrajectoryConfig,
    output_path: Path,
    desc: str = "Generating trajectories"
) -> int:
    """
    Generate trajectories for all functions and save to JSONL.

    Args:
        functions: List of (n_dims, function) tuples
        config: Trajectory generation config
        output_path: Path to save JSONL file
        desc: Progress bar description

    Returns:
        Number of trajectories generated
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, 'w') as f:
        for i, (n_dims, func) in enumerate(tqdm(
            functions,
            desc=desc
        )):
            generator = TrajectoryGenerator(config, n_dims)
            trajectory = generator.generate_trajectory(func, seed=i)

            record = {
                "function_id": i,
                "n_dims": n_dims,
                "n_trials": len(trajectory),
                "best_score": min(t["score"] for t in trajectory),
                "trajectory": trajectory,
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    return count


def load_trajectories(
    path: Path,
    max_trajectories: Optional[int] = None
) -> List[Dict]:
    """
    Load trajectories from JSONL file.

    Args:
        path: Path to JSONL file
        max_trajectories: Maximum number to load

    Returns:
        List of trajectory records
    """
    trajectories = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_trajectories and i >= max_trajectories:
                break
            trajectories.append(json.loads(line))
    return trajectories


class DataAugmenter:
    """Augmentation strategies for trajectories."""

    @staticmethod
    def shuffle_history(trajectory: List[Dict], keep_last: int = 1) -> List[Dict]:
        """
        Shuffle early trials to enforce permutation invariance.

        Args:
            trajectory: Original trajectory
            keep_last: Number of last trials to keep in order

        Returns:
            Augmented trajectory
        """
        if len(trajectory) <= keep_last:
            return trajectory.copy()

        prefix = trajectory[:-keep_last]
        suffix = trajectory[-keep_last:]

        np.random.shuffle(prefix)
        return prefix + suffix

    @staticmethod
    def subsample(
        trajectory: List[Dict],
        n_samples: int,
        keep_best: bool = True
    ) -> List[Dict]:
        """
        Subsample trajectory to shorter length.

        Args:
            trajectory: Original trajectory
            n_samples: Target length
            keep_best: If True, always include the best trial

        Returns:
            Subsampled trajectory
        """
        if len(trajectory) <= n_samples:
            return trajectory.copy()

        if keep_best:
            # Find best trial
            best_idx = np.argmin([t["score"] for t in trajectory])
            indices = [best_idx]

            # Sample remaining
            remaining = list(set(range(len(trajectory))) - {best_idx})
            indices.extend(np.random.choice(
                remaining,
                size=min(n_samples - 1, len(remaining)),
                replace=False
            ))
            indices = sorted(indices)
        else:
            indices = sorted(np.random.choice(
                len(trajectory),
                size=n_samples,
                replace=False
            ))

        return [trajectory[i] for i in indices]

    @staticmethod
    def add_noise(trajectory: List[Dict], noise_std: float = 0.01) -> List[Dict]:
        """
        Add small noise to parameter values.

        Args:
            trajectory: Original trajectory
            noise_std: Standard deviation of noise

        Returns:
            Noisy trajectory
        """
        noisy = []
        for trial in trajectory:
            new_trial = trial.copy()
            new_params = {}
            for k, v in trial["params"].items():
                if isinstance(v, (int, float)):
                    new_params[k] = v + np.random.normal(0, noise_std)
                else:
                    new_params[k] = v
            new_trial["params"] = new_params
            noisy.append(new_trial)
        return noisy
