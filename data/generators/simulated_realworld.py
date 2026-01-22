"""
Simulated Real-World Data Generator

Generates optimization trajectories using realistic ML hyperparameter spaces.
This is a FALLBACK when HPOBench/YAHPO are not available.

While not as good as actual real-world data, these trajectories better
represent real HPO tasks than pure synthetic functions (GP, symbolic).

Key differences from synthetic:
1. Uses realistic parameter names and ranges (learning_rate, batch_size, etc.)
2. Includes categorical parameters (optimizer type, activation functions)
3. Uses parameter relationships that mimic real ML training dynamics
4. Includes realistic score distributions (validation loss patterns)
"""

import numpy as np
import optuna
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from tqdm import tqdm
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class SimulatedMLConfig:
    """Configuration for a simulated ML hyperparameter space."""
    name: str
    params: Dict[str, Dict[str, Any]]
    n_trajectories: int = 200
    trials_per_trajectory: int = 64

    # Simulation parameters
    noise_std: float = 0.05
    optimal_score: float = 0.1  # Best achievable score
    baseline_score: float = 0.8  # Random baseline score


# Pre-defined ML hyperparameter configurations
NEURAL_NETWORK_CONFIG = SimulatedMLConfig(
    name="neural_network_classification",
    params={
        "learning_rate": {"type": "float", "low": 1e-5, "high": 0.1, "log": True},
        "batch_size": {"type": "int", "low": 16, "high": 512, "log": True},
        "num_layers": {"type": "int", "low": 1, "high": 8},
        "hidden_size": {"type": "int", "low": 32, "high": 1024, "log": True},
        "dropout": {"type": "float", "low": 0.0, "high": 0.7},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 0.1, "log": True},
        "optimizer": {"type": "categorical", "choices": ["adam", "sgd", "adamw", "rmsprop"]},
        "activation": {"type": "categorical", "choices": ["relu", "gelu", "tanh", "silu"]},
    },
    n_trajectories=400,
)

XGBOOST_CONFIG = SimulatedMLConfig(
    name="xgboost_tuning",
    params={
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "n_estimators": {"type": "int", "low": 50, "high": 1000, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 10},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
    n_trajectories=300,
)

RANDOM_FOREST_CONFIG = SimulatedMLConfig(
    name="random_forest_tuning",
    params={
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 5, "high": 50},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", "0.5", "0.8"]},
        "bootstrap": {"type": "categorical", "choices": ["true", "false"]},
    },
    n_trajectories=200,
)

SVM_CONFIG = SimulatedMLConfig(
    name="svm_tuning",
    params={
        "C": {"type": "float", "low": 1e-4, "high": 1000.0, "log": True},
        "gamma": {"type": "float", "low": 1e-6, "high": 10.0, "log": True},
        "kernel": {"type": "categorical", "choices": ["rbf", "poly", "sigmoid"]},
        "degree": {"type": "int", "low": 2, "high": 5},
    },
    n_trajectories=200,
)

TRANSFORMER_CONFIG = SimulatedMLConfig(
    name="transformer_finetuning",
    params={
        "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        "batch_size": {"type": "int", "low": 8, "high": 64, "log": True},
        "num_epochs": {"type": "int", "low": 1, "high": 10},
        "warmup_ratio": {"type": "float", "low": 0.0, "high": 0.2},
        "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
        "max_grad_norm": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        "scheduler": {"type": "categorical", "choices": ["linear", "cosine", "constant"]},
        "optimizer": {"type": "categorical", "choices": ["adamw", "adafactor"]},
    },
    n_trajectories=200,
)

DEFAULT_CONFIGS = [
    NEURAL_NETWORK_CONFIG,
    XGBOOST_CONFIG,
    RANDOM_FOREST_CONFIG,
    SVM_CONFIG,
    TRANSFORMER_CONFIG,
]


class SimulatedMLObjective:
    """
    Creates a simulated ML objective function with realistic behavior.

    The function is designed to:
    1. Have a global optimum at reasonable hyperparameter values
    2. Show realistic sensitivity to different parameters
    3. Include interactions between parameters (e.g., lr and batch_size)
    4. Penalize extreme values appropriately
    """

    def __init__(
        self,
        config: SimulatedMLConfig,
        seed: int = None,
    ):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Generate random "optimal" configuration
        self.optimal_values = self._generate_optimal_config()

        # Parameter importance weights (randomly varied)
        self.importance = {}
        for name in config.params:
            self.importance[name] = self.rng.uniform(0.5, 2.0)

        # Normalize importance
        total_importance = sum(self.importance.values())
        for name in self.importance:
            self.importance[name] /= total_importance

    def _generate_optimal_config(self) -> Dict[str, Any]:
        """Generate a random optimal configuration."""
        optimal = {}
        for name, spec in self.config.params.items():
            if spec["type"] == "categorical":
                optimal[name] = self.rng.choice(spec["choices"])
            elif spec["type"] == "int":
                low, high = spec["low"], spec["high"]
                if spec.get("log", False):
                    val = np.exp(self.rng.uniform(np.log(low), np.log(high)))
                else:
                    val = self.rng.uniform(low, high)
                optimal[name] = int(round(val))
            else:  # float
                low, high = spec["low"], spec["high"]
                if spec.get("log", False):
                    val = np.exp(self.rng.uniform(np.log(low), np.log(high)))
                else:
                    val = self.rng.uniform(low, high)
                optimal[name] = val
        return optimal

    def _normalize_value(self, name: str, value: Any) -> float:
        """Normalize a parameter value to [0, 1]."""
        spec = self.config.params[name]

        if spec["type"] == "categorical":
            choices = spec["choices"]
            if value == self.optimal_values[name]:
                return 0.0  # Optimal
            else:
                return self.rng.uniform(0.3, 1.0)  # Non-optimal categorical

        low, high = spec["low"], spec["high"]

        if spec.get("log", False):
            value = np.log(value + 1e-10)
            low, high = np.log(low + 1e-10), np.log(high + 1e-10)

        return (value - low) / (high - low + 1e-10)

    def __call__(self, params: Dict[str, Any]) -> float:
        """Evaluate the simulated objective."""
        # Calculate distance from optimal for each parameter
        total_loss = 0.0

        for name, value in params.items():
            if name not in self.config.params:
                continue

            spec = self.config.params[name]
            optimal = self.optimal_values[name]
            importance = self.importance[name]

            if spec["type"] == "categorical":
                # Categorical: penalty for wrong choice
                if value != optimal:
                    loss = self.rng.uniform(0.1, 0.4) * importance
                else:
                    loss = 0.0
            else:
                # Numerical: squared distance
                norm_val = self._normalize_value(name, value)
                norm_opt = self._normalize_value(name, optimal)
                loss = ((norm_val - norm_opt) ** 2) * importance

            total_loss += loss

        # Add interactions (e.g., learning rate and batch size)
        if "learning_rate" in params and "batch_size" in params:
            lr = params["learning_rate"]
            bs = params["batch_size"]
            # Penalty for high lr with small batch or low lr with large batch
            lr_norm = self._normalize_value("learning_rate", lr)
            bs_norm = self._normalize_value("batch_size", bs)
            interaction = abs(lr_norm - bs_norm) * 0.1
            total_loss += interaction

        # Scale to realistic range
        score = self.config.optimal_score + total_loss * (
            self.config.baseline_score - self.config.optimal_score
        )

        # Add noise
        score += self.rng.normal(0, self.config.noise_std)

        # Clip to valid range
        return float(np.clip(score, 0.0, 1.0))


class SimulatedRealWorldGenerator:
    """
    Generates simulated real-world optimization trajectories.

    These trajectories use realistic ML hyperparameter spaces and
    simulated objective functions that mimic real training dynamics.
    """

    def __init__(
        self,
        configs: List[SimulatedMLConfig] = None,
        seed: int = 42,
    ):
        self.configs = configs or DEFAULT_CONFIGS
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_trajectory(
        self,
        config: SimulatedMLConfig,
        trajectory_seed: int,
    ) -> Dict[str, Any]:
        """Generate a single optimization trajectory."""
        # Create objective function
        objective = SimulatedMLObjective(config, seed=trajectory_seed)

        # Track trajectory
        trajectory = []

        def optuna_objective(trial: optuna.Trial) -> float:
            params = {}

            for name, spec in config.params.items():
                if spec["type"] == "float":
                    val = trial.suggest_float(
                        name, spec["low"], spec["high"],
                        log=spec.get("log", False)
                    )
                elif spec["type"] == "int":
                    val = trial.suggest_int(
                        name, spec["low"], spec["high"],
                        log=spec.get("log", False)
                    )
                elif spec["type"] == "categorical":
                    val = trial.suggest_categorical(name, spec["choices"])
                else:
                    val = trial.suggest_float(name, spec["low"], spec["high"])

                params[name] = val

            score = objective(params)

            # Normalize params to generic names (x0, x1, ...) for training
            param_names = list(config.params.keys())
            normalized_params = {}
            for i, name in enumerate(param_names):
                val = params[name]
                # Normalize numerical values to [0, 1]
                spec = config.params[name]
                if spec["type"] in ("float", "int"):
                    low, high = spec["low"], spec["high"]
                    if spec.get("log", False):
                        val = (np.log(val + 1e-10) - np.log(low + 1e-10)) / (
                            np.log(high + 1e-10) - np.log(low + 1e-10)
                        )
                    else:
                        val = (val - low) / (high - low + 1e-10)
                    val = float(np.clip(val, 0.0, 1.0))
                elif spec["type"] == "categorical":
                    # Map categorical to index normalized to [0, 1]
                    choices = spec["choices"]
                    idx = choices.index(val) if val in choices else 0
                    val = idx / max(1, len(choices) - 1)

                normalized_params[f"x{i}"] = val

            trajectory.append({
                "params": normalized_params,
                "score": score,
                "original_params": params,
            })

            return score

        # Run optimization
        sampler = optuna.samplers.TPESampler(seed=trajectory_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            optuna_objective,
            n_trials=config.trials_per_trajectory,
            show_progress_bar=False,
            catch=(Exception,),
        )

        return {
            "source": "simulated_real_world",
            "config_name": config.name,
            "n_dims": len(config.params),
            "n_trials": len(trajectory),
            "best_score": min(t["score"] for t in trajectory) if trajectory else 1.0,
            "param_mapping": {f"x{i}": name for i, name in enumerate(config.params.keys())},
            "trajectory": trajectory,
        }

    def generate_all(
        self,
        output_path: Path = None,
        show_progress: bool = True,
    ) -> List[Dict]:
        """Generate trajectories for all configurations."""
        all_trajectories = []

        iterator = self.configs
        if show_progress:
            iterator = tqdm(iterator, desc="Generating simulated real-world data")

        trajectory_idx = 0
        for config in iterator:
            for i in range(config.n_trajectories):
                try:
                    traj = self.generate_trajectory(
                        config,
                        trajectory_seed=self.seed + trajectory_idx,
                    )
                    all_trajectories.append(traj)

                    if output_path:
                        with open(output_path, 'a') as f:
                            f.write(json.dumps(traj) + '\n')

                    trajectory_idx += 1
                except Exception as e:
                    warnings.warn(f"Failed to generate trajectory {trajectory_idx}: {e}")
                    trajectory_idx += 1

        return all_trajectories


def generate_simulated_real_world_data(
    output_dir: Path,
    configs: List[Dict] = None,
    seed: int = 42,
) -> Tuple[Path, int]:
    """
    Generate simulated real-world data from YAML config.

    Args:
        output_dir: Directory to save generated data
        configs: List of config dicts from YAML (or None for defaults)
        seed: Random seed

    Returns:
        Tuple of (output_path, n_trajectories)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simulated_real_world.jsonl"

    # Clear existing file
    if output_path.exists():
        output_path.unlink()

    # Convert YAML configs to SimulatedMLConfig objects
    if configs:
        ml_configs = []
        for cfg in configs:
            ml_configs.append(SimulatedMLConfig(
                name=cfg["name"],
                params=cfg["params"],
                n_trajectories=cfg.get("n_trajectories", 200),
            ))
    else:
        ml_configs = DEFAULT_CONFIGS

    # Generate data
    generator = SimulatedRealWorldGenerator(configs=ml_configs, seed=seed)
    trajectories = generator.generate_all(output_path=output_path)

    print(f"Generated {len(trajectories)} simulated real-world trajectories")
    return output_path, len(trajectories)


def get_data_statistics(trajectories: List[Dict]) -> Dict[str, Any]:
    """Get statistics about generated data."""
    stats = {
        "total_trajectories": len(trajectories),
        "by_config": {},
        "total_trials": 0,
        "avg_best_score": 0.0,
        "dimensions": set(),
    }

    for traj in trajectories:
        config_name = traj.get("config_name", "unknown")
        if config_name not in stats["by_config"]:
            stats["by_config"][config_name] = 0
        stats["by_config"][config_name] += 1

        stats["total_trials"] += traj.get("n_trials", 0)
        stats["avg_best_score"] += traj.get("best_score", 1.0)
        stats["dimensions"].add(traj.get("n_dims", 0))

    if trajectories:
        stats["avg_best_score"] /= len(trajectories)

    stats["dimensions"] = sorted(list(stats["dimensions"]))

    return stats
