"""
Combined Dataset

Merges synthetic and real-world optimization trajectories for training.
Using mixed data helps the model generalize better to real HPO tasks.

IMPORTANT: Real-world data is CRITICAL for model generalization.
Models trained on synthetic-only data WILL FAIL on real HPO tasks.

Recommended data composition:
- Production: 70% real-world, 30% synthetic
- Smoke test: 60% real-world, 40% synthetic

If real-world sources (HPOBench, YAHPO) are unavailable, the system
falls back to "simulated real-world" data using realistic ML hyperparameter
spaces. This is better than pure synthetic but not as good as actual data.
"""

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import random
import numpy as np
import warnings
import logging

from data.tokenizer import SequenceTokenizer, Trial
from .synthetic import TrajectoryDataset, collate_fn

logger = logging.getLogger(__name__)


class CombinedTrajectoryDataset(Dataset):
    """
    Dataset combining multiple trajectory sources.

    Supports:
    - Synthetic trajectories (GP, symbolic)
    - HPOBench trajectories
    - OpenML traces
    - Custom real-world traces
    """

    def __init__(
        self,
        data_paths: List[Path],
        tokenizer: SequenceTokenizer,
        max_length: int = 512,
        shuffle_history: bool = True,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize combined dataset.

        Args:
            data_paths: List of JSONL files to load
            tokenizer: Sequence tokenizer
            max_length: Maximum sequence length
            shuffle_history: Whether to shuffle trial history
            weights: Optional sampling weights for each source
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_history = shuffle_history
        self.param_names = list(tokenizer.param_specs.keys())

        # Load all trajectories
        self.trajectories = []
        self.source_indices = []  # Track which source each trajectory came from
        self._dimension_warnings = set()  # Track dimension mismatch warnings

        for idx, path in enumerate(data_paths):
            path = Path(path)
            if path.exists():
                with open(path, 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        self.trajectories.append(record)
                        self.source_indices.append(idx)

                        # Check for dimension mismatch (first trajectory only per file)
                        if len(self.trajectories) == 1 or self.source_indices[-2] != idx:
                            self._check_dimensions(record, path)

    def _check_dimensions(self, record: Dict, path: Path):
        """Warn if trajectory has more dimensions than tokenizer supports."""
        trajectory = record.get('trajectory', [])
        if trajectory:
            first_trial = trajectory[0]
            data_params = set(first_trial.get('params', {}).keys())
            tokenizer_params = set(self.param_names)

            # Find params in data that tokenizer doesn't know about
            missing = data_params - tokenizer_params
            if missing and str(path) not in self._dimension_warnings:
                self._dimension_warnings.add(str(path))
                import warnings
                warnings.warn(
                    f"Data from {path.name} has params {missing} not in tokenizer. "
                    f"These will be ignored. Tokenizer has: {sorted(tokenizer_params)[:5]}... "
                    f"Consider increasing n_dims when creating tokenizer."
                )

        # Set up sampling weights
        if weights:
            self.weights = weights
        else:
            self.weights = [1.0] * len(data_paths)

    def __len__(self) -> int:
        return len(self.trajectories)

    def _convert_to_trials(self, trajectory: List[Dict]) -> List[Trial]:
        """Convert trajectory dicts to Trial objects."""
        trials = []
        for t in trajectory:
            params = {}
            for name in self.param_names:
                if name in t.get('params', {}):
                    params[name] = t['params'][name]
            if params:  # Only add if we have matching params
                trials.append(Trial(params=params, score=t.get('score', 0.5)))
        return trials

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.trajectories[idx]
        trajectory = record.get('trajectory', [])

        # Convert to Trial objects
        trials = self._convert_to_trials(trajectory)

        if len(trials) == 0:
            # Return dummy data if no valid trials
            return {
                'input_ids': torch.zeros(10, dtype=torch.long),
                'labels': torch.zeros(10, dtype=torch.long),
            }

        # Shuffle history
        if self.shuffle_history and len(trials) > 1:
            last = trials[-1]
            rest = trials[:-1]
            random.shuffle(rest)
            trials = rest + [last]

        # Compute target regret
        best_score = min(t.score for t in trials)
        if best_score < 0.1:
            target_regret = 0
        elif best_score < 0.3:
            target_regret = 1
        else:
            target_regret = 2

        # Tokenize
        token_ids = self.tokenizer.encode_trajectory(trials, target_regret)

        # Truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Create input/label pairs
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def create_mixed_dataloaders(
    synthetic_train_path: Path,
    synthetic_val_path: Path,
    real_world_paths: List[Path],
    tokenizer: SequenceTokenizer,
    batch_size: int = 64,
    max_length: int = 512,
    synthetic_weight: float = 0.7,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders with mixed synthetic and real-world data.

    Args:
        synthetic_train_path: Path to synthetic training data
        synthetic_val_path: Path to synthetic validation data
        real_world_paths: Paths to real-world trajectory files
        tokenizer: Sequence tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        synthetic_weight: Weight for synthetic vs real data
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Combine all training paths
    train_paths = [synthetic_train_path] + real_world_paths

    # Calculate weights
    n_sources = len(train_paths)
    real_weight = (1 - synthetic_weight) / max(1, n_sources - 1)
    weights = [synthetic_weight] + [real_weight] * (n_sources - 1)

    # Create datasets
    train_dataset = CombinedTrajectoryDataset(
        data_paths=train_paths,
        tokenizer=tokenizer,
        max_length=max_length,
        shuffle_history=True,
        weights=weights,
    )

    val_dataset = TrajectoryDataset(
        data_path=synthetic_val_path,
        tokenizer=tokenizer,
        max_length=max_length,
        shuffle_history=False,
    )

    pad_token_id = tokenizer.vocab.pad_token_id

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        num_workers=num_workers,
    )

    return train_loader, val_loader


def generate_real_world_data(
    output_dir: Path,
    use_hpobench: bool = True,
    use_openml: bool = False,
    use_yahpo: bool = False,
    use_simulated_fallback: bool = True,
    hpobench_benchmarks: List[str] = None,
    yahpo_scenarios: List[str] = None,
    simulated_configs: List[Dict] = None,
    n_trajectories_per_benchmark: int = 200,
    min_real_world_trajectories: int = 500,
) -> Dict[str, Any]:
    """
    Generate real-world training data from various sources.

    IMPORTANT: This function will warn if insufficient real-world data
    is generated. Use the simulated fallback to ensure adequate data.

    Args:
        output_dir: Directory to save data
        use_hpobench: Whether to use HPOBench
        use_openml: Whether to use OpenML traces
        use_yahpo: Whether to use YAHPO Gym
        use_simulated_fallback: Whether to use simulated real-world as fallback
        hpobench_benchmarks: Which HPOBench benchmarks to use
        yahpo_scenarios: Which YAHPO scenarios to use
        simulated_configs: Configs for simulated real-world (from YAML)
        n_trajectories_per_benchmark: Trajectories per benchmark
        min_real_world_trajectories: Warn if below this threshold

    Returns:
        Dict with:
            - paths: Dict mapping source name to output path
            - n_trajectories: Total trajectories generated
            - sources_used: List of data sources that provided data
            - warnings: List of warnings
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "paths": {},
        "n_trajectories": 0,
        "sources_used": [],
        "warnings": [],
    }

    # Track trajectory counts per source
    source_counts = {}

    # =========================================================================
    # 1. Try HPOBench (highest quality real data)
    # =========================================================================
    if use_hpobench:
        try:
            from .hpobench import HPOBenchLoader, try_import_hpobench

            if try_import_hpobench():
                loader = HPOBenchLoader(data_dir=str(output_dir / 'hpobench_cache'))

                benchmarks = hpobench_benchmarks or ['fcnet_protein', 'fcnet_naval']
                hpobench_path = output_dir / 'hpobench_trajectories.jsonl'

                count = 0
                for bench_name in benchmarks:
                    print(f"Generating {n_trajectories_per_benchmark} trajectories from HPOBench/{bench_name}...")
                    loader.generate_trajectories(
                        benchmark_name=bench_name,
                        n_trajectories=n_trajectories_per_benchmark,
                        n_trials=64,
                        output_path=hpobench_path,
                    )
                    count += n_trajectories_per_benchmark

                result["paths"]['hpobench'] = hpobench_path
                source_counts['hpobench'] = count
                result["sources_used"].append('hpobench')
                print(f"  HPOBench: {count} trajectories")
            else:
                msg = "HPOBench not installed. Install with: pip install hpobench"
                result["warnings"].append(msg)
                print(f"  WARNING: {msg}")
        except Exception as e:
            msg = f"HPOBench generation failed: {e}"
            result["warnings"].append(msg)
            print(f"  WARNING: {msg}")

    # =========================================================================
    # 2. Try YAHPO Gym
    # =========================================================================
    if use_yahpo:
        try:
            # YAHPO integration would go here
            # For now, just note it's not implemented
            msg = "YAHPO integration not fully implemented yet"
            result["warnings"].append(msg)
            print(f"  WARNING: {msg}")
        except Exception as e:
            msg = f"YAHPO generation failed: {e}"
            result["warnings"].append(msg)

    # =========================================================================
    # 3. Try OpenML
    # =========================================================================
    if use_openml:
        try:
            from .hpobench import load_openml_traces

            openml_path = output_dir / 'openml_trajectories.jsonl'
            print("Loading OpenML traces...")
            trajectories = load_openml_traces(
                task_ids=[3, 12, 31, 53, 3917],
                n_configs_per_task=100,
                output_path=openml_path,
            )
            count = len(trajectories)
            result["paths"]['openml'] = openml_path
            source_counts['openml'] = count
            result["sources_used"].append('openml')
            print(f"  OpenML: {count} trajectories")
        except Exception as e:
            msg = f"OpenML loading failed: {e}"
            result["warnings"].append(msg)
            print(f"  WARNING: {msg}")

    # =========================================================================
    # 4. Calculate current total and check if fallback needed
    # =========================================================================
    current_total = sum(source_counts.values())
    result["n_trajectories"] = current_total

    # =========================================================================
    # 5. Use simulated real-world as fallback if needed
    # =========================================================================
    need_fallback = (
        use_simulated_fallback and
        current_total < min_real_world_trajectories
    )

    if need_fallback:
        try:
            from data.generators.simulated_realworld import (
                generate_simulated_real_world_data,
                DEFAULT_CONFIGS,
            )

            print(f"\n  Real-world data insufficient ({current_total} < {min_real_world_trajectories})")
            print("  Generating simulated real-world data as fallback...")

            simulated_path, simulated_count = generate_simulated_real_world_data(
                output_dir=output_dir,
                configs=simulated_configs,
                seed=42,
            )

            result["paths"]['simulated_real_world'] = simulated_path
            source_counts['simulated_real_world'] = simulated_count
            result["sources_used"].append('simulated_real_world')

            # Update total
            result["n_trajectories"] = sum(source_counts.values())

            msg = (
                f"Using simulated real-world data ({simulated_count} trajectories). "
                "For better results, install HPOBench or YAHPO."
            )
            result["warnings"].append(msg)
            print(f"  {msg}")

        except Exception as e:
            msg = f"Simulated real-world generation failed: {e}"
            result["warnings"].append(msg)
            print(f"  ERROR: {msg}")

    # =========================================================================
    # 6. Final check and warnings
    # =========================================================================
    final_total = result["n_trajectories"]

    if final_total == 0:
        critical_msg = (
            "CRITICAL: No real-world data generated! "
            "Model trained on synthetic-only data WILL FAIL on real HPO tasks. "
            "Install HPOBench (pip install hpobench) or enable simulated fallback."
        )
        result["warnings"].append(critical_msg)
        warnings.warn(critical_msg, UserWarning)
        print(f"\n  {critical_msg}")

    elif final_total < min_real_world_trajectories:
        warn_msg = (
            f"WARNING: Only {final_total} real-world trajectories generated "
            f"(recommended: {min_real_world_trajectories}+). "
            "Model may underperform on real HPO tasks."
        )
        result["warnings"].append(warn_msg)
        warnings.warn(warn_msg, UserWarning)
        print(f"\n  {warn_msg}")

    else:
        print(f"\n  Total real-world trajectories: {final_total}")
        print(f"  Sources: {', '.join(result['sources_used'])}")

    # Store breakdown
    result["source_counts"] = source_counts

    return result


def compute_data_statistics(
    synthetic_path: Path,
    real_world_paths: List[Path],
) -> Dict[str, Any]:
    """
    Compute statistics about the training data composition.

    Returns dict with trajectory counts, ratios, and warnings.
    """
    stats = {
        "synthetic_count": 0,
        "real_world_count": 0,
        "total_count": 0,
        "synthetic_ratio": 0.0,
        "real_world_ratio": 0.0,
        "warnings": [],
    }

    # Count synthetic
    if synthetic_path and Path(synthetic_path).exists():
        with open(synthetic_path, 'r') as f:
            stats["synthetic_count"] = sum(1 for _ in f)

    # Count real-world
    for path in real_world_paths:
        if path and Path(path).exists():
            with open(path, 'r') as f:
                stats["real_world_count"] += sum(1 for _ in f)

    stats["total_count"] = stats["synthetic_count"] + stats["real_world_count"]

    if stats["total_count"] > 0:
        stats["synthetic_ratio"] = stats["synthetic_count"] / stats["total_count"]
        stats["real_world_ratio"] = stats["real_world_count"] / stats["total_count"]

    # Check for issues
    if stats["real_world_ratio"] < 0.5:
        stats["warnings"].append(
            f"Real-world data ratio is low ({stats['real_world_ratio']:.1%}). "
            "Recommend >= 50% real-world data for good generalization."
        )

    if stats["real_world_count"] == 0:
        stats["warnings"].append(
            "No real-world data! Model will likely fail on real HPO tasks."
        )

    return stats
