"""
Smoke Test Phase 2: Data Generation Validation

Generates synthetic functions and optimization trajectories
for training the model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import json

from data.generators import (
    generate_gp_dataset,
    generate_symbolic_dataset,
    generate_all_trajectories,
    TrajectoryConfig,
)


def generate_yahpo_trajectories(output_path: Path, seed: int = 42) -> int:
    """Generate trajectories from YAHPO-Gym benchmarks (60% of total data)."""
    try:
        from data.datasets.benchmark_zoo import YAHPOLoader
        import subprocess
        
        # Check if YAHPO is available
        try:
            import yahpo_gym
        except ImportError:
            print("  WARNING: YAHPO Gym not installed. Skipping YAHPO data generation.")
            print("  Install with: pip install yahpo-gym")
            return 0

        # Setup local YAHPO data
        yahpo_data_path = Path("data/yahpo_data")
        if not yahpo_data_path.exists() or not any(yahpo_data_path.iterdir()):
            print(f"  Downloading YAHPO data to {yahpo_data_path}...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/slds-lmu/yahpo_data.git", str(yahpo_data_path)],
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                print(f"  WARNING: Failed to download YAHPO data: {e}")
                print("  Please download manually: https://github.com/slds-lmu/yahpo_data")
                return 0

        print(f"  Generating YAHPO trajectories...")
        # Pass the explicit data path to the loader
        loader = YAHPOLoader(data_path=str(yahpo_data_path.resolve()))
        total = 0

        # Generate 3000 total YAHPO trajectories across 4 scenarios
        scenarios = [
            ('lcbench', 1000),      # Learning curves, 7 params
            ('rbv2_xgboost', 1000), # XGBoost tuning, 14 params
            ('rbv2_svm', 500),      # SVM tuning, 6 params
            ('rbv2_ranger', 500),   # Random forest, 8 params
        ]

        for scenario, n_traj in scenarios:
            try:
                print(f"    Generating {n_traj} trajectories for {scenario}...")
                trajectories = loader.generate_trajectories(
                    scenario=scenario,
                    n_trajectories=n_traj,
                    n_trials=32,
                    output_path=output_path
                )
                total += len(trajectories)
            except Exception as e:
                print(f"    WARNING: Failed to generate {scenario}: {e}")

        return total
    except Exception as e:
        print(f"  WARNING: YAHPO generation failed: {e}")
        return 0


def generate_smoke_test_data(
    output_dir: Path,
    n_gp_functions: int = 500,
    n_symbolic_functions: int = 500,
    trials_per_function: int = 32,
    dims_list: list = None,
    seed: int = 42,
    comprehensive: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Generate synthetic data for smoke testing.

    Args:
        output_dir: Directory to save data
        n_gp_functions: Number of GP functions
        n_symbolic_functions: Number of symbolic functions
        trials_per_function: Trials per trajectory
        dims_list: List of dimensions to use
        seed: Random seed
        comprehensive: Whether to include YAHPO data

    Returns:
        (success, results_dict)
    """
    print("\nPhase 2: Data Generation")
    print("-" * 40)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dims_list = dims_list or [2, 3, 5]
    results = {
        'gp_functions': 0,
        'symbolic_functions': 0,
        'yahpo_trajectories': 0,
        'n_trajectories': 0,
        'errors': [],
    }

    np.random.seed(seed)

    if comprehensive:
        # Comprehensive mode: 40% synthetic (2000 total)
        n_gp_functions = 1000
        n_symbolic_functions = 1000
        print(f"  Comprehensive mode: Using {n_gp_functions} GP + {n_symbolic_functions} Symbolic functions")

    try:
        # Generate GP functions
        print(f"  Generating {n_gp_functions} GP functions...")
        gp_functions = generate_gp_dataset(
            n_functions=n_gp_functions,
            dims_list=dims_list,
            bounds=(-5.0, 5.0),
            seed=seed,
        )
        results['gp_functions'] = len(gp_functions)
        print(f"    Generated {len(gp_functions)} GP functions")

        # Generate symbolic functions
        print(f"  Generating {n_symbolic_functions} symbolic functions...")
        symbolic_functions = generate_symbolic_dataset(
            n_functions=n_symbolic_functions,
            dims_list=dims_list,
            bounds=(-5.0, 5.0),
            seed=seed + 1000,
        )
        results['symbolic_functions'] = len(symbolic_functions)
        print(f"    Generated {len(symbolic_functions)} symbolic functions")

        # Combine functions
        all_functions = gp_functions + symbolic_functions
        np.random.shuffle(all_functions)

        # Split into train/val
        n_val = max(100, len(all_functions) // 10)
        val_functions = all_functions[:n_val]
        train_functions = all_functions[n_val:]

        # Generate trajectories
        traj_config = TrajectoryConfig(
            n_trials=trials_per_function,
            noise_std=0.02,
            bounds=(0.0, 1.0),
            sampler="tpe",
        )

        print(f"  Generating training trajectories ({len(train_functions)} functions)...")
        train_path = output_dir / "train_trajectories.jsonl"
        n_train = generate_all_trajectories(
            functions=train_functions,
            config=traj_config,
            output_path=train_path,
            desc="Training trajectories",
        )

        print(f"  Generating validation trajectories ({len(val_functions)} functions)...")
        val_path = output_dir / "val_trajectories.jsonl"
        n_val = generate_all_trajectories(
            functions=val_functions,
            config=traj_config,
            output_path=val_path,
            desc="Validation trajectories",
        )

        results['n_trajectories'] = n_train + n_val
        results['n_train'] = n_train
        results['n_val'] = n_val
        results['train_path'] = str(train_path)
        results['val_path'] = str(val_path)

        print(f"    Total trajectories: {results['n_trajectories']}")
        print(f"    Train: {n_train}, Val: {n_val}")

        # Generate YAHPO data if comprehensive
        if comprehensive:
            n_yahpo = generate_yahpo_trajectories(train_path, seed=seed)
            results['yahpo_trajectories'] = n_yahpo
            results['n_trajectories'] += n_yahpo
            print(f"    Added {n_yahpo} YAHPO trajectories")

        # Verify data
        print("  Verifying generated data...")
        with open(train_path, 'r') as f:
            first_line = f.readline()
            first_record = json.loads(first_line)
            traj_length = len(first_record['trajectory'])

        if traj_length != trials_per_function:
            results['errors'].append(f"Trajectory length mismatch: {traj_length} != {trials_per_function}")

        # Save metadata
        metadata = {
            'n_gp_functions': n_gp_functions,
            'n_symbolic_functions': n_symbolic_functions,
            'trials_per_function': trials_per_function,
            'dims_list': dims_list,
            'n_train': n_train,
            'n_val': n_val,
            'seed': seed,
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        success = len(results['errors']) == 0 and results['n_trajectories'] > 0
        return success, results

    except Exception as e:
        results['errors'].append(str(e))
        print(f"  ERROR: {e}")
        return False, results


if __name__ == "__main__":
    output_dir = Path("./outputs/smoke_test/data")
    success, results = generate_smoke_test_data(
        output_dir,
        n_gp_functions=200,
        n_symbolic_functions=200,
        trials_per_function=32,
    )
    print(f"\nData generation: {'SUCCESS' if success else 'FAILED'}")
    if results['errors']:
        for err in results['errors']:
            print(f"  Error: {err}")
    sys.exit(0 if success else 1)
