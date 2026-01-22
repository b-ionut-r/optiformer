"""
Smoke Test Phase 4c: Categorical Parameter Evaluation

Tests the model's ability to handle categorical hyperparameters,
which are CRITICAL for real-world HPO tasks (optimizer type,
activation functions, etc.).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import optuna
from pathlib import Path
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass

from model import OptiFormer
from data.tokenizer import SequenceTokenizer, ParameterSpec, Trial
from samplers import OptiFormerSampler

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class CategoricalTestCase:
    """A test case for categorical parameter handling."""
    name: str
    description: str
    param_specs: List[ParameterSpec]
    n_trials: int = 30
    n_seeds: int = 3


def create_mixed_param_specs() -> List[ParameterSpec]:
    """Create parameter specs with mixed continuous and categorical."""
    return [
        ParameterSpec("learning_rate", 1e-5, 0.1, log_scale=True),
        ParameterSpec("hidden_size", 32, 512, is_integer=True),
        ParameterSpec("dropout", 0.0, 0.5),
        ParameterSpec(
            "optimizer", 0, 2,
            is_categorical=True,
            categories=["adam", "sgd", "adamw"]
        ),
        ParameterSpec(
            "activation", 0, 2,
            is_categorical=True,
            categories=["relu", "tanh", "gelu"]
        ),
    ]


def create_pure_categorical_specs() -> List[ParameterSpec]:
    """Create parameter specs with only categorical parameters."""
    return [
        ParameterSpec(
            "model_type", 0, 2,
            is_categorical=True,
            categories=["mlp", "cnn", "transformer"]
        ),
        ParameterSpec(
            "optimizer", 0, 3,
            is_categorical=True,
            categories=["adam", "sgd", "rmsprop", "adamw"]
        ),
        ParameterSpec(
            "scheduler", 0, 3,
            is_categorical=True,
            categories=["cosine", "step", "exponential", "none"]
        ),
        ParameterSpec(
            "normalization", 0, 2,
            is_categorical=True,
            categories=["batch", "layer", "none"]
        ),
    ]


def create_many_categories_specs() -> List[ParameterSpec]:
    """Create parameter specs with many categorical choices."""
    return [
        ParameterSpec(
            "architecture", 0, 7,
            is_categorical=True,
            categories=[
                "resnet18", "resnet34", "resnet50", "vgg16",
                "vgg19", "efficientnet_b0", "efficientnet_b1", "mobilenet_v2"
            ]
        ),
        ParameterSpec(
            "augmentation", 0, 4,
            is_categorical=True,
            categories=["none", "basic", "autoaugment", "randaugment", "trivialaugment"]
        ),
        ParameterSpec("learning_rate", 1e-5, 0.01, log_scale=True),
    ]


def create_simulated_categorical_objective(param_specs: List[ParameterSpec], seed: int):
    """Create a simulated objective that rewards certain categorical choices."""
    rng = np.random.RandomState(seed)

    # Randomly pick "optimal" categorical choices
    optimal_cats = {}
    for spec in param_specs:
        if spec.is_categorical:
            optimal_cats[spec.name] = rng.choice(spec.categories)

    # Random optimal for continuous
    optimal_cont = {}
    for spec in param_specs:
        if not spec.is_categorical:
            if spec.log_scale:
                optimal_cont[spec.name] = np.exp(rng.uniform(
                    np.log(spec.low), np.log(spec.high)
                ))
            else:
                optimal_cont[spec.name] = rng.uniform(spec.low, spec.high)

    def objective(params: Dict[str, Any]) -> float:
        score = 0.5  # Baseline

        # Categorical contribution
        for spec in param_specs:
            if spec.is_categorical:
                if spec.name in params:
                    if params[spec.name] == optimal_cats[spec.name]:
                        score -= 0.1  # Reward optimal choice
                    else:
                        score += rng.uniform(0.0, 0.05)  # Small penalty

        # Continuous contribution
        for spec in param_specs:
            if not spec.is_categorical and spec.name in params:
                val = params[spec.name]
                opt = optimal_cont[spec.name]

                if spec.log_scale:
                    val = np.log(val + 1e-10)
                    opt = np.log(opt + 1e-10)
                    norm = np.log(spec.high) - np.log(spec.low)
                else:
                    norm = spec.high - spec.low

                distance = abs(val - opt) / norm
                score += distance * 0.1

        # Add noise
        score += rng.normal(0, 0.02)
        return float(np.clip(score, 0.0, 1.0))

    return objective


def evaluate_categorical_handling(
    model: OptiFormer,
    test_case: CategoricalTestCase,
    device: str,
) -> Dict[str, Any]:
    """
    Evaluate model's categorical parameter handling for a test case.

    Returns metrics about:
    - Whether categorical suggestions are valid (in the choice list)
    - Distribution of categorical choices (not always same value)
    - Performance vs random baseline
    """
    results = {
        "name": test_case.name,
        "description": test_case.description,
        "valid_suggestions": 0,
        "total_suggestions": 0,
        "category_distributions": {},
        "optiformer_scores": [],
        "random_scores": [],
        "passed": False,
    }

    # Initialize category tracking
    for spec in test_case.param_specs:
        if spec.is_categorical:
            results["category_distributions"][spec.name] = {
                cat: 0 for cat in spec.categories
            }

    tokenizer = SequenceTokenizer(test_case.param_specs, num_bins=1000)

    sampler = OptiFormerSampler(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=0.8,
        min_trials_for_model=2,
        exploration_rate=0.05,  # Low exploration to test model behavior
    )

    for seed in range(test_case.n_seeds):
        objective = create_simulated_categorical_objective(
            test_case.param_specs, seed
        )

        # OptiFormer run
        def optuna_objective(trial: optuna.Trial) -> float:
            params = {}
            for spec in test_case.param_specs:
                if spec.is_categorical:
                    val = trial.suggest_categorical(spec.name, spec.categories)
                    params[spec.name] = val

                    # Track distribution
                    if val in results["category_distributions"][spec.name]:
                        results["category_distributions"][spec.name][val] += 1

                    # Check validity
                    results["total_suggestions"] += 1
                    if val in spec.categories:
                        results["valid_suggestions"] += 1
                elif spec.is_integer:
                    params[spec.name] = trial.suggest_int(
                        spec.name, int(spec.low), int(spec.high)
                    )
                else:
                    params[spec.name] = trial.suggest_float(
                        spec.name, spec.low, spec.high,
                        log=spec.log_scale
                    )

            return objective(params)

        # Run with OptiFormer
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            optuna_objective, n_trials=test_case.n_trials,
            show_progress_bar=False, catch=(Exception,)
        )
        if study.best_trial:
            results["optiformer_scores"].append(study.best_value)

        # Run with Random baseline
        random_sampler = optuna.samplers.RandomSampler(seed=seed)
        study_random = optuna.create_study(
            direction="minimize", sampler=random_sampler
        )
        study_random.optimize(
            optuna_objective, n_trials=test_case.n_trials,
            show_progress_bar=False, catch=(Exception,)
        )
        if study_random.best_trial:
            results["random_scores"].append(study_random.best_value)

    # Compute metrics
    if results["total_suggestions"] > 0:
        results["valid_rate"] = results["valid_suggestions"] / results["total_suggestions"]
    else:
        results["valid_rate"] = 0.0

    # Check for diversity (not always picking same category)
    diversity_ok = True
    for spec_name, dist in results["category_distributions"].items():
        total = sum(dist.values())
        if total > 0:
            max_ratio = max(dist.values()) / total
            # If one category is picked >90% of the time, that's suspicious
            if max_ratio > 0.9:
                diversity_ok = False
                results[f"{spec_name}_diversity_warning"] = True

    results["diversity_ok"] = diversity_ok

    # Win rate vs random
    if results["optiformer_scores"] and results["random_scores"]:
        wins = sum(
            1 for o, r in zip(results["optiformer_scores"], results["random_scores"])
            if o < r
        )
        results["vs_random_win_rate"] = wins / len(results["optiformer_scores"])
    else:
        results["vs_random_win_rate"] = 0.0

    # Determine pass/fail
    # Pass if: valid_rate >= 0.95 AND diversity_ok
    results["passed"] = results["valid_rate"] >= 0.95 and diversity_ok

    return results


def evaluate_categorical(
    model_path: Path,
    output_dir: Path,
    device: str = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run all categorical parameter tests.

    Args:
        model_path: Path to trained model checkpoint
        output_dir: Directory for results
        device: Device for inference

    Returns:
        (success, results_dict)
    """
    print("\nPhase 4c: Categorical Parameter Evaluation")
    print("-" * 40)

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "test_cases": {},
        "overall_valid_rate": 0.0,
        "all_passed": False,
        "errors": [],
    }

    try:
        # Load model
        print(f"  Loading model from {model_path}...")
        if not model_path.exists():
            results["errors"].append(f"Model not found: {model_path}")
            return False, results

        model = OptiFormer.load(str(model_path), device=device)
        model.eval()

        # Define test cases
        test_cases = [
            CategoricalTestCase(
                name="mixed_nn_config",
                description="Neural network with mixed hyperparameter types",
                param_specs=create_mixed_param_specs(),
                n_trials=25,
                n_seeds=3,
            ),
            CategoricalTestCase(
                name="pure_categorical",
                description="All categorical parameters",
                param_specs=create_pure_categorical_specs(),
                n_trials=20,
                n_seeds=3,
            ),
            CategoricalTestCase(
                name="many_categories",
                description="Parameters with many categorical choices",
                param_specs=create_many_categories_specs(),
                n_trials=25,
                n_seeds=3,
            ),
        ]

        # Run each test case
        total_valid = 0
        total_suggestions = 0
        all_passed = True

        for test_case in test_cases:
            print(f"\n  Testing: {test_case.name}")
            print(f"    {test_case.description}")

            case_results = evaluate_categorical_handling(
                model, test_case, device
            )
            results["test_cases"][test_case.name] = case_results

            total_valid += case_results["valid_suggestions"]
            total_suggestions += case_results["total_suggestions"]

            print(f"    Valid rate: {case_results['valid_rate']:.1%}")
            print(f"    Diversity OK: {case_results['diversity_ok']}")
            print(f"    vs Random: {case_results['vs_random_win_rate']:.1%}")
            print(f"    Passed: {case_results['passed']}")

            if not case_results["passed"]:
                all_passed = False

        # Overall metrics
        if total_suggestions > 0:
            results["overall_valid_rate"] = total_valid / total_suggestions

        results["all_passed"] = all_passed

        print(f"\n  Overall Results:")
        print(f"    Valid rate: {results['overall_valid_rate']:.1%}")
        print(f"    All passed: {all_passed}")

        return all_passed, results

    except Exception as e:
        import traceback
        results["errors"].append(str(e))
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False, results


if __name__ == "__main__":
    model_path = Path("./outputs/smoke_test/model/best.pt")
    output_dir = Path("./outputs/smoke_test/eval_categorical")

    success, results = evaluate_categorical(
        model_path=model_path,
        output_dir=output_dir,
    )

    print(f"\nCategorical evaluation: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Valid rate: {results['overall_valid_rate']:.1%}")
    print(f"  All passed: {results['all_passed']}")

    sys.exit(0 if success else 1)
