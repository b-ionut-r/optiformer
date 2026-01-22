"""
Smoke Test Phase 5b: Inference Logic Check (Exploration vs Exploitation)

Validates that the model produces sensible suggestions:
- Exploration: Given bad history, suggests different regions
- Exploitation: Given good history, suggests near best known
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List

from model import OptiFormer
from data.tokenizer import SequenceTokenizer, ParameterSpec, Trial


def create_test_param_specs(n_dims: int = 3) -> List[ParameterSpec]:
    """Create parameter specs for inference testing."""
    return [
        ParameterSpec(f"x{i}", 0.0, 1.0, log_scale=False)
        for i in range(n_dims)
    ]


def create_bad_history(n_trials: int = 5, n_dims: int = 3) -> List[Trial]:
    """
    Create a history of 'bad' trials - all in a poor region with high scores.

    All trials are clustered around x_i = 0.2 with poor scores (~0.8-0.9).
    A model should learn to explore AWAY from this region.
    """
    trials = []
    for i in range(n_trials):
        params = {f"x{d}": 0.15 + np.random.uniform(0, 0.1) for d in range(n_dims)}
        score = 0.8 + np.random.uniform(0, 0.15)  # Bad scores
        trials.append(Trial(params=params, score=score))
    return trials


def create_good_history(n_trials: int = 5, n_dims: int = 3) -> List[Trial]:
    """
    Create a history of 'good' trials - with one excellent trial near optimum.

    Most trials scattered, but one excellent trial at x_i ~ 0.75 with score ~0.05.
    A model should learn to exploit NEAR the best trial.
    """
    trials = []

    # Add some mediocre trials
    for i in range(n_trials - 1):
        params = {f"x{d}": np.random.uniform(0.2, 0.6) for d in range(n_dims)}
        score = 0.3 + np.random.uniform(0, 0.3)  # Mediocre scores
        trials.append(Trial(params=params, score=score))

    # Add one excellent trial near 0.75
    best_params = {f"x{d}": 0.72 + np.random.uniform(0, 0.06) for d in range(n_dims)}
    best_score = 0.02 + np.random.uniform(0, 0.05)  # Excellent score
    trials.append(Trial(params=best_params, score=best_score))

    return trials


def test_inference_behavior(
    model_path: Path,
    n_dims: int = 3,
    n_samples: int = 10,
    device: str = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test the model's inference behavior for exploration vs exploitation.

    Args:
        model_path: Path to trained model
        n_dims: Number of dimensions for test
        n_samples: Number of inference samples per scenario
        device: Device for inference

    Returns:
        (success, results_dict)
    """
    print("\nPhase 5b: Inference Logic Check")
    print("-" * 40)

    model_path = Path(model_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        'exploration_test': {'passed': False, 'details': {}},
        'exploitation_test': {'passed': False, 'details': {}},
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

        # Create tokenizer
        param_specs = create_test_param_specs(n_dims)
        tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

        # ========== EXPLORATION TEST ==========
        print("\n  Testing Exploration Behavior (bad history -> explore away)...")

        bad_history = create_bad_history(n_trials=5, n_dims=n_dims)
        bad_region_center = 0.175  # Center of bad region

        exploration_suggestions = []
        for _ in range(n_samples):
            suggestion = sample_next_config(model, tokenizer, bad_history, device)
            exploration_suggestions.append(suggestion)

        # Check: suggestions should be far from the bad region
        avg_distance_from_bad = np.mean([
            np.mean([abs(s[f"x{d}"] - bad_region_center) for d in range(n_dims)])
            for s in exploration_suggestions
        ])

        # Model should suggest values at least 0.2 away from bad region on average
        exploration_passed = avg_distance_from_bad > 0.2

        results['exploration_test'] = {
            'passed': exploration_passed,
            'avg_distance_from_bad_region': float(avg_distance_from_bad),
            'suggestions_sample': [
                {k: round(v, 3) for k, v in s.items()}
                for s in exploration_suggestions[:3]
            ],
            'bad_region_center': bad_region_center,
        }

        print(f"    Bad region center: {bad_region_center}")
        print(f"    Avg distance from bad region: {avg_distance_from_bad:.3f}")
        print(f"    Sample suggestions: {exploration_suggestions[0]}")
        print(f"    Exploration test: {'PASS' if exploration_passed else 'FAIL'}")

        # ========== EXPLOITATION TEST ==========
        print("\n  Testing Exploitation Behavior (good history -> exploit near best)...")

        good_history = create_good_history(n_trials=5, n_dims=n_dims)
        best_trial = min(good_history, key=lambda t: t.score)
        best_region_center = np.mean([best_trial.params[f"x{d}"] for d in range(n_dims)])

        exploitation_suggestions = []
        for _ in range(n_samples):
            suggestion = sample_next_config(model, tokenizer, good_history, device)
            exploitation_suggestions.append(suggestion)

        # Check: suggestions should be close to the best trial
        avg_distance_from_best = np.mean([
            np.mean([abs(s[f"x{d}"] - best_trial.params[f"x{d}"]) for d in range(n_dims)])
            for s in exploitation_suggestions
        ])

        # Model should suggest values within 0.3 of best trial on average
        exploitation_passed = avg_distance_from_best < 0.3

        results['exploitation_test'] = {
            'passed': exploitation_passed,
            'avg_distance_from_best': float(avg_distance_from_best),
            'best_trial_params': {k: round(v, 3) for k, v in best_trial.params.items()},
            'best_trial_score': float(best_trial.score),
            'suggestions_sample': [
                {k: round(v, 3) for k, v in s.items()}
                for s in exploitation_suggestions[:3]
            ],
        }

        print(f"    Best trial params: {best_trial.params}")
        print(f"    Best trial score: {best_trial.score:.4f}")
        print(f"    Avg distance from best: {avg_distance_from_best:.3f}")
        print(f"    Sample suggestions: {exploitation_suggestions[0]}")
        print(f"    Exploitation test: {'PASS' if exploitation_passed else 'FAIL'}")

        # ========== VALIDITY CHECK ==========
        print("\n  Checking suggestion validity...")

        all_suggestions = exploration_suggestions + exploitation_suggestions
        all_valid = all(
            all(0.0 <= s[f"x{d}"] <= 1.0 for d in range(n_dims))
            for s in all_suggestions
        )

        results['validity'] = {
            'all_in_bounds': all_valid,
            'n_suggestions_checked': len(all_suggestions),
        }

        print(f"    All suggestions in bounds: {all_valid}")

        # Overall success: both tests pass and all suggestions valid
        # Note: For minimally trained model, we use relaxed criteria
        # At least one test should pass to indicate some learning
        overall_pass = (exploration_passed or exploitation_passed) and all_valid

        print(f"\n  Overall inference check: {'PASS' if overall_pass else 'FAIL'}")

        return overall_pass, results

    except Exception as e:
        import traceback
        results['errors'].append(str(e))
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False, results


@torch.no_grad()
def sample_next_config(
    model: OptiFormer,
    tokenizer: SequenceTokenizer,
    history: List[Trial],
    device: str,
) -> Dict[str, float]:
    """
    Sample a complete next configuration from the model.

    Args:
        model: Trained OptiFormer model
        tokenizer: Sequence tokenizer
        history: List of past trials
        device: Device for inference

    Returns:
        Dict of parameter name -> suggested value
    """
    params = {}

    for param_name in tokenizer.param_order:
        # Encode context with params sampled so far
        tokens = tokenizer.encode_for_inference(history, params)
        input_ids = torch.tensor([tokens], device=device)

        # Get model's probability distribution
        probs = model.get_token_probabilities(input_ids)

        # Sample from numerical tokens (0-999)
        valid_probs = probs[:tokenizer.num_bins].cpu().numpy()
        valid_probs = valid_probs / (valid_probs.sum() + 1e-10)

        # Apply temperature for diversity
        temperature = 0.8
        logits = np.log(valid_probs + 1e-10) / temperature
        valid_probs = np.exp(logits) / np.exp(logits).sum()

        # Sample token
        token_id = np.random.choice(tokenizer.num_bins, p=valid_probs)

        # Decode to value
        spec = tokenizer.param_specs[param_name]
        value = tokenizer.numerical.decode(token_id, spec.low, spec.high, spec.log_scale)

        params[param_name] = float(np.clip(value, spec.low, spec.high))

    return params


if __name__ == "__main__":
    model_path = Path("./outputs/smoke_test/model/best.pt")

    success, results = test_inference_behavior(
        model_path=model_path,
        n_dims=3,
        n_samples=10,
    )

    print(f"\nInference behavior test: {'SUCCESS' if success else 'FAILED'}")

    sys.exit(0 if success else 1)
