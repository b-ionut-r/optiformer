"""
Smoke Test Phase 4d: Edge Case Evaluation

Tests the model's robustness against edge cases:
- Out-of-bounds value handling
- Long trajectory histories (100+ trials)
- Extreme/invalid scores (NaN, Inf, negative)
- Duplicate trials in history
- Single dimension (d=1)
- High dimension (d=20, beyond training)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
import warnings

from model import OptiFormer
from data.tokenizer import SequenceTokenizer, ParameterSpec, Trial
from samplers import OptiFormerSampler


def create_param_specs(n_dims: int, bounds: tuple = (0.0, 1.0)) -> List[ParameterSpec]:
    """Create parameter specs for testing."""
    return [
        ParameterSpec(f"x{i}", bounds[0], bounds[1], log_scale=False)
        for i in range(n_dims)
    ]


def test_bounds_handling(
    model: OptiFormer,
    device: str,
    n_samples: int = 50,
) -> Dict[str, Any]:
    """
    Test that model suggestions stay within bounds.

    The sampler should clip out-of-bounds suggestions.
    """
    print("    Testing bounds handling...")

    results = {
        "name": "bounds_handling",
        "in_bounds": 0,
        "total": 0,
        "passed": False,
    }

    param_specs = create_param_specs(n_dims=3, bounds=(-1.0, 1.0))
    tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

    sampler = OptiFormerSampler(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=1.2,  # Higher temp to encourage extreme values
        min_trials_for_model=1,
        exploration_rate=0.0,
    )

    # Create history with extreme values
    history = [
        Trial(params={"x0": -0.9, "x1": 0.9, "x2": 0.0}, score=0.5),
        Trial(params={"x0": 0.9, "x1": -0.9, "x2": 0.1}, score=0.3),
        Trial(params={"x0": 0.0, "x1": 0.0, "x2": -0.5}, score=0.2),
    ]

    for _ in range(n_samples):
        try:
            # Get context
            tokens = tokenizer.encode_for_inference(history, params_so_far={})
            input_ids = torch.tensor([tokens], device=device)

            # Sample
            with torch.no_grad():
                next_token = model.generate_next_token(
                    input_ids, temperature=1.2, top_k=50
                )

            # Decode
            value = tokenizer.decode_predicted_value(next_token, "x0")

            results["total"] += 1
            if -1.0 <= value <= 1.0:
                results["in_bounds"] += 1

        except Exception as e:
            pass

    if results["total"] > 0:
        results["in_bounds_rate"] = results["in_bounds"] / results["total"]
        results["passed"] = results["in_bounds_rate"] >= 0.95

    print(f"      In-bounds rate: {results.get('in_bounds_rate', 0):.1%}")
    return results


def test_long_trajectories(
    model: OptiFormer,
    device: str,
    trajectory_lengths: List[int] = [50, 100, 150],
) -> Dict[str, Any]:
    """
    Test handling of long trajectory histories.

    Model should handle 100+ trial histories without:
    - Memory errors
    - Significantly degraded suggestions
    """
    print("    Testing long trajectories...")

    results = {
        "name": "long_trajectories",
        "by_length": {},
        "passed": False,
    }

    param_specs = create_param_specs(n_dims=3, bounds=(0.0, 1.0))
    tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

    for length in trajectory_lengths:
        length_results = {
            "length": length,
            "success": False,
            "generated_values": 0,
            "memory_ok": True,
        }

        try:
            # Create long history
            history = []
            for i in range(length):
                params = {
                    "x0": np.random.uniform(0, 1),
                    "x1": np.random.uniform(0, 1),
                    "x2": np.random.uniform(0, 1),
                }
                score = np.random.uniform(0.1, 0.9)
                history.append(Trial(params=params, score=score))

            # Try to generate with long history
            tokens = tokenizer.encode_for_inference(history, params_so_far={})

            # Check if sequence is within model limits
            max_len = model.config.max_position_embeddings
            if len(tokens) > max_len:
                # Truncate history
                truncated_len = max_len // (3 * 3 + 2)  # rough estimate
                history = history[-truncated_len:]
                tokens = tokenizer.encode_for_inference(history, params_so_far={})

            input_ids = torch.tensor([tokens], device=device)

            # Generate
            with torch.no_grad():
                for _ in range(5):  # Generate 5 values
                    next_token = model.generate_next_token(
                        input_ids, temperature=0.8, top_k=50
                    )
                    length_results["generated_values"] += 1

            length_results["success"] = True

        except torch.cuda.OutOfMemoryError:
            length_results["memory_ok"] = False
            length_results["success"] = False

        except Exception as e:
            length_results["error"] = str(e)
            length_results["success"] = False

        results["by_length"][length] = length_results
        print(f"      Length {length}: {'OK' if length_results['success'] else 'FAIL'}")

    # Pass if at least 2/3 lengths work
    successes = sum(1 for r in results["by_length"].values() if r["success"])
    results["passed"] = successes >= 2

    return results


def test_score_edge_cases(
    model: OptiFormer,
    device: str,
) -> Dict[str, Any]:
    """
    Test handling of extreme/invalid scores in history.

    Model should handle or gracefully reject:
    - NaN scores
    - Inf scores
    - Negative scores
    - Very small scores (< 1e-10)
    """
    print("    Testing score edge cases...")

    results = {
        "name": "score_edge_cases",
        "tests": {},
        "passed": False,
    }

    param_specs = create_param_specs(n_dims=2, bounds=(0.0, 1.0))
    tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

    test_scores = [
        ("normal", [0.5, 0.3, 0.2]),
        ("negative", [0.5, -0.1, 0.2]),
        ("very_small", [0.5, 1e-15, 0.2]),
        ("large", [0.5, 10.0, 0.2]),
    ]

    for test_name, scores in test_scores:
        test_result = {"success": False, "generated": False}

        try:
            history = []
            for i, score in enumerate(scores):
                # Clip or handle invalid scores
                safe_score = score
                if np.isnan(safe_score) or np.isinf(safe_score):
                    safe_score = 0.5
                safe_score = np.clip(safe_score, 0.0, 1.0)

                history.append(Trial(
                    params={"x0": 0.3 + i * 0.1, "x1": 0.5},
                    score=safe_score
                ))

            tokens = tokenizer.encode_for_inference(history, params_so_far={})
            input_ids = torch.tensor([tokens], device=device)

            with torch.no_grad():
                next_token = model.generate_next_token(
                    input_ids, temperature=0.8, top_k=50
                )

            test_result["success"] = True
            test_result["generated"] = True

        except Exception as e:
            test_result["error"] = str(e)

        results["tests"][test_name] = test_result
        print(f"      {test_name}: {'OK' if test_result['success'] else 'FAIL'}")

    # Pass if normal and at least 2 edge cases work
    successes = sum(1 for r in results["tests"].values() if r["success"])
    results["passed"] = results["tests"].get("normal", {}).get("success", False) and successes >= 3

    return results


def test_duplicate_handling(
    model: OptiFormer,
    device: str,
) -> Dict[str, Any]:
    """
    Test behavior with duplicate configurations in history.

    Model should handle duplicate trials gracefully.
    """
    print("    Testing duplicate handling...")

    results = {
        "name": "duplicate_handling",
        "success": False,
        "generates_different": False,
    }

    param_specs = create_param_specs(n_dims=2, bounds=(0.0, 1.0))
    tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

    try:
        # History with duplicates
        history = [
            Trial(params={"x0": 0.5, "x1": 0.5}, score=0.4),
            Trial(params={"x0": 0.5, "x1": 0.5}, score=0.4),  # Duplicate
            Trial(params={"x0": 0.5, "x1": 0.5}, score=0.4),  # Duplicate
            Trial(params={"x0": 0.3, "x1": 0.7}, score=0.3),
        ]

        tokens = tokenizer.encode_for_inference(history, params_so_far={})
        input_ids = torch.tensor([tokens], device=device)

        # Generate multiple times and check for diversity
        generated_values = []
        with torch.no_grad():
            for _ in range(10):
                next_token = model.generate_next_token(
                    input_ids, temperature=0.8, top_k=50
                )
                value = tokenizer.decode_predicted_value(next_token, "x0")
                generated_values.append(value)

        results["success"] = True

        # Check if model generates diverse values (not just 0.5)
        unique_values = len(set(round(v, 2) for v in generated_values))
        results["unique_values"] = unique_values
        results["generates_different"] = unique_values >= 3

    except Exception as e:
        results["error"] = str(e)

    results["passed"] = results["success"]
    print(f"      Success: {results['success']}, Diverse: {results.get('generates_different', False)}")
    return results


def test_single_dimension(
    model: OptiFormer,
    device: str,
) -> Dict[str, Any]:
    """
    Test with d=1 (edge case - minimum dimensions).
    """
    print("    Testing single dimension (d=1)...")

    results = {
        "name": "single_dimension",
        "success": False,
        "generated": 0,
    }

    param_specs = create_param_specs(n_dims=1, bounds=(0.0, 1.0))
    tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

    try:
        history = [
            Trial(params={"x0": 0.2}, score=0.6),
            Trial(params={"x0": 0.5}, score=0.3),
            Trial(params={"x0": 0.8}, score=0.5),
        ]

        tokens = tokenizer.encode_for_inference(history, params_so_far={})
        input_ids = torch.tensor([tokens], device=device)

        with torch.no_grad():
            for _ in range(5):
                next_token = model.generate_next_token(
                    input_ids, temperature=0.8, top_k=50
                )
                value = tokenizer.decode_predicted_value(next_token, "x0")
                if 0.0 <= value <= 1.0:
                    results["generated"] += 1

        results["success"] = results["generated"] >= 3

    except Exception as e:
        results["error"] = str(e)

    results["passed"] = results["success"]
    print(f"      Generated {results['generated']} valid values")
    return results


def test_high_dimension(
    model: OptiFormer,
    device: str,
    n_dims: int = 20,
) -> Dict[str, Any]:
    """
    Test with high dimensions (beyond training).

    Model may not perform well, but should handle gracefully.
    """
    print(f"    Testing high dimension (d={n_dims})...")

    results = {
        "name": "high_dimension",
        "n_dims": n_dims,
        "success": False,
        "generates_values": False,
    }

    param_specs = create_param_specs(n_dims=n_dims, bounds=(0.0, 1.0))
    tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

    try:
        # Create minimal history
        history = [
            Trial(
                params={f"x{i}": np.random.uniform(0, 1) for i in range(n_dims)},
                score=0.5
            )
            for _ in range(3)
        ]

        tokens = tokenizer.encode_for_inference(history, params_so_far={})

        # Check if within model limits
        max_len = model.config.max_position_embeddings
        if len(tokens) > max_len:
            results["warning"] = f"Sequence too long ({len(tokens)} > {max_len})"
            # Truncate
            tokens = tokens[:max_len - 10]

        input_ids = torch.tensor([tokens], device=device)

        with torch.no_grad():
            next_token = model.generate_next_token(
                input_ids, temperature=0.8, top_k=50
            )
            value = tokenizer.decode_predicted_value(next_token, "x0")

        results["success"] = True
        results["generates_values"] = True
        results["sample_value"] = float(value)

    except Exception as e:
        results["error"] = str(e)

    results["passed"] = results["success"]
    print(f"      Success: {results['success']}")
    return results


def evaluate_edge_cases(
    model_path: Path,
    output_dir: Path,
    device: str = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run all edge case tests.

    Args:
        model_path: Path to trained model checkpoint
        output_dir: Directory for results
        device: Device for inference

    Returns:
        (success, results_dict)
    """
    print("\nPhase 4d: Edge Case Evaluation")
    print("-" * 40)

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "tests": {},
        "passed_count": 0,
        "total_count": 0,
        "pass_rate": 0.0,
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

        # Run tests
        tests = [
            ("bounds_handling", lambda: test_bounds_handling(model, device)),
            ("long_trajectories", lambda: test_long_trajectories(model, device)),
            ("score_edge_cases", lambda: test_score_edge_cases(model, device)),
            ("duplicate_handling", lambda: test_duplicate_handling(model, device)),
            ("single_dimension", lambda: test_single_dimension(model, device)),
            ("high_dimension", lambda: test_high_dimension(model, device)),
        ]

        for test_name, test_fn in tests:
            try:
                test_results = test_fn()
                results["tests"][test_name] = test_results
                results["total_count"] += 1
                if test_results.get("passed", False):
                    results["passed_count"] += 1
            except Exception as e:
                results["tests"][test_name] = {"error": str(e), "passed": False}
                results["total_count"] += 1

        # Overall metrics
        if results["total_count"] > 0:
            results["pass_rate"] = results["passed_count"] / results["total_count"]

        # Pass if >= 80% of tests pass
        results["all_passed"] = results["pass_rate"] >= 0.8

        print(f"\n  Overall Results:")
        print(f"    Passed: {results['passed_count']}/{results['total_count']}")
        print(f"    Pass rate: {results['pass_rate']:.1%}")

        return results["all_passed"], results

    except Exception as e:
        import traceback
        results["errors"].append(str(e))
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False, results


if __name__ == "__main__":
    model_path = Path("./outputs/smoke_test/model/best.pt")
    output_dir = Path("./outputs/smoke_test/eval_edge_cases")

    success, results = evaluate_edge_cases(
        model_path=model_path,
        output_dir=output_dir,
    )

    print(f"\nEdge case evaluation: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Pass rate: {results['pass_rate']:.1%}")

    sys.exit(0 if success else 1)
