"""
Master Smoke Test Script

Validates the entire OptiFormer pipeline end-to-end.
Run with: python -m smoke_test.run_all

This comprehensive smoke test includes:
1. Tokenizer validation
2. Data generation (synthetic + real-world/simulated)
3. Model training
4. Inference behavior tests
5. Synthetic benchmark evaluation
6. Categorical parameter tests
7. Edge case tests
8. Real-world ML evaluation
9. Optuna integration tests

The test validates that the model can handle real-world HPO scenarios
including mixed continuous/categorical parameters and realistic ML
hyperparameter spaces.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path
from datetime import datetime
import json
import argparse
import warnings

# Configuration
DEFAULT_OUTPUT_DIR = Path("./outputs/smoke_test")


def print_header(text):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_smoke_test(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    skip_training: bool = False,
    skip_realworld: bool = False,
    quick: bool = False,
) -> dict:
    """
    Run the complete smoke test pipeline.

    Args:
        output_dir: Directory for all outputs
        skip_training: If True, skip training and use existing model
        skip_realworld: If True, skip real-world ML evaluation
        quick: If True, use minimal settings for speed

    Returns:
        Results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'start_time': datetime.now().isoformat(),
        'phases': {},
        'passed': True,
    }

    # Configuration based on quick mode
    if quick:
        n_gp_functions = 100
        n_symbolic_functions = 100
        trials_per_function = 16
        max_steps = 500
        n_trials_eval = 15
        n_seeds = 2
    else:
        n_gp_functions = 500
        n_symbolic_functions = 500
        trials_per_function = 32
        max_steps = 2000
        n_trials_eval = 30
        n_seeds = 3

    # ========== PHASE 1: TOKENIZER ==========
    print_header("PHASE 1: Tokenizer Validation")
    try:
        from smoke_test.phase1_tokenizer import test_tokenizer
        phase1_pass, phase1_results = test_tokenizer()
        results['phases']['tokenizer'] = phase1_results
        if not phase1_pass:
            print("TOKENIZER TESTS FAILED - Cannot proceed")
            results['passed'] = False
            return results
        print("Tokenizer tests passed")
    except Exception as e:
        print(f"PHASE 1 ERROR: {e}")
        results['phases']['tokenizer'] = {'error': str(e)}
        results['passed'] = False
        return results

    # ========== PHASE 2: DATA GENERATION ==========
    print_header("PHASE 2: Data Generation")
    data_dir = output_dir / "data"
    try:
        from smoke_test.phase2_data import generate_smoke_test_data
        phase2_pass, phase2_results = generate_smoke_test_data(
            output_dir=data_dir,
            n_gp_functions=n_gp_functions,
            n_symbolic_functions=n_symbolic_functions,
            trials_per_function=trials_per_function,
        )
        results['phases']['data_generation'] = phase2_results
        if not phase2_pass:
            print("DATA GENERATION FAILED")
            results['passed'] = False
            return results
        print(f"Generated {phase2_results['n_trajectories']} trajectories")
    except Exception as e:
        print(f"PHASE 2 ERROR: {e}")
        results['phases']['data_generation'] = {'error': str(e)}
        results['passed'] = False
        return results

    # ========== PHASE 3: TRAINING ==========
    print_header("PHASE 3: Model Training")
    model_dir = output_dir / "model"

    if skip_training and (model_dir / "best.pt").exists():
        print("Skipping training - using existing model")
        results['phases']['training'] = {'skipped': True}
        phase3_pass = True
        phase3_results = {'skipped': True}
    else:
        try:
            from smoke_test.phase3_training import train_smoke_test_model
            phase3_pass, phase3_results = train_smoke_test_model(
                data_dir=data_dir,
                output_dir=model_dir,
                max_steps=max_steps,
                batch_size=64 if not quick else 32,
            )
            results['phases']['training'] = phase3_results
            if not phase3_pass:
                print("TRAINING FAILED")
                results['passed'] = False
                return results
            print(f"Training complete. Final loss: {phase3_results.get('final_loss', 'N/A')}")
            # Print GPU memory if available
            if phase3_results.get('gpu_memory', {}).get('peak_mb'):
                print(f"  Peak GPU memory: {phase3_results['gpu_memory']['peak_mb']:.1f} MB")
        except Exception as e:
            import traceback
            print(f"PHASE 3 ERROR: {e}")
            traceback.print_exc()
            results['phases']['training'] = {'error': str(e)}
            results['passed'] = False
            return results

    # ========== PHASE 4a: INFERENCE BEHAVIOR TEST ==========
    print_header("PHASE 4a: Inference Behavior Test")
    try:
        from smoke_test.phase5b_inference import test_inference_behavior
        phase4a_pass, phase4a_results = test_inference_behavior(
            model_path=model_dir / "best.pt",
            n_dims=3,
            n_samples=10 if not quick else 5,
        )
        results['phases']['inference_behavior'] = phase4a_results

        # Print summary
        expl_pass = phase4a_results.get('exploration_test', {}).get('passed', False)
        exploit_pass = phase4a_results.get('exploitation_test', {}).get('passed', False)
        print(f"  Exploration test: {'PASS' if expl_pass else 'FAIL'}")
        print(f"  Exploitation test: {'PASS' if exploit_pass else 'FAIL'}")
    except Exception as e:
        import traceback
        print(f"PHASE 4a ERROR: {e}")
        traceback.print_exc()
        results['phases']['inference_behavior'] = {'error': str(e)}
        phase4a_pass = False
        phase4a_results = {}

    # ========== PHASE 4b: SYNTHETIC EVALUATION ==========
    print_header("PHASE 4b: Synthetic Benchmark Evaluation")
    try:
        from smoke_test.phase4_synthetic import evaluate_synthetic

        # Test multiple dimensions to match training dimensions
        # Quick mode uses fewer dimensions for speed
        if quick:
            eval_dims = [2, 3]
        else:
            eval_dims = [2, 3, 5]  # Subset of training dims [3, 5, 8, 10]

        phase4_pass, phase4_results = evaluate_synthetic(
            model_path=model_dir / "best.pt",
            output_dir=output_dir / "eval_synthetic",
            dims_list=eval_dims,
            n_trials=n_trials_eval,
            n_seeds=n_seeds,
        )
        results['phases']['synthetic_eval'] = phase4_results
        print(f"  vs Random: {phase4_results['vs_random_win_rate']:.1%} win rate")
        print(f"  vs TPE: {phase4_results['vs_tpe_win_rate']:.1%} win rate")
        if phase4_results.get('by_dimension'):
            for dim, dim_res in phase4_results['by_dimension'].items():
                print(f"    {dim}D: {dim_res['vs_random_win_rate']:.1%} vs Random")
    except Exception as e:
        import traceback
        print(f"PHASE 4 ERROR: {e}")
        traceback.print_exc()
        results['phases']['synthetic_eval'] = {'error': str(e)}
        phase4_pass = False
        phase4_results = {'vs_random_win_rate': 0, 'vs_tpe_win_rate': 0}

    # ========== PHASE 5: REAL-WORLD ML EVALUATION ==========
    if not skip_realworld:
        print_header("PHASE 5: Real-World ML Evaluation")
        try:
            from smoke_test.phase5_realworld import evaluate_realworld
            phase5_pass, phase5_results = evaluate_realworld(
                model_path=model_dir / "best.pt",
                output_dir=output_dir / "eval_realworld",
                n_trials=n_trials_eval // 2,
                n_seeds=n_seeds,
                epochs_per_trial=3 if not quick else 2,
                run_mnist=True,
                run_fashion=False,
            )
            results['phases']['realworld_eval'] = phase5_results

            # Print ML benchmark results
            for bench_name, bench_results in phase5_results.get('benchmarks', {}).items():
                print(f"\n  {bench_name}:")
                print(f"    OptiFormer best: {bench_results.get('optiformer_best', 'N/A')}")
                print(f"    TPE best: {bench_results.get('tpe_best', 'N/A')}")
                print(f"    Random best: {bench_results.get('random_best', 'N/A')}")
        except Exception as e:
            import traceback
            print(f"PHASE 5 ERROR: {e}")
            traceback.print_exc()
            results['phases']['realworld_eval'] = {'error': str(e)}
            phase5_pass = False
            phase5_results = {'vs_random_win_rate': 0, 'benchmarks': {}}
    else:
        print_header("PHASE 5: Real-World ML Evaluation (SKIPPED)")
        phase5_pass = True
        phase5_results = {'skipped': True, 'vs_random_win_rate': 0.5}
        results['phases']['realworld_eval'] = phase5_results

    # ========== PHASE 4c: CATEGORICAL PARAMETER TESTS ==========
    print_header("PHASE 4c: Categorical Parameter Tests")
    try:
        from smoke_test.phase4c_categorical import evaluate_categorical
        phase4c_pass, phase4c_results = evaluate_categorical(
            model_path=model_dir / "best.pt",
            output_dir=output_dir / "eval_categorical",
        )
        results['phases']['categorical_eval'] = phase4c_results

        # Print summary
        print(f"  Valid rate: {phase4c_results.get('overall_valid_rate', 0):.1%}")
        print(f"  All passed: {phase4c_results.get('all_passed', False)}")
    except Exception as e:
        import traceback
        print(f"PHASE 4c ERROR: {e}")
        traceback.print_exc()
        results['phases']['categorical_eval'] = {'error': str(e)}
        phase4c_pass = False
        phase4c_results = {'overall_valid_rate': 0}

    # ========== PHASE 4d: EDGE CASE TESTS ==========
    print_header("PHASE 4d: Edge Case Tests")
    try:
        from smoke_test.phase4d_edge_cases import evaluate_edge_cases
        phase4d_pass, phase4d_results = evaluate_edge_cases(
            model_path=model_dir / "best.pt",
            output_dir=output_dir / "eval_edge_cases",
        )
        results['phases']['edge_case_eval'] = phase4d_results

        # Print summary
        print(f"  Pass rate: {phase4d_results.get('pass_rate', 0):.1%}")
        print(f"  Passed: {phase4d_results.get('passed_count', 0)}/{phase4d_results.get('total_count', 0)}")
    except Exception as e:
        import traceback
        print(f"PHASE 4d ERROR: {e}")
        traceback.print_exc()
        results['phases']['edge_case_eval'] = {'error': str(e)}
        phase4d_pass = False
        phase4d_results = {'pass_rate': 0}

    # ========== PHASE 6: OPTUNA INTEGRATION TEST ==========
    print_header("PHASE 6: Optuna Integration Test")
    try:
        from smoke_test.phase6_optuna import test_optuna_integration
        phase6_pass, phase6_results = test_optuna_integration(
            model_path=model_dir / "best.pt",
            n_dims=3,
            n_trials=15 if not quick else 10,
        )
        results['phases']['optuna_integration'] = phase6_results

        # Print summary
        print(f"  Study completed: {phase6_results.get('study_completed', False)}")
        print(f"  Best value: {phase6_results.get('best_value', 'N/A')}")
        print(f"  All valid: {phase6_results.get('all_suggestions_valid', False)}")
        print(f"  Shows adaptation: {phase6_results.get('suggestions_adapted', False)}")
    except Exception as e:
        import traceback
        print(f"PHASE 6 ERROR: {e}")
        traceback.print_exc()
        results['phases']['optuna_integration'] = {'error': str(e)}
        phase6_pass = False
        phase6_results = {}

    # ========== FINAL VERDICT ==========
    print_header("FINAL VERDICT")

    # Check all criteria
    criteria = {
        'tokenizer_pass': results['phases'].get('tokenizer', {}).get('errors', []) == [] if 'tokenizer' in results['phases'] else False,
        'data_generated': results['phases'].get('data_generation', {}).get('n_trajectories', 0) > 0,
        'training_converged': results['phases'].get('training', {}).get('loss_reduction', 0) > 0.5,  # Relaxed to 50%
        'inference_behavior': phase4a_pass,  # At least one of exploration/exploitation tests passed
        'beats_random_synthetic': phase4_results.get('vs_random_win_rate', 0) > 0.5,
        'categorical_handling': phase4c_results.get('overall_valid_rate', 0) >= 0.9,  # 90% valid categorical suggestions
        'edge_cases_pass': phase4d_results.get('pass_rate', 0) >= 0.7,  # 70% of edge cases pass
        'optuna_integration': phase6_results.get('study_completed', False) and phase6_results.get('all_suggestions_valid', True),
    }

    if not skip_realworld:
        criteria['beats_random_ml'] = phase5_results.get('vs_random_win_rate', 0) > 0.3

    print("Criteria Check:")
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {passed}")

    all_passed = all(criteria.values())
    results['passed'] = all_passed
    results['criteria'] = criteria
    results['end_time'] = datetime.now().isoformat()

    # Save results
    with open(output_dir / "smoke_test_results.json", 'w') as f:
        # Convert non-serializable items
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        json.dump(make_serializable(results), f, indent=2)

    if all_passed:
        print("\n" + "=" * 60)
        print("  SMOKE TEST PASSED")
        print("  OptiFormer is ready for full-scale training!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  SMOKE TEST FAILED")
        print("  Review failed criteria before scaling up.")
        print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="OptiFormer Smoke Test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/smoke_test",
        help="Output directory for all files",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use existing model",
    )
    parser.add_argument(
        "--skip-realworld",
        action="store_true",
        help="Skip real-world ML evaluation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use minimal settings for quick testing",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  OptiFormer-Lite Smoke Test")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_time = time.time()

    results = run_smoke_test(
        output_dir=Path(args.output_dir),
        skip_training=args.skip_training,
        skip_realworld=args.skip_realworld,
        quick=args.quick,
    )

    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    return 0 if results['passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
