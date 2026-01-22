"""
Smoke Test Phase 1: Tokenizer Validation

Validates that the tokenizer works correctly before proceeding
with data generation and training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple, Dict, Any

from data.tokenizer import (
    NumericalTokenizer,
    NumericalTokenizerConfig,
    SequenceTokenizer,
    ParameterSpec,
    Trial,
)


def test_numerical_tokenizer() -> Tuple[bool, Dict[str, Any]]:
    """Test numerical tokenizer roundtrip accuracy."""
    print("  Testing numerical tokenizer...")
    results = {'tests': [], 'errors': []}

    tokenizer = NumericalTokenizer(NumericalTokenizerConfig(num_bins=1000))

    # Test 1: Basic roundtrip
    test_values = [0.0, 0.5, 1.0, 0.123, 0.999, 0.001]
    max_error = 0.0
    for val in test_values:
        token = tokenizer.encode(val, 0.0, 1.0, log_scale=False)
        recovered = tokenizer.decode(token, 0.0, 1.0, log_scale=False)
        error = abs(val - recovered)
        max_error = max(max_error, error)

    passed = max_error < 0.002
    results['tests'].append({
        'name': 'basic_roundtrip',
        'passed': passed,
        'max_error': max_error,
    })
    if not passed:
        results['errors'].append(f"Basic roundtrip error too high: {max_error}")

    # Test 2: Log scale roundtrip
    test_lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    max_relative_error = 0.0
    for lr in test_lrs:
        token = tokenizer.encode(lr, 1e-5, 1e-1, log_scale=True)
        recovered = tokenizer.decode(token, 1e-5, 1e-1, log_scale=True)
        rel_error = abs(lr - recovered) / lr
        max_relative_error = max(max_relative_error, rel_error)

    passed = max_relative_error < 0.1
    results['tests'].append({
        'name': 'log_scale_roundtrip',
        'passed': passed,
        'max_relative_error': max_relative_error,
    })
    if not passed:
        results['errors'].append(f"Log scale error too high: {max_relative_error}")

    # Test 3: Statistical roundtrip
    np.random.seed(42)
    errors = []
    for _ in range(1000):
        val = np.random.uniform(0.0, 1.0)
        token = tokenizer.encode(val, 0.0, 1.0, False)
        recovered = tokenizer.decode(token, 0.0, 1.0, False)
        errors.append(abs(val - recovered))

    mean_error = np.mean(errors)
    passed = mean_error < 0.001
    results['tests'].append({
        'name': 'statistical_roundtrip',
        'passed': passed,
        'mean_error': mean_error,
    })
    if not passed:
        results['errors'].append(f"Statistical mean error too high: {mean_error}")

    # Test 4: Token range
    all_valid = True
    for _ in range(100):
        val = np.random.uniform(0.0, 1.0)
        token = tokenizer.encode(val, 0.0, 1.0, False)
        if not (0 <= token < 1000):
            all_valid = False
            break

    results['tests'].append({
        'name': 'token_range_valid',
        'passed': all_valid,
    })
    if not all_valid:
        results['errors'].append("Token out of range")

    all_passed = all(t['passed'] for t in results['tests'])
    return all_passed, results


def test_sequence_tokenizer() -> Tuple[bool, Dict[str, Any]]:
    """Test sequence tokenizer with full trajectories."""
    print("  Testing sequence tokenizer...")
    results = {'tests': [], 'errors': []}

    # Create tokenizer with sample param specs
    param_specs = [
        ParameterSpec("learning_rate", 1e-5, 1e-1, log_scale=True),
        ParameterSpec("batch_size", 16, 256, is_integer=True),
        ParameterSpec("dropout", 0.0, 0.5),
        ParameterSpec("optimizer", 0, 1, is_categorical=True, categories=["adam", "sgd"]),
    ]

    tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

    # Test 1: Vocabulary size
    vocab_size = tokenizer.vocab_size
    passed = 1000 < vocab_size < 1500
    results['tests'].append({
        'name': 'vocab_size_reasonable',
        'passed': passed,
        'vocab_size': vocab_size,
    })
    if not passed:
        results['errors'].append(f"Vocab size unexpected: {vocab_size}")

    # Test 2: Trajectory roundtrip
    trials = [
        Trial({"learning_rate": 0.01, "batch_size": 64, "dropout": 0.2, "optimizer": "adam"}, score=0.15),
        Trial({"learning_rate": 0.001, "batch_size": 128, "dropout": 0.1, "optimizer": "sgd"}, score=0.12),
    ]

    tokens = tokenizer.encode_trajectory(trials, target_regret=0)
    recovered = tokenizer.decode_trajectory(tokens)

    length_match = len(recovered) == len(trials)
    categorical_match = all(
        trials[i].params["optimizer"] == recovered[i].params["optimizer"]
        for i in range(len(trials))
    )

    passed = length_match and categorical_match
    results['tests'].append({
        'name': 'trajectory_roundtrip',
        'passed': passed,
        'length_match': length_match,
        'categorical_match': categorical_match,
    })
    if not passed:
        results['errors'].append("Trajectory roundtrip failed")

    # Test 3: Empty trajectory
    empty_tokens = tokenizer.encode_trajectory([], target_regret=0)
    empty_recovered = tokenizer.decode_trajectory(empty_tokens)
    passed = len(empty_recovered) == 0
    results['tests'].append({
        'name': 'empty_trajectory',
        'passed': passed,
    })

    # Test 4: Inference encoding
    tokens = tokenizer.encode_for_inference(trials[:1])
    passed = len(tokens) > 0 and tokens[0] == tokenizer.vocab.bos_token_id
    results['tests'].append({
        'name': 'inference_encoding',
        'passed': passed,
    })

    all_passed = all(t['passed'] for t in results['tests'])
    return all_passed, results


def test_tokenizer() -> Tuple[bool, Dict[str, Any]]:
    """Run all tokenizer tests."""
    print("\nPhase 1: Tokenizer Validation")
    print("-" * 40)

    results = {
        'numerical': {},
        'sequence': {},
    }

    # Test numerical tokenizer
    num_passed, num_results = test_numerical_tokenizer()
    results['numerical'] = num_results
    print(f"    Numerical tokenizer: {'PASS' if num_passed else 'FAIL'}")

    # Test sequence tokenizer
    seq_passed, seq_results = test_sequence_tokenizer()
    results['sequence'] = seq_results
    print(f"    Sequence tokenizer: {'PASS' if seq_passed else 'FAIL'}")

    all_passed = num_passed and seq_passed

    if not all_passed:
        print("\n  Errors:")
        for err in results['numerical'].get('errors', []):
            print(f"    - {err}")
        for err in results['sequence'].get('errors', []):
            print(f"    - {err}")

    return all_passed, results


if __name__ == "__main__":
    passed, results = test_tokenizer()
    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
