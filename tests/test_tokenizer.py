"""
Comprehensive tokenizer tests - ALL MUST PASS before any training.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer.numerical import NumericalTokenizer, NumericalTokenizerConfig
from data.tokenizer.sequence import SequenceTokenizer, ParameterSpec, Trial
from data.tokenizer.vocabulary import Vocabulary, VocabularyConfig


class TestNumericalTokenizer:
    """Tests for numerical (float) tokenization."""

    @pytest.fixture
    def tokenizer(self):
        return NumericalTokenizer(NumericalTokenizerConfig(num_bins=1000))

    def test_basic_roundtrip(self, tokenizer):
        """Basic encode-decode should preserve value within tolerance."""
        test_values = [0.0, 0.5, 1.0, 0.123, 0.999, 0.001]
        for val in test_values:
            token = tokenizer.encode(val, 0.0, 1.0, log_scale=False)
            recovered = tokenizer.decode(token, 0.0, 1.0, log_scale=False)
            assert abs(val - recovered) < 0.002, f"Failed for {val}: got {recovered}"

    def test_range_mapping(self, tokenizer):
        """Test encoding with different ranges."""
        # Learning rate range
        lr = 0.001
        token = tokenizer.encode(lr, 1e-5, 1e-1, log_scale=True)
        recovered = tokenizer.decode(token, 1e-5, 1e-1, log_scale=True)
        # Log scale has higher relative error at low values
        relative_error = abs(lr - recovered) / lr
        assert relative_error < 0.05, f"LR roundtrip error too high: {relative_error}"

    def test_log_scale_preserves_resolution(self, tokenizer):
        """Log scale should give good resolution at small values."""
        small_values = [1e-5, 1e-4, 1e-3, 1e-2]
        for val in small_values:
            token = tokenizer.encode(val, 1e-5, 1e-1, log_scale=True)
            recovered = tokenizer.decode(token, 1e-5, 1e-1, log_scale=True)
            relative_error = abs(val - recovered) / val
            assert relative_error < 0.1, f"Log scale error at {val}: {relative_error}"

    def test_boundary_values(self, tokenizer):
        """Edge cases at boundaries."""
        # Exactly at low bound
        token = tokenizer.encode(0.0, 0.0, 1.0, False)
        assert token == 0

        # Exactly at high bound
        token = tokenizer.encode(1.0, 0.0, 1.0, False)
        assert token == 999  # Should be max bin

    def test_out_of_range_clamping(self, tokenizer):
        """Values outside range should be clamped."""
        # Below range
        token = tokenizer.encode(-0.5, 0.0, 1.0, False)
        assert token == 0

        # Above range
        token = tokenizer.encode(1.5, 0.0, 1.0, False)
        assert token == 999

    def test_batch_roundtrip_statistics(self, tokenizer):
        """Statistical test over many random values."""
        np.random.seed(42)
        errors = []
        for _ in range(10000):
            val = np.random.uniform(0.0, 1.0)
            token = tokenizer.encode(val, 0.0, 1.0, False)
            recovered = tokenizer.decode(token, 0.0, 1.0, False)
            errors.append(abs(val - recovered))

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        assert mean_error < 0.001, f"Mean error too high: {mean_error}"
        assert max_error < 0.002, f"Max error too high: {max_error}"

    def test_token_range(self, tokenizer):
        """All tokens should be in valid range."""
        for _ in range(1000):
            val = np.random.uniform(0.0, 1.0)
            token = tokenizer.encode(val, 0.0, 1.0, False)
            assert 0 <= token < tokenizer.num_bins

    def test_integer_range(self, tokenizer):
        """Test encoding integer ranges (batch_size, hidden_size, etc.)."""
        # Batch size: 32 to 256
        for batch_size in [32, 64, 128, 256]:
            token = tokenizer.encode(batch_size, 32, 256, False)
            recovered = tokenizer.decode(token, 32, 256, False)
            assert abs(batch_size - round(recovered)) <= 1, f"Batch size error at {batch_size}"

    def test_array_encoding(self, tokenizer):
        """Test encoding arrays of values."""
        values = np.array([0.1, 0.5, 0.9])
        tokens = tokenizer.encode(values, 0.0, 1.0, False)
        recovered = tokenizer.decode(tokens, 0.0, 1.0, False)
        assert len(tokens) == 3
        assert np.allclose(values, recovered, atol=0.002)


class TestVocabulary:
    """Tests for vocabulary management."""

    @pytest.fixture
    def vocab(self):
        return Vocabulary(VocabularyConfig(num_bins=1000))

    def test_special_tokens_exist(self, vocab):
        """Special tokens should be registered."""
        assert vocab.pad_token_id >= 0
        assert vocab.bos_token_id >= 0
        assert vocab.eos_token_id >= 0
        assert vocab.trial_sep_token_id >= 0
        assert vocab.score_token_id >= 0

    def test_special_tokens_distinct(self, vocab):
        """Special tokens should have distinct IDs."""
        ids = [
            vocab.pad_token_id,
            vocab.bos_token_id,
            vocab.eos_token_id,
            vocab.trial_sep_token_id,
            vocab.score_token_id,
        ]
        assert len(ids) == len(set(ids))

    def test_register_parameter(self, vocab):
        """Parameter registration should work."""
        token_id = vocab.register_parameter("learning_rate")
        assert token_id >= vocab.param_start

        # Second registration should return same ID
        token_id_2 = vocab.register_parameter("learning_rate")
        assert token_id == token_id_2

    def test_register_categorical(self, vocab):
        """Categorical registration should work."""
        vocab.register_parameter("optimizer")
        tokens = vocab.register_categorical("optimizer", ["adam", "sgd"])
        assert len(tokens) == 2
        assert tokens["adam"] != tokens["sgd"]

    def test_decode_token(self, vocab):
        """Token decoding should work."""
        # Numerical
        token_type, value = vocab.decode_token(500)
        assert token_type == "numerical"
        assert value == 500

        # Special
        token_type, value = vocab.decode_token(vocab.bos_token_id)
        assert token_type == "special"
        assert "<BOS>" in value

    def test_vocab_size_grows(self, vocab):
        """Vocab size should grow with registrations."""
        initial_size = vocab.vocab_size

        vocab.register_parameter("lr")
        vocab.register_parameter("bs")
        vocab.register_categorical("optimizer", ["adam", "sgd", "rmsprop"])

        assert vocab.vocab_size > initial_size


class TestSequenceTokenizer:
    """Tests for full sequence tokenization."""

    @pytest.fixture
    def param_specs(self):
        return [
            ParameterSpec("learning_rate", 1e-5, 1e-1, log_scale=True),
            ParameterSpec("batch_size", 16, 256, log_scale=False, is_integer=True),
            ParameterSpec("dropout", 0.0, 0.5, log_scale=False),
            ParameterSpec("optimizer", 0, 1, is_categorical=True,
                         categories=["adam", "sgd"]),
        ]

    @pytest.fixture
    def tokenizer(self, param_specs):
        return SequenceTokenizer(param_specs, num_bins=1000)

    def test_trajectory_roundtrip(self, tokenizer):
        """Full trajectory encode-decode roundtrip."""
        trials = [
            Trial({"learning_rate": 0.01, "batch_size": 64,
                   "dropout": 0.2, "optimizer": "adam"}, score=0.15),
            Trial({"learning_rate": 0.001, "batch_size": 128,
                   "dropout": 0.1, "optimizer": "sgd"}, score=0.12),
        ]

        tokens = tokenizer.encode_trajectory(trials, target_regret=0)
        recovered = tokenizer.decode_trajectory(tokens)

        assert len(recovered) == len(trials)
        for orig, rec in zip(trials, recovered):
            # Check categorical exact match
            assert orig.params["optimizer"] == rec.params["optimizer"]
            # Check numerical within tolerance
            assert abs(orig.params["dropout"] - rec.params["dropout"]) < 0.01
            # Check integer
            assert abs(orig.params["batch_size"] - rec.params["batch_size"]) <= 1

    def test_vocab_size(self, tokenizer):
        """Vocab size should match configuration."""
        assert tokenizer.vocab_size > 1000  # At least numerical bins
        assert tokenizer.vocab_size < 1500  # Not too large for smoke test

    def test_encode_for_inference(self, tokenizer):
        """Inference encoding should work."""
        history = [
            Trial({"learning_rate": 0.01, "batch_size": 64,
                   "dropout": 0.2, "optimizer": "adam"}, score=0.15),
        ]

        tokens = tokenizer.encode_for_inference(history)
        assert len(tokens) > 0
        # Should start with BOS
        assert tokens[0] == tokenizer.vocab.bos_token_id

    def test_decode_predicted_value(self, tokenizer):
        """Predicted value decoding should work."""
        # Numerical
        value = tokenizer.decode_predicted_value(500, "dropout")
        assert 0.0 <= value <= 0.5

        # Categorical
        cat_token = tokenizer.vocab.get_categorical_token("optimizer", "adam")
        value = tokenizer.decode_predicted_value(cat_token, "optimizer")
        assert value == "adam"

    def test_empty_trajectory(self, tokenizer):
        """Empty trajectory should encode/decode."""
        tokens = tokenizer.encode_trajectory([], target_regret=0)
        assert tokenizer.vocab.bos_token_id in tokens
        assert tokenizer.vocab.eos_token_id in tokens

        recovered = tokenizer.decode_trajectory(tokens)
        assert len(recovered) == 0

    def test_single_trial(self, tokenizer):
        """Single trial should encode/decode."""
        trials = [
            Trial({"learning_rate": 0.01, "batch_size": 64,
                   "dropout": 0.2, "optimizer": "adam"}, score=0.15),
        ]

        tokens = tokenizer.encode_trajectory(trials, target_regret=0)
        recovered = tokenizer.decode_trajectory(tokens)

        assert len(recovered) == 1
        assert recovered[0].params["optimizer"] == "adam"

    def test_different_target_regrets(self, tokenizer):
        """Different target regrets should produce different tokens."""
        trials = [
            Trial({"learning_rate": 0.01, "batch_size": 64,
                   "dropout": 0.2, "optimizer": "adam"}, score=0.15),
        ]

        tokens_0 = tokenizer.encode_trajectory(trials, target_regret=0)
        tokens_1 = tokenizer.encode_trajectory(trials, target_regret=1)
        tokens_2 = tokenizer.encode_trajectory(trials, target_regret=2)

        # Second token should be different (target regret)
        assert tokens_0[1] != tokens_1[1]
        assert tokens_1[1] != tokens_2[1]

    def test_score_normalization(self, tokenizer):
        """Scores should be properly normalized."""
        trials = [
            Trial({"learning_rate": 0.01, "batch_size": 64,
                   "dropout": 0.2, "optimizer": "adam"}, score=0.0),  # Best
            Trial({"learning_rate": 0.001, "batch_size": 128,
                   "dropout": 0.1, "optimizer": "sgd"}, score=1.0),  # Worst
        ]

        tokens = tokenizer.encode_trajectory(trials, target_regret=0)
        recovered = tokenizer.decode_trajectory(tokens)

        assert abs(recovered[0].score - 0.0) < 0.01
        assert abs(recovered[1].score - 1.0) < 0.01


class TestTokenizerIntegration:
    """Integration tests combining all tokenizer components."""

    def test_full_pipeline(self):
        """Test complete tokenization pipeline."""
        # Define search space
        param_specs = [
            ParameterSpec("lr", 1e-5, 1e-1, log_scale=True),
            ParameterSpec("hidden", 32, 512, is_integer=True),
            ParameterSpec("activation", 0, 1, is_categorical=True,
                         categories=["relu", "gelu", "tanh"]),
        ]

        tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

        # Create synthetic trajectory
        np.random.seed(42)
        trials = []
        for _ in range(10):
            trials.append(Trial(
                params={
                    "lr": np.random.uniform(1e-5, 1e-1),
                    "hidden": np.random.randint(32, 513),
                    "activation": np.random.choice(["relu", "gelu", "tanh"]),
                },
                score=np.random.uniform(0, 1),
            ))

        # Encode and decode
        tokens = tokenizer.encode_trajectory(trials, target_regret=0)
        recovered = tokenizer.decode_trajectory(tokens)

        assert len(recovered) == len(trials)

        # Verify preservation
        for orig, rec in zip(trials, recovered):
            assert orig.params["activation"] == rec.params["activation"]
            assert abs(orig.params["hidden"] - rec.params["hidden"]) <= 1

    def test_inference_workflow(self):
        """Test inference encoding workflow."""
        param_specs = [
            ParameterSpec("x", 0.0, 1.0),
            ParameterSpec("y", 0.0, 1.0),
        ]

        tokenizer = SequenceTokenizer(param_specs, num_bins=1000)

        # Simulate optimization history
        history = [
            Trial({"x": 0.5, "y": 0.5}, score=0.3),
            Trial({"x": 0.3, "y": 0.7}, score=0.2),
        ]

        # Encode for inference (predicting next params)
        tokens = tokenizer.encode_for_inference(history)

        # Should end with the first param token
        assert tokens[-1] == tokenizer.vocab.get_param_token("x")

        # Simulate model prediction and decode
        predicted_token = 750  # Some numerical token
        x_value = tokenizer.decode_predicted_value(predicted_token, "x")
        assert 0.0 <= x_value <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
