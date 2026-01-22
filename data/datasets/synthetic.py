"""
PyTorch Dataset for Optimization Trajectories

Loads tokenized optimization trajectories for training the OptiFormer model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import random
import numpy as np

from data.tokenizer import SequenceTokenizer, ParameterSpec, Trial


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset of tokenized optimization trajectories.

    Loads trajectories from JSONL files and tokenizes them on-the-fly.
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: SequenceTokenizer,
        max_length: int = 512,
        shuffle_history: bool = True,
        subsample_prob: float = 0.3,
        min_trials: int = 5,
        cache_tokens: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to JSONL trajectory file
            tokenizer: Sequence tokenizer instance
            max_length: Maximum sequence length
            shuffle_history: If True, randomly shuffle history for augmentation
            subsample_prob: Probability of subsampling trajectory
            min_trials: Minimum number of trials to keep
            cache_tokens: If True, cache tokenized sequences (uses more memory)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_history = shuffle_history
        self.subsample_prob = subsample_prob
        self.min_trials = min_trials
        self.cache_tokens = cache_tokens

        # Load all trajectories
        self.trajectories = []
        self.param_names = list(tokenizer.param_specs.keys())

        data_path = Path(data_path)
        if data_path.exists():
            with open(data_path, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    self.trajectories.append(record)

        # Token cache (if enabled)
        self._token_cache: Dict[int, List[int]] = {}

    def __len__(self) -> int:
        return len(self.trajectories)

    def _convert_to_trials(self, trajectory: List[Dict]) -> List[Trial]:
        """Convert trajectory dicts to Trial objects."""
        trials = []
        for t in trajectory:
            params = {}
            for name in self.param_names:
                if name in t['params']:
                    params[name] = t['params'][name]
            trials.append(Trial(params=params, score=t['score']))
        return trials

    def _augment_trajectory(self, trials: List[Trial]) -> List[Trial]:
        """Apply data augmentation to trajectory."""
        # Subsample with probability
        if random.random() < self.subsample_prob and len(trials) > self.min_trials:
            n_keep = random.randint(self.min_trials, len(trials))
            # Always keep the last trial and best trial
            best_idx = np.argmin([t.score for t in trials])
            keep_indices = {len(trials) - 1, best_idx}

            # Sample remaining indices
            available = list(set(range(len(trials))) - keep_indices)
            n_sample = min(n_keep - len(keep_indices), len(available))
            if n_sample > 0:
                keep_indices.update(random.sample(available, n_sample))

            trials = [trials[i] for i in sorted(keep_indices)]

        # Shuffle history (preserving last trial)
        if self.shuffle_history and len(trials) > 1:
            last = trials[-1]
            rest = trials[:-1]
            random.shuffle(rest)
            trials = rest + [last]

        return trials

    def _compute_target_regret(self, trials: List[Trial]) -> int:
        """Compute target regret level based on best score."""
        best_score = min(t.score for t in trials)
        if best_score < 0.1:
            return 0  # Optimal
        elif best_score < 0.3:
            return 1  # Good
        else:
            return 2  # Poor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.trajectories[idx]
        trajectory = record['trajectory']

        # Convert to Trial objects
        trials = self._convert_to_trials(trajectory)

        # Apply augmentation
        trials = self._augment_trajectory(trials)

        # Compute target regret
        target_regret = self._compute_target_regret(trials)

        # Tokenize
        token_ids = self.tokenizer.encode_trajectory(trials, target_regret)

        # Truncate if needed
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Create input/label pairs (shifted by 1)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'n_dims': record.get('n_dims', len(self.param_names)),
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Collate function with padding.

    Args:
        batch: List of samples from dataset
        pad_token_id: Token ID for padding

    Returns:
        Batched and padded tensors
    """
    max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len

        # Pad input_ids
        input_ids.append(torch.cat([
            item['input_ids'],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ]))

        # Pad labels with -100 (ignored in loss)
        labels.append(torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ]))

        # Create attention mask
        attention_mask.append(torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ]))

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask),
    }


def create_dataloaders(
    train_path: Path,
    val_path: Path,
    tokenizer: SequenceTokenizer,
    batch_size: int = 64,
    max_length: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_path: Path to training data JSONL
        val_path: Path to validation data JSONL
        tokenizer: Sequence tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TrajectoryDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=max_length,
        shuffle_history=True,
    )

    val_dataset = TrajectoryDataset(
        data_path=val_path,
        tokenizer=tokenizer,
        max_length=max_length,
        shuffle_history=False,
        subsample_prob=0.0,  # No augmentation for validation
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


class InMemoryDataset(Dataset):
    """
    In-memory dataset for small-scale experiments.

    Stores all tokenized sequences in memory for faster iteration.
    """

    def __init__(
        self,
        trajectories: List[List[Trial]],
        tokenizer: SequenceTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize in-memory dataset.

        Args:
            trajectories: List of trial lists
            tokenizer: Sequence tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize all trajectories
        self.token_sequences = []
        for trials in trajectories:
            target_regret = 0 if min(t.score for t in trials) < 0.1 else 1
            tokens = tokenizer.encode_trajectory(trials, target_regret)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            self.token_sequences.append(tokens)

    def __len__(self) -> int:
        return len(self.token_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.token_sequences[idx]
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long),
        }
