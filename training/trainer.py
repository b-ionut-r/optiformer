"""
Training Loop for OptiFormer

Main training implementation with logging, checkpointing, and early stopping.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import time
import json
import numpy as np

from ..model import OptiFormer
from ..data.datasets import collate_fn


class TrainingConfig:
    """Training configuration."""

    def __init__(
        self,
        # Optimization
        batch_size: int = 128,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        # Schedule
        max_steps: int = 5000,
        warmup_steps: int = 200,
        lr_scheduler: str = "cosine",
        # Precision
        fp16: bool = True,
        bf16: bool = False,
        # Logging
        logging_steps: int = 50,
        eval_steps: int = 500,
        save_steps: int = 1000,
        # Early stopping
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.001,
        # Device
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.lr_scheduler = lr_scheduler
        self.fp16 = fp16
        self.bf16 = bf16
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = device

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


class EarlyStopping:
    """Early stopping tracker."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_loss
            return False

        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class OptiFormerTrainer:
    """
    Trainer for OptiFormer model.

    Handles training loop, evaluation, logging, and checkpointing.
    """

    def __init__(
        self,
        model: OptiFormer,
        train_dataset,
        val_dataset,
        config: TrainingConfig,
        output_dir: Path,
        pad_token_id: int = 0,
    ):
        """
        Initialize trainer.

        Args:
            model: OptiFormer model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            output_dir: Directory for outputs
            pad_token_id: Token ID for padding
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pad_token_id = pad_token_id

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id),
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_token_id),
            num_workers=2,
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler
        total_steps = config.max_steps
        warmup_steps = config.warmup_steps

        if config.lr_scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        # Mixed precision
        self.use_amp = config.fp16 or config.bf16
        self.scaler = GradScaler() if config.fp16 else None
        self.amp_dtype = torch.bfloat16 if config.bf16 else torch.float16

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[Tuple[int, float]] = []
        self.initial_loss = None

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dict with training results
        """
        self.model.train()
        start_time = time.time()

        pbar = tqdm(total=self.config.max_steps, desc="Training")
        train_iter = iter(self.train_loader)

        accumulated_loss = 0.0
        num_accumulated = 0

        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

            accumulated_loss += loss.item() * self.config.gradient_accumulation_steps
            num_accumulated += 1

            # Gradient accumulation step
            if num_accumulated >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                # Record loss
                step_loss = accumulated_loss / num_accumulated
                self.train_losses.append(step_loss)

                if self.initial_loss is None:
                    self.initial_loss = step_loss

                accumulated_loss = 0.0
                num_accumulated = 0

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = sum(self.train_losses[-50:]) / min(len(self.train_losses), 50)
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}'
                    })

                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.global_step > 0:
                    val_loss = self.evaluate()
                    self.val_losses.append((self.global_step, val_loss))

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best.pt')

                    # Early stopping check
                    if self.early_stopping(val_loss):
                        print(f"\nEarly stopping triggered at step {self.global_step}")
                        break

                    self.model.train()

                # Regular checkpointing
                if self.global_step % self.config.save_steps == 0 and self.global_step > 0:
                    self.save_checkpoint(f'step_{self.global_step}.pt')

                self.global_step += 1
                pbar.update(1)

        pbar.close()

        # Final save
        self.save_checkpoint('final.pt')

        # Calculate results
        elapsed_time = time.time() - start_time
        final_train_loss = sum(self.train_losses[-100:]) / min(len(self.train_losses), 100)
        loss_reduction = (self.initial_loss - final_train_loss) / self.initial_loss if self.initial_loss else 0

        results = {
            'final_train_loss': final_train_loss,
            'initial_train_loss': self.initial_loss,
            'best_val_loss': self.best_val_loss,
            'loss_reduction': loss_reduction,
            'total_steps': self.global_step,
            'elapsed_time': elapsed_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        # Save results
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump({
                k: v for k, v in results.items()
                if k not in ['train_losses', 'val_losses']
            }, f, indent=2)

        return results

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                    )
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )

            total_loss += outputs.loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"\nStep {self.global_step} - Val Loss: {avg_loss:.4f}")

        return avg_loss

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        self.model.save(str(path))
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / filename
        self.model = OptiFormer.load(str(path), device=str(self.device))


def train_optiformer(
    model: OptiFormer,
    train_dataset,
    val_dataset,
    config: TrainingConfig,
    output_dir: str,
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """
    Convenience function to train OptiFormer.

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        output_dir: Output directory
        pad_token_id: Padding token ID

    Returns:
        Training results dict
    """
    trainer = OptiFormerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        output_dir=Path(output_dir),
        pad_token_id=pad_token_id,
    )

    return trainer.train()
