"""
CIFAR-10 CNN Hyperparameter Optimization Benchmark

Medium-difficulty ML benchmark (~2min per trial)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, Any, List
from pathlib import Path


class SimpleCNN(nn.Module):
    """Configurable CNN for CIFAR-10."""

    def __init__(
        self,
        num_conv_layers: int = 3,
        num_filters: int = 64,
        kernel_size: int = 3,
        fc_size: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.features = self._make_conv_layers(
            num_conv_layers, num_filters, kernel_size, dropout
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            flat_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, 10),
        )

    def _make_conv_layers(
        self,
        num_layers: int,
        num_filters: int,
        kernel_size: int,
        dropout: float
    ) -> nn.Sequential:
        """Build convolutional layers."""
        layers: List[nn.Module] = []
        in_channels = 3

        for i in range(num_layers):
            out_channels = num_filters * (2 ** min(i, 2))  # Cap growth
            padding = kernel_size // 2

            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout / 2),
            ])
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((2, 2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CIFAR10Benchmark:
    """CIFAR-10 CNN optimization benchmark."""

    def __init__(
        self,
        data_dir: str = './data',
        epochs_per_trial: int = 10,
        subset_size: int = 10000,
        device: str = 'cuda',
    ):
        """
        Initialize CIFAR-10 benchmark.

        Args:
            data_dir: Directory for CIFAR data
            epochs_per_trial: Training epochs per evaluation
            subset_size: Number of training samples (for speed)
            device: Device for training
        """
        self.epochs = epochs_per_trial
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.data_dir = data_dir

        # Data augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

        train_full = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_full = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=test_transform
        )

        # Use subset for speed
        np.random.seed(42)
        indices = np.random.choice(len(train_full), subset_size, replace=False)
        self.train_data = Subset(train_full, indices)
        self.test_data = test_full

    def evaluate(self, params: Dict[str, Any]) -> float:
        """
        Train CNN with given hyperparameters and return validation error.

        Args:
            params: Dict with keys:
                - num_conv_layers: int (2-5)
                - num_filters: int (16-128)
                - kernel_size: int (3 or 5)
                - fc_size: int (64-512)
                - learning_rate: float (1e-5 to 1e-2)
                - dropout: float (0.0-0.5)
                - batch_size: int (32-128)

        Returns:
            Validation error (1 - accuracy), lower is better
        """
        try:
            # Create model
            model = SimpleCNN(
                num_conv_layers=int(params.get('num_conv_layers', 3)),
                num_filters=int(params.get('num_filters', 64)),
                kernel_size=int(params.get('kernel_size', 3)),
                fc_size=int(params.get('fc_size', 256)),
                dropout=float(params.get('dropout', 0.2)),
            ).to(self.device)

            # Data loaders
            train_loader = DataLoader(
                self.train_data,
                batch_size=int(params.get('batch_size', 64)),
                shuffle=True,
                num_workers=2,
            )
            test_loader = DataLoader(
                self.test_data,
                batch_size=128,
                num_workers=2,
            )

            # Optimizer
            lr = float(params.get('learning_rate', 0.001))
            opt = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
            criterion = nn.CrossEntropyLoss()

            # Training
            model.train()
            for epoch in range(self.epochs):
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    opt.step()
                scheduler.step()

            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = model(x)
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)

            accuracy = correct / total
            return 1.0 - accuracy  # Return error (minimize)

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 1.0  # Return worst case

    @property
    def search_space(self) -> Dict[str, Dict[str, Any]]:
        """Return the search space definition."""
        return {
            'num_conv_layers': {'type': 'int', 'low': 2, 'high': 5},
            'num_filters': {'type': 'int', 'low': 16, 'high': 128},
            'kernel_size': {'type': 'categorical', 'choices': [3, 5]},
            'fc_size': {'type': 'int', 'low': 64, 'high': 512},
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'batch_size': {'type': 'int', 'low': 32, 'high': 128},
        }

    @property
    def param_specs(self):
        """Get parameter specifications for tokenizer."""
        from ...data.tokenizer import ParameterSpec
        return [
            ParameterSpec('num_conv_layers', 2, 5, is_integer=True),
            ParameterSpec('num_filters', 16, 128, is_integer=True),
            ParameterSpec('kernel_size', 0, 1, is_categorical=True, categories=['3', '5']),
            ParameterSpec('fc_size', 64, 512, is_integer=True),
            ParameterSpec('learning_rate', 1e-5, 1e-2, log_scale=True),
            ParameterSpec('dropout', 0.0, 0.5),
            ParameterSpec('batch_size', 32, 128, is_integer=True),
        ]
