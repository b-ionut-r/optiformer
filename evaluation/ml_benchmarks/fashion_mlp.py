"""
Fashion-MNIST MLP Hyperparameter Optimization Benchmark

Similar to MNIST but slightly harder - useful for testing generalization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, Any


class FashionMLP(nn.Module):
    """Configurable MLP for Fashion-MNIST."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: str = 'relu',
    ):
        super().__init__()

        layers = []
        in_size = 784

        act_fn_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
        }
        act_fn = act_fn_map.get(activation, nn.ReLU)

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                act_fn(),
                nn.Dropout(dropout),
            ])
            in_size = hidden_size

        layers.append(nn.Linear(hidden_size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class FashionMNISTBenchmark:
    """Fashion-MNIST MLP optimization benchmark."""

    def __init__(
        self,
        data_dir: str = './data',
        epochs_per_trial: int = 5,
        subset_size: int = 10000,
        device: str = 'cuda',
    ):
        """
        Initialize Fashion-MNIST benchmark.

        Args:
            data_dir: Directory for data
            epochs_per_trial: Training epochs per evaluation
            subset_size: Number of training samples (for speed)
            device: Device for training
        """
        self.epochs = epochs_per_trial
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.data_dir = data_dir

        # Load data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        train_full = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_full = datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=transform
        )

        # Use subset for speed
        np.random.seed(42)
        indices = np.random.choice(len(train_full), subset_size, replace=False)
        self.train_data = Subset(train_full, indices)
        self.test_data = test_full

    def evaluate(self, params: Dict[str, Any]) -> float:
        """
        Train MLP with given hyperparameters and return validation error.

        Args:
            params: Dict with hyperparameter values

        Returns:
            Validation error (1 - accuracy), lower is better
        """
        # Create model
        model = FashionMLP(
            hidden_size=int(params.get('hidden_size', 128)),
            num_layers=int(params.get('num_layers', 2)),
            dropout=float(params.get('dropout', 0.2)),
            activation=params.get('activation', 'relu'),
        ).to(self.device)

        # Data loaders
        train_loader = DataLoader(
            self.train_data,
            batch_size=int(params.get('batch_size', 64)),
            shuffle=True,
        )
        test_loader = DataLoader(self.test_data, batch_size=256)

        # Optimizer
        lr = float(params.get('learning_rate', 0.001))
        opt = optim.Adam(model.parameters(), lr=lr)
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
        return 1.0 - accuracy

    @property
    def search_space(self) -> Dict[str, Dict[str, Any]]:
        """Return the search space definition."""
        return {
            'hidden_size': {'type': 'int', 'low': 64, 'high': 512},
            'num_layers': {'type': 'int', 'low': 1, 'high': 4},
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'batch_size': {'type': 'int', 'low': 32, 'high': 256},
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'gelu']},
        }

    @property
    def param_specs(self):
        """Get parameter specifications for tokenizer."""
        from ...data.tokenizer import ParameterSpec
        return [
            ParameterSpec('hidden_size', 64, 512, is_integer=True),
            ParameterSpec('num_layers', 1, 4, is_integer=True),
            ParameterSpec('learning_rate', 1e-5, 1e-1, log_scale=True),
            ParameterSpec('dropout', 0.0, 0.5),
            ParameterSpec('batch_size', 32, 256, is_integer=True),
            ParameterSpec('activation', 0, 2, is_categorical=True, categories=['relu', 'tanh', 'gelu']),
        ]
