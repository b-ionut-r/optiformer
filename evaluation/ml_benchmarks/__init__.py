"""
ML Benchmarks for Hyperparameter Optimization

Real-world machine learning tasks for evaluating optimization performance.
"""

from .mnist_mlp import MNISTBenchmark, MLP
from .cifar_cnn import CIFAR10Benchmark, SimpleCNN
from .fashion_mlp import FashionMNISTBenchmark, FashionMLP

__all__ = [
    'MNISTBenchmark',
    'MLP',
    'CIFAR10Benchmark',
    'SimpleCNN',
    'FashionMNISTBenchmark',
    'FashionMLP',
]
