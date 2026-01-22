"""
OptiFormer Datasets

PyTorch datasets for training the OptiFormer model.
"""

from .synthetic import (
    TrajectoryDataset,
    InMemoryDataset,
    collate_fn,
    create_dataloaders,
)
from .hpobench import (
    HPOBenchLoader,
    RealWorldTraceCollector,
    load_openml_traces,
    try_import_hpobench,
    normalize_params_to_generic,
    get_hpobench_param_ranges,
)
from .combined import (
    CombinedTrajectoryDataset,
    create_mixed_dataloaders,
    generate_real_world_data,
)
from .benchmark_zoo import (
    YAHPOLoader,
    JAHSBenchLoader,
    LCBenchLoader,
    generate_diverse_real_world_data,
    print_installation_instructions,
    check_available_benchmarks,
    validate_benchmark_integration,
    try_import_yahpo,
    try_import_jahs,
)

__all__ = [
    # Synthetic
    'TrajectoryDataset',
    'InMemoryDataset',
    'collate_fn',
    'create_dataloaders',
    # Real-world
    'HPOBenchLoader',
    'RealWorldTraceCollector',
    'load_openml_traces',
    'try_import_hpobench',
    'normalize_params_to_generic',
    'get_hpobench_param_ranges',
    # Combined
    'CombinedTrajectoryDataset',
    'create_mixed_dataloaders',
    'generate_real_world_data',
    # Benchmark Zoo (diverse real-world benchmarks)
    'YAHPOLoader',
    'JAHSBenchLoader',
    'LCBenchLoader',
    'generate_diverse_real_world_data',
    'print_installation_instructions',
    'check_available_benchmarks',
    'validate_benchmark_integration',
    'try_import_yahpo',
    'try_import_jahs',
]
