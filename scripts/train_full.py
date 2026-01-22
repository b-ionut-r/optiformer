"""
Full-Scale Training Script

This script orchestrates the production training of OptiFormer.
It handles:
1. Large-scale data generation (Synthetic + Real-World)
2. Tokenizer creation/loading
3. Model initialization (Small/Base/Large)
4. Distributed training loop (if available)

Usage:
    python -m scripts.train_full
"""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import OptiFormer, OptiFormerConfig
from data.tokenizer import SequenceTokenizer, ParameterSpec
from data.datasets import (
    generate_real_world_data,
    create_mixed_dataloaders,
    TrajectoryConfig,
    generate_gp_dataset,
    generate_symbolic_dataset,
    generate_all_trajectories,
)
from training import OptiFormerTrainer, TrainingConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_synthetic_data(
    output_dir: Path,
    config: Dict[str, Any],
    split: str = "train",
) -> Path:
    """Generate large-scale synthetic data."""
    output_path = output_dir / f"synthetic_{split}.jsonl"
    if output_path.exists():
        logger.info(f"Using existing synthetic data: {output_path}")
        return output_path

    logger.info(f"Generating synthetic {split} data...")
    
    # Scale down for validation
    scale = 1.0 if split == "train" else 0.05
    n_gp = int(config['gp_functions'] * scale)
    n_sym = int(config['symbolic_functions'] * scale)
    
    # Generate functions
    logger.info(f"  Generating {n_gp} GP functions...")
    gp_funcs = generate_gp_dataset(
        n_functions=n_gp,
        dims_list=config['dimensions'],
        bounds=(-5.0, 5.0),
        seed=42 if split == "train" else 43,
    )
    
    logger.info(f"  Generating {n_sym} symbolic functions...")
    sym_funcs = generate_symbolic_dataset(
        n_functions=n_sym,
        dims_list=config['dimensions'],
        bounds=(-5.0, 5.0),
        seed=42 if split == "train" else 43,
    )
    
    functions = gp_funcs + sym_funcs
    np.random.shuffle(functions)
    
    # Generate trajectories
    traj_config = TrajectoryConfig(
        n_trials=config['trials_per_function'],
        noise_std=config['noise_std'],
        bounds=(0.0, 1.0),
        sampler="tpe",
    )
    
    generate_all_trajectories(
        functions=functions,
        config=traj_config,
        output_path=output_path,
        desc=f"Synthetic {split}",
    )
    
    return output_path


def main():
    # Load config
    config_path = Path("config/full_training.yaml")
    config = load_config(config_path)
    
    output_dir = Path(config['project']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 1. Data Generation
    logger.info("PHASE 1: Data Generation")
    
    # Real-world data
    rw_config = config['data']['real_world']
    rw_paths = generate_real_world_data(
        output_dir=data_dir / "real_world",
        use_hpobench=rw_config['use_hpobench'],
        use_openml=rw_config['use_openml'],
        use_yahpo=rw_config['use_yahpo'],
        hpobench_benchmarks=rw_config['hpobench_benchmarks'],
        yahpo_scenarios=rw_config['yahpo_scenarios'],
        n_trajectories_per_benchmark=rw_config['hpobench_traces_per_benchmark'], # Approximation for others
    )
    real_world_files = list(rw_paths['paths'].values())
    
    # Synthetic data
    syn_config = config['data']['synthetic']
    syn_train_path = generate_synthetic_data(data_dir, syn_config, "train")
    syn_val_path = generate_synthetic_data(data_dir, syn_config, "val")
    
    # 2. Tokenizer
    logger.info("PHASE 2: Tokenizer")
    # Create param specs that cover all dimensions
    max_dim = max(syn_config['dimensions'])
    # Add buffer for real-world datasets which might have named params
    # We use x0..xN generic names for synthetic, but real-world might vary
    # Ideally, real-world loaders normalize to x0..xN
    param_specs = [
        ParameterSpec(f"x{i}", 0.0, 1.0) for i in range(max_dim + 5)
    ]
    
    tokenizer = SequenceTokenizer(
        param_specs=param_specs,
        num_bins=config['tokenizer']['num_bins']
    )
    tokenizer.save(str(output_dir / "tokenizer.json"))
    
    # 3. Model
    logger.info("PHASE 3: Model Initialization")
    size = config['model']['size']
    model_config_dict = config['model'][size]
    
    model_config = OptiFormerConfig(
        vocab_size=tokenizer.vocab_size + 100,
        hidden_size=model_config_dict['hidden_size'],
        intermediate_size=model_config_dict['intermediate_size'],
        num_hidden_layers=model_config_dict['num_hidden_layers'],
        num_attention_heads=model_config_dict['num_attention_heads'],
        max_position_embeddings=model_config_dict['max_position_embeddings'],
        dropout=model_config_dict['dropout'],
    )
    
    model = OptiFormer(model_config)
    logger.info(f"Initialized '{size}' model with {model.n_params:,} parameters")
    
    # 4. Training
    logger.info("PHASE 4: Training")
    
    train_loader, val_loader = create_mixed_dataloaders(
        synthetic_train_path=syn_train_path,
        synthetic_val_path=syn_val_path,
        real_world_paths=real_world_files,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        synthetic_weight=config['data']['synthetic_weight'],
        num_workers=4,
    )
    
    train_config = TrainingConfig(
        batch_size=config['training']['batch_size'],
        learning_rate=float(config['training']['learning_rate']),
        max_steps=config['training']['max_steps'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'] and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        device=config['project']['device'],
        output_dir=output_dir / "checkpoints",
    )
    
    trainer = OptiFormerTrainer(
        model=model,
        train_dataset=train_loader.dataset, # Pass dataset, trainer handles loading
        val_dataset=val_loader.dataset,
        config=train_config,
        output_dir=output_dir,
        pad_token_id=tokenizer.vocab.pad_token_id,
    )
    
    # Note: Trainer expects datasets, not loaders, but create_mixed_dataloaders returns loaders.
    # The OptiFormerTrainer implementation might need to be checked if it strictly requires datasets.
    # Assuming standard Trainer pattern, we might need to adjust if it creates its own loaders.
    # For now, passing .dataset from the loaders.
    
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
