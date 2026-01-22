# OptiFormer: A Complete Guide to Building a Transformer-Based Hyperparameter Optimizer

This document explains the entire OptiFormer system from first principles. It's designed so you can understand every component and replicate or extend the work for your research.

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Why This Approach Works](#2-why-this-approach-works)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Component 1: Tokenization](#4-component-1-tokenization)
5. [Component 2: Data Generation](#5-component-2-data-generation)
6. [Component 3: The Model](#6-component-3-the-model)
7. [Component 4: Training](#7-component-4-training)
8. [Component 5: Inference & Sampling](#8-component-5-inference--sampling)
9. [Component 6: Evaluation](#9-component-6-evaluation)
10. [Key Design Decisions](#10-key-design-decisions)
11. [What Makes This Publishable](#11-what-makes-this-publishable)
12. [Extending the Work](#12-extending-the-work)
13. [References](#13-references)

---

## 1. The Core Idea

### Traditional Hyperparameter Optimization

Traditional Bayesian Optimization (BO) works like this:

```
1. Evaluate objective function at initial points
2. Fit a surrogate model (usually Gaussian Process) to observed data
3. Use an acquisition function (EI, UCB, etc.) to select next point
4. Repeat until budget exhausted
```

**Problems with traditional BO:**
- GP inference is O(n³) in the number of observations
- Acquisition function optimization can be expensive
- Doesn't transfer knowledge between different optimization tasks
- Requires careful tuning of kernel hyperparameters

### The OptiFormer Approach: Algorithm Distillation

Instead of implementing BO directly, we **learn optimization behavior** from an existing optimizer (TPE):

```
1. Generate many synthetic objective functions
2. Run TPE on each function, recording the full trajectory
3. Train a Transformer to predict "what would TPE do next?"
4. At inference time, use the Transformer instead of TPE
```

This is called **algorithm distillation** — we're distilling the knowledge of TPE into a neural network.

### Why a Transformer?

The optimization history is a **sequence**:

```
Trial 1: params={lr: 0.01, batch_size: 32}, score=0.85
Trial 2: params={lr: 0.001, batch_size: 64}, score=0.72
Trial 3: params={lr: 0.005, batch_size: 128}, score=0.68
...
Trial N: params={???}  ← What should we try next?
```

This is exactly what Transformers excel at — predicting the next element in a sequence given the history. The self-attention mechanism can:
- Identify which previous trials were good/bad
- Learn patterns like "if low learning rate worked, try even lower"
- Handle variable-length histories naturally

---

## 2. Why This Approach Works

### Theoretical Justification

1. **Universal Function Approximation**: Transformers can approximate any sequence-to-sequence function given enough capacity and data.

2. **Implicit Surrogate Modeling**: The Transformer implicitly learns a surrogate model through its attention weights — it learns which trials are "similar" and how to interpolate.

3. **Amortized Inference**: Unlike GP-BO which refits the surrogate at each step, the Transformer does a single forward pass. Training is expensive, inference is cheap.

4. **Transfer Learning Potential**: A Transformer trained on diverse functions can potentially generalize to new function types without retraining.

### Empirical Evidence (from literature)

- **OptFormer (Google, 2022)**: Similar approach, showed competitive performance with GP-BO
- **ABLATION (Berkeley, 2021)**: Demonstrated that learned optimizers can match or beat hand-designed ones
- **PFN4BO**: Prior-fitted networks achieve strong results with minimal hyperparameter tuning

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Function   │    │  Trajectory  │    │  Tokenized   │      │
│  │  Generators  │───▶│  Generator   │───▶│  Sequences   │      │
│  │  (GP/Symb)   │    │    (TPE)     │    │              │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│                                          ┌──────────────┐      │
│                                          │  Transformer │      │
│                                          │   Training   │      │
│                                          │  (Next-Token)│      │
│                                          └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Optimization│    │   Tokenize   │    │  Transformer │      │
│  │   History    │───▶│   History    │───▶│   Predict    │      │
│  │              │    │              │    │  Next Token  │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│                                          ┌──────────────┐      │
│                                          │   Decode to  │      │
│                                          │  Parameters  │      │
│                                          └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Component 1: Tokenization

### The Challenge

Transformers operate on discrete tokens, but hyperparameters are:
- **Continuous** (learning rate: 0.001)
- **Integer** (batch size: 32)
- **Categorical** (optimizer: "adam")
- **Log-scale** (learning rate varies over orders of magnitude)

We need to convert all of these to discrete tokens and back.

### Solution: Quantization

**For continuous parameters:**
```python
# Normalize to [0, 1]
normalized = (value - low) / (high - low)

# For log-scale parameters (like learning rate)
normalized = (log(value) - log(low)) / (log(high) - log(low))

# Quantize to one of 1000 bins
token = int(normalized * 999)  # Token in range [0, 999]
```

**Why 1000 bins?**
- Gives ~0.1% precision, sufficient for most hyperparameters
- Vocabulary stays manageable (1000 numerical tokens)
- Finer than typical human-specified precision

**Decoding (token → value):**
```python
# Use bin CENTER, not edge (reduces systematic bias)
normalized = (token + 0.5) / 1000
value = low + normalized * (high - low)
# For log-scale: value = exp(log(low) + normalized * (log(high) - log(low)))
```

### Vocabulary Structure

```
Token IDs:
[0-999]      : Numerical values (quantized)
[1000]       : PAD token
[1001]       : BOS (beginning of sequence)
[1002]       : EOS (end of sequence)
[1003]       : TRIAL_SEP (separates trials)
[1004]       : SCORE (precedes score value)
[1005-1007]  : TARGET_REGRET_0/1/2 (conditioning tokens)
[1008+]      : Parameter name tokens (x0, x1, ..., learning_rate, etc.)
[param_end+] : Categorical value tokens (adam, sgd, relu, etc.)
```

### Sequence Format

A trajectory is encoded as:

```
<BOS> <TARGET_REGRET_0>
  <TRIAL_SEP>
    <PARAM_learning_rate> [0.001 → token 234]
    <PARAM_batch_size> [64 → token 127]
    <PARAM_optimizer> [adam → token 1050]
    <SCORE> [0.85 → token 850]
  <TRIAL_SEP>
    <PARAM_learning_rate> [0.0001 → token 156]
    <PARAM_batch_size> [128 → token 255]
    <PARAM_optimizer> [sgd → token 1051]
    <SCORE> [0.72 → token 720]
  <TRIAL_SEP>
    <PARAM_learning_rate> [??? ← MODEL PREDICTS THIS]
<EOS>
```

### Why This Format?

1. **Parameter name tokens**: Tell the model which parameter comes next
2. **Fixed parameter order**: Makes learning easier (always same structure)
3. **Score after parameters**: Model sees full trial before score
4. **Target regret conditioning**: Can ask for "optimal" vs "good" vs "bad" suggestions

### Key Files

- `data/tokenizer/numerical.py`: Continuous value quantization
- `data/tokenizer/categorical.py`: Categorical value mapping
- `data/tokenizer/vocabulary.py`: Token ID management
- `data/tokenizer/sequence.py`: Full trajectory encoding/decoding

---

## 5. Component 2: Data Generation

### Why Synthetic Functions?

Real HPO data is:
- Expensive to collect (each trial = training a model)
- Limited in quantity (thousands, not millions)
- Biased toward specific domains

Synthetic functions let us generate unlimited training data with known optima.

### Function Type 1: Gaussian Process Priors

**What it is:** Sample functions from a GP prior with random kernel hyperparameters.

**Why it works:** GP samples are smooth, continuous functions that resemble many real objective landscapes.

```python
# Simplified GP sampling
def sample_gp_function(n_dims, kernel='rbf'):
    # Random lengthscale
    lengthscale = np.exp(np.random.randn() * 0.5)

    # Generate support points
    X_support = np.random.uniform(-5, 5, (100, n_dims))

    # Compute kernel matrix
    K = rbf_kernel(X_support, lengthscale=lengthscale)

    # Sample function values
    y_support = np.random.multivariate_normal(np.zeros(100), K)

    # Return interpolator
    return RBFInterpolator(X_support, y_support)
```

**Kernel choices:**
- **RBF (Radial Basis Function)**: Very smooth functions
- **Matern-1.5**: Moderately smooth (once differentiable)
- **Matern-2.5**: Smoother than 1.5, rougher than RBF

### Function Type 2: Symbolic Expressions

**What it is:** Randomly generated expression trees like `sin(x) + cos(y) * exp(-z²)`

**Why it works:** Captures non-smooth, discontinuous behavior that GPs can't represent (like real neural network loss landscapes).

```python
# Operators used
UNARY_OPS = [sin, cos, exp, abs, sqrt, square, negate]
BINARY_OPS = [add, subtract, multiply, protected_divide]

# Random tree generation
def random_expression(depth, n_dims):
    if depth == 0:
        return random_variable(n_dims) or random_constant()

    if random() < 0.3:  # Unary
        op = random.choice(UNARY_OPS)
        child = random_expression(depth - 1, n_dims)
        return op(child)
    else:  # Binary
        op = random.choice(BINARY_OPS)
        left = random_expression(depth - 1, n_dims)
        right = random_expression(depth - 1, n_dims)
        return op(left, right)
```

### Trajectory Generation

For each function, run TPE optimization:

```python
def generate_trajectory(objective_fn, n_trials=32):
    study = optuna.create_study(sampler=TPESampler())
    study.optimize(objective_fn, n_trials=n_trials)

    trajectory = []
    for trial in study.trials:
        trajectory.append({
            'params': trial.params,
            'score': trial.value,
        })
    return trajectory
```

**Why TPE as the teacher?**
- Efficient and well-understood
- Good default performance
- Fast enough to generate many trajectories
- Alternative: use ensemble of optimizers (TPE + CMA-ES + Random)

### Data Augmentation

1. **History shuffling**: Randomly permute trial order (optimization shouldn't depend on order)
2. **Subsampling**: Keep best trials + random subset (model sees varied history lengths)
3. **Noise injection**: Add small noise to parameters (regularization)

### Key Files

- `data/generators/gp_prior.py`: GP function sampling
- `data/generators/symbolic.py`: Expression tree generation
- `data/generators/trajectory.py`: TPE optimization runner

---

## 6. Component 3: The Model

### Architecture Choice: Decoder-Only Transformer

We use a decoder-only architecture (like GPT) because:
1. **Causal attention**: Each token only attends to previous tokens (natural for sequential prediction)
2. **Next-token prediction**: Standard LM objective works perfectly
3. **Efficient inference**: Generate one token at a time

### Implementation: LlamaForCausalLM

We use HuggingFace's Llama implementation because:
- Well-tested and optimized
- RoPE positional embeddings (better length generalization)
- Flash attention support
- Easy to scale up/down

```python
from transformers import LlamaForCausalLM, LlamaConfig

config = LlamaConfig(
    vocab_size=vocab_size,      # ~1200 tokens
    hidden_size=256,            # Model dimension
    num_hidden_layers=6,        # Transformer blocks
    num_attention_heads=8,      # Attention heads
    intermediate_size=1024,     # FFN hidden dim (4x hidden)
    max_position_embeddings=2048,
    rope_theta=10000.0,         # RoPE base frequency
)
model = LlamaForCausalLM(config)
```

### Model Sizes

| Size  | Hidden | Layers | Heads | FFN    | Params | Use Case |
|-------|--------|--------|-------|--------|--------|----------|
| Nano  | 256    | 6      | 8     | 1024   | ~8M    | Smoke tests |
| Small | 384    | 8      | 12    | 1536   | ~25M   | Validation |
| Base  | 512    | 12     | 16    | 2048   | ~50M   | Production |
| Large | 768    | 16     | 16    | 3072   | ~150M  | Full-scale |

### Training Objective

Standard causal language modeling (next-token prediction):

```python
# Input:  <BOS> <TR_0> <TS> <P_lr> [234] <P_bs> [127] <SCORE> [850] ...
# Target: <TR_0> <TS> <P_lr> [234] <P_bs> [127] <SCORE> [850] ... <EOS>

# Shift by 1 position
inputs = tokens[:-1]
targets = tokens[1:]

# Cross-entropy loss
loss = F.cross_entropy(model(inputs).logits, targets)
```

### Key Files

- `model/config.py`: Model configuration and size presets
- `model/optiformer.py`: Model wrapper with save/load utilities

---

## 7. Component 4: Training

### Training Configuration

```python
config = TrainingConfig(
    batch_size=128,
    learning_rate=3e-4,           # Standard for small transformers
    max_steps=50000,
    warmup_steps=1000,            # Linear warmup
    lr_scheduler='cosine',        # Cosine decay to 0
    weight_decay=0.1,             # AdamW regularization
    gradient_clip=1.0,            # Prevent explosions
    fp16=True,                    # Mixed precision
)
```

### Mixed Precision Training

Use FP16/BF16 for:
- 2x memory reduction
- Faster matrix multiplications on modern GPUs

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

with autocast('cuda', dtype=torch.float16):
    loss = model(batch).loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Early Stopping

Monitor validation loss:
```python
if val_loss < best_val_loss - min_delta:
    best_val_loss = val_loss
    patience_counter = 0
    save_checkpoint()
else:
    patience_counter += 1
    if patience_counter >= patience:
        stop_training()
```

### What to Monitor

1. **Training loss**: Should decrease steadily
2. **Validation loss**: Should decrease, then plateau
3. **Loss reduction %**: Aim for >60% from initial
4. **GPU memory**: Should stay within bounds
5. **Learning rate**: Should follow schedule

### Key Files

- `training/trainer.py`: Main training loop

---

## 8. Component 5: Inference & Sampling

### The Inference Process

Given optimization history, generate next trial:

```python
def suggest(history: List[Trial]) -> Dict[str, Any]:
    params = {}

    for param_name in param_order:
        # Encode history + params so far
        tokens = tokenizer.encode_for_inference(history, params)
        input_ids = torch.tensor([tokens])

        # Get model's probability distribution over next token
        logits = model(input_ids).logits[0, -1, :]  # Last position
        probs = softmax(logits / temperature)

        # Mask invalid tokens (only allow valid values for this param)
        valid_tokens = get_valid_tokens(param_name)
        probs[~valid_tokens] = 0
        probs = probs / probs.sum()  # Renormalize

        # Sample
        token = torch.multinomial(probs, 1).item()

        # Decode to parameter value
        value = tokenizer.decode_value(token, param_name)
        params[param_name] = value

    return params
```

### Sampling Strategies

**Temperature:**
- `T < 1.0`: More deterministic, exploit known good regions
- `T = 1.0`: As trained
- `T > 1.0`: More random, explore more

**Top-k sampling:**
```python
# Only consider top k tokens
top_k_probs, top_k_indices = probs.topk(k)
token = top_k_indices[torch.multinomial(top_k_probs, 1)]
```

**Top-p (nucleus) sampling:**
```python
# Consider smallest set of tokens with cumulative prob >= p
sorted_probs, sorted_indices = probs.sort(descending=True)
cumsum = sorted_probs.cumsum(0)
mask = cumsum <= p
token = sorted_indices[torch.multinomial(sorted_probs * mask, 1)]
```

### Optuna Integration

Implement `BaseSampler` interface:

```python
class OptiFormerSampler(BaseSampler):
    def sample_relative(self, study, trial, search_space):
        # Convert Optuna history to our format
        history = self._convert_history(study.trials)

        # Use model to suggest
        suggestion = self.generator.suggest(history)

        return suggestion
```

### Key Files

- `model/generation.py`: Inference utilities
- `samplers/optiformer_sampler.py`: Optuna integration

---

## 9. Component 6: Evaluation

### Synthetic Benchmarks

Standard optimization test functions:

| Function   | Properties | Optimum |
|------------|------------|---------|
| Sphere     | Convex, smooth | f(0,...,0) = 0 |
| Rastrigin  | Many local minima | f(0,...,0) = 0 |
| Rosenbrock | Curved valley | f(1,...,1) = 0 |
| Ackley     | Flat + deep hole | f(0,...,0) = 0 |
| Levy       | Complex multimodal | f(1,...,1) = 0 |

**Evaluation protocol:**
```python
for benchmark in [sphere, rastrigin, rosenbrock, ackley, levy]:
    for seed in range(n_seeds):
        # Run optimization
        optiformer_result = optimize(benchmark, OptiFormerSampler(), n_trials)
        random_result = optimize(benchmark, RandomSampler(), n_trials)
        tpe_result = optimize(benchmark, TPESampler(), n_trials)

        # Compare final regret
        results.append({
            'benchmark': benchmark.name,
            'optiformer': optiformer_result.best_value,
            'random': random_result.best_value,
            'tpe': tpe_result.best_value,
        })

# Compute win rates
win_rate_vs_random = (optiformer < random).mean()
win_rate_vs_tpe = (optiformer < tpe).mean()
```

### Real-World ML Benchmarks

Test on actual hyperparameter tuning tasks:

**MNIST MLP:**
```python
search_space = {
    'hidden_size': (32, 512),      # Integer
    'num_layers': (1, 4),          # Integer
    'learning_rate': (1e-5, 1e-1), # Log-scale
    'dropout': (0.0, 0.5),         # Continuous
    'batch_size': (32, 256),       # Integer
    'optimizer': ['adam', 'sgd'],  # Categorical
}
```

### Metrics

1. **Win rate**: % of runs where OptiFormer beats baseline
2. **Final regret**: Best value found - known optimum
3. **Convergence speed**: Trials to reach threshold
4. **Statistical significance**: Use multiple seeds, report confidence intervals

### Key Files

- `evaluation/synthetic_benchmarks/functions.py`: Test functions
- `evaluation/ml_benchmarks/`: Real ML tasks
- `evaluation/evaluate.py`: Evaluation runner

---

## 10. Key Design Decisions

### Decision 1: Tokenization Precision (1000 bins)

**Options considered:**
- 100 bins: Too coarse (1% error)
- 1000 bins: Good balance (0.1% error)
- 10000 bins: Vocabulary too large

**Chosen:** 1000 bins — sufficient precision for HPO without vocabulary explosion.

### Decision 2: Teacher Algorithm (TPE)

**Options considered:**
- Random search: Too weak, model learns nothing useful
- TPE: Good balance of quality and speed
- GP-BO: Too slow for generating millions of trajectories
- Ensemble: Best quality but complex

**Chosen:** TPE — fast, effective, widely used baseline.

**Extension:** Train on multiple teachers for diversity.

### Decision 3: Model Architecture (Decoder-only)

**Options considered:**
- Encoder-only (BERT-style): Doesn't naturally generate sequences
- Encoder-decoder: Overkill for this task
- Decoder-only (GPT-style): Perfect fit for sequential generation

**Chosen:** Decoder-only — matches the causal nature of optimization.

### Decision 4: Positional Encoding (RoPE)

**Options considered:**
- Absolute (sinusoidal): Poor length generalization
- Learned absolute: Same problem
- RoPE: Good length generalization
- ALiBi: Also good, but less tested

**Chosen:** RoPE — comes with Llama, proven effective.

### Decision 5: Target Regret Conditioning

**The idea:** Add a token indicating desired outcome quality.

**Why it helps:**
- Training data includes suboptimal trajectories
- Without conditioning, model might suggest average (not optimal) values
- Conditioning on "optimal" biases toward good suggestions

---

## 11. What Makes This Publishable

### Novelty Claims (verify against literature)

1. **Algorithm distillation for HPO**: Learning to optimize from optimization traces
2. **Tokenized representation**: Converting continuous HPO to discrete sequence modeling
3. **Target regret conditioning**: Novel conditioning mechanism for quality control

### Required Experiments for Publication

1. **Baselines**: Compare against:
   - Random search
   - TPE (Optuna default)
   - GP-BO (scikit-optimize)
   - SMAC3
   - BOHB/Hyperband

2. **Benchmarks**:
   - Synthetic (sphere, rastrigin, etc.)
   - HPOBench (real tabular data)
   - YAHPO-Gym (diverse scenarios)
   - Real ML tasks (MNIST, CIFAR, etc.)

3. **Analysis**:
   - Ablation studies (model size, data size, tokenization bins)
   - Transfer learning (train on synthetic, test on real)
   - Computational cost comparison

4. **Statistical rigor**:
   - Multiple seeds (10-30)
   - Confidence intervals
   - Statistical tests (Wilcoxon signed-rank)

### Related Work to Cite

- OptFormer (Chen et al., 2022) — Google's similar approach
- ABLATION (Anonymous, 2021) — Learned optimizers
- PFN4BO — Prior-fitted networks
- TPE (Bergstra et al., 2011) — The teacher algorithm
- Transformers (Vaswani et al., 2017) — Architecture foundation

---

## 12. Extending the Work

### Improvement Ideas

1. **Multi-teacher distillation**: Train on TPE + CMA-ES + SMAC trajectories
2. **Real-world data mixing**: Add HPOBench/YAHPO trajectories to training
3. **Uncertainty quantification**: Predict distribution over next value, not just point
4. **Multi-fidelity**: Handle early stopping / learning curve prediction
5. **Constrained optimization**: Handle parameter constraints
6. **Conditional generation**: Condition on task metadata (dataset size, model type)

### Scaling Up

1. **More data**: Generate 100K+ trajectories
2. **Larger model**: Use base (50M) or large (150M) configuration
3. **Longer training**: 100K+ steps with proper validation
4. **Distributed training**: Multi-GPU for larger batches

### Alternative Architectures

1. **State-space models (Mamba)**: Faster inference, competitive quality
2. **Mixture of experts**: Scale parameters without scaling compute
3. **Retrieval augmentation**: Retrieve similar past trajectories

---

## 13. References

### Core Papers

1. Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
2. Bergstra et al. (2011). "Algorithms for Hyper-Parameter Optimization." NeurIPS.
3. Chen et al. (2022). "Towards Learning Universal Hyperparameter Optimizers with Transformers." NeurIPS.
4. Snoek et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." NeurIPS.

### Implementation References

5. HuggingFace Transformers documentation
6. Optuna documentation
7. PyTorch AMP documentation

### Benchmark References

8. HPOBench (Eggensperger et al., 2021)
9. YAHPO-Gym (Pfisterer et al., 2022)

---

## Appendix A: File Structure Reference

```
optiformer/
├── model/
│   ├── config.py          # Model configurations (nano/small/base/large)
│   ├── optiformer.py      # Model wrapper (save/load/inference)
│   └── generation.py      # Inference utilities
│
├── data/
│   ├── tokenizer/
│   │   ├── numerical.py   # Float → token conversion
│   │   ├── categorical.py # String → token conversion
│   │   ├── vocabulary.py  # Token ID management
│   │   └── sequence.py    # Full trajectory encoding
│   │
│   ├── generators/
│   │   ├── gp_prior.py    # GP function sampling
│   │   ├── symbolic.py    # Expression tree generation
│   │   └── trajectory.py  # TPE optimization runner
│   │
│   └── datasets/
│       └── synthetic.py   # PyTorch Dataset implementation
│
├── training/
│   └── trainer.py         # Training loop
│
├── samplers/
│   └── optiformer_sampler.py  # Optuna integration
│
├── evaluation/
│   ├── synthetic_benchmarks/
│   │   └── functions.py   # Sphere, Rastrigin, etc.
│   └── ml_benchmarks/
│       ├── mnist_mlp.py   # MNIST hyperparameter tuning
│       └── ...
│
└── smoke_test/
    ├── run_all.py         # Master test script
    ├── phase1_tokenizer.py
    ├── phase2_data.py
    ├── phase3_training.py
    └── ...
```

---

## Appendix B: Quick Start Checklist

### To replicate the smoke test:

- [ ] Understand tokenization (Section 4)
- [ ] Understand data generation (Section 5)
- [ ] Understand model architecture (Section 6)
- [ ] Run `python -m smoke_test.run_all --quick`
- [ ] Verify all phases pass

### To train a real model:

- [ ] Generate large dataset (10K+ trajectories)
- [ ] Train base model for 50K+ steps
- [ ] Evaluate on synthetic benchmarks
- [ ] Evaluate on real ML tasks

### To write a paper:

- [ ] Verify novelty against OptFormer, ABLATION, PFN4BO
- [ ] Run comprehensive benchmarks (HPOBench, YAHPO)
- [ ] Ablation studies (model size, data size, teachers)
- [ ] Statistical analysis (multiple seeds, confidence intervals)

---

*Document generated for OptiFormer research project. Last updated: January 2026.*
