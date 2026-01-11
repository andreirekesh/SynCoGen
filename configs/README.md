# SynCoGen Gin Configuration System

SynCoGen uses [Gin-config](https://github.com/google/gin-config) for experiment configuration. Gin is a lightweight framework that allows you to configure Python classes and functions through text files, enabling reproducible experiments without code changes.

## Directory Structure

```
configs/
├── README.md                    # This file
├── experiments/                 # Complete experiment configurations
│   ├── synspace_original_cond.gin
│   ├── synspace_original_uncond.gin
│   └── overfit.gin
├── training/                    # Training and diffusion base configs
│   ├── base.gin                # Base configuration with shared defaults
│   ├── ema.gin                 # Exponential moving average config
│   └── trainer.gin             # PyTorch Lightning trainer config
├── model/                       # Model architecture configs
│   ├── semla.gin               # SEMLA backbone config
│   └── semla_pharm.gin         # SEMLA-Pharm backbone config
├── noise/                       # Noise schedule configurations
│   ├── linear.gin
│   ├── cosine.gin
│   ├── geometric.gin
│   ├── brownian.gin
│   └── loglinear.gin
├── loss/                        # Loss function configurations
│   ├── nll.gin                 # Negative log-likelihood (graph)
│   ├── mse.gin                 # Mean squared error (coordinates)
│   ├── bond_length.gin
│   ├── pairwise_distance.gin
│   └── smooth_lddt.gin
├── optim/                       # Optimizer configs
│   └── adamw.gin
├── scheduler/                   # Learning rate scheduler configs
│   ├── constant.gin
│   ├── linear.gin
│   ├── cosine.gin
│   └── warmup_cosine.gin
├── sampling/                    # Sampling strategy configs
│   ├── discrete_strategies/    # Graph sampling strategies
│   │   ├── mdlm.gin            # Masked Diffusion Language Model
│   │   └── p2.gin              # Path Planning
│   └── integrators/            # Coordinate integration schemes
│       └── euler.gin
└── logging/                     # Logging infrastructure
    ├── loggers/
    │   └── wandb.gin           # Weights & Biases logger
    ├── callbacks/
    │   └── checkpoint.gin     # Model checkpointing
    └── metrics/                # Evaluation metrics
        ├── default.gin
        ├── validity.gin
        ├── uniqueness.gin
        ├── novelty.gin
        ├── fragment_usage.gin
        └── energy.gin
```

## How Gin Works

### Basic Concepts

1. **Configurable Objects**: Classes and functions decorated with `@gin.configurable` can have their parameters set via `.gin` files.

2. **Parameter Scoping**: Parameters are scoped by class/function name:
   ```gin
   LinearNoise.sigma_max = 10.0        # For LinearNoise class
   MSELoss.coef = 1.0                  # For MSELoss class
   ```

3. **Includes**: Config files can include other configs using relative paths:
   ```gin
   include "configs/training/base.gin"
   include "configs/noise/linear.gin"
   ```

4. **Object References**: Use `@ClassName()` to reference configured objects:
   ```gin
   Diffusion.discrete_noise = @LinearNoise()
   Diffusion.losses = [@MSELoss(), @BondLengthLoss()]
   ```

### Usage

**From command line:**
```bash
python train.py --config configs/experiments/synspace_original_cond.gin --vocab_dir vocabulary/original
```

**Override parameters:**
```bash
python train.py --config configs/experiments/synspace_original_cond.gin --vocab_dir vocabulary/original \
    --gin "Optimizer.lr=1e-3" \
    --gin "Trainer.max_steps=50000"
```
