# SynCoGen Gin Configuration System

This directory contains the gin configuration system for SynCoGen. Gin-config provides a lightweight configuration framework for Python, allowing you to configure your experiments through simple text files.

## Directory Structure

```
configs/
├── README.md                    # This file
├── base.gin                     # Base configuration with defaults
├── noise/                       # Noise schedule configurations
│   ├── linear.gin
│   ├── cosine.gin
│   └── geometric.gin
├── loss/                        # Loss function configurations
│   ├── mse.gin
│   └── bond_length.gin
└── experiments/                 # Complete experiment configurations
    ├── default.gin
    ├── cosine_bond_loss.gin
    └── overfit.gin
```

## Quick Start

### Basic Usage

To use a gin configuration in your code:

```python
import gin

# Parse a configuration file
gin.parse_config_file('syncogen/configs/experiments/default.gin')

# Now instantiate classes - they'll use values from the config
noise_schedule = Linear()  # Uses sigma_min and sigma_max from config
loss = WeightedMSELoss()   # Uses mse_coef and square_time_weight from config
```

### Command Line Usage

You can also load configs from the command line:

```bash
python train.py --gin_file=syncogen/configs/experiments/default.gin
```

Or override specific parameters:

```bash
python train.py \
    --gin_file=syncogen/configs/experiments/default.gin \
    --gin_param="Linear.sigma_max=20.0" \
    --gin_param="TRAIN_BATCH_SIZE=64"
```

## Configuration Files

### Base Configuration (`base.gin`)

The base configuration contains common settings shared across all experiments:
- Data paths and dataset parameters
- Graph configuration (padding sizes)
- Default dataloader settings
- File reader settings

**Important**: Most experiment configs include the base config, so changes to `base.gin` affect all experiments.

### Noise Schedule Configurations

Located in `configs/noise/`, these configure different noise schedules for diffusion:

- **`linear.gin`**: Linear noise schedule
  - `sigma_min`: Minimum noise level (default: 0.0)
  - `sigma_max`: Maximum noise level (default: 10.0)

- **`cosine.gin`**: Cosine noise schedule
  - `eps`: Small epsilon for numerical stability (default: 1e-3)

- **`geometric.gin`**: Geometric noise schedule
  - `sigma_min`: Minimum noise level (default: 1e-3)
  - `sigma_max`: Maximum noise level (default: 1.0)

### Loss Configurations

Located in `configs/loss/`, these configure different loss functions:

- **`mse.gin`**: MSE losses for coordinate diffusion
  - `WeightedMSELoss.mse_coef`: Coefficient for MSE loss (default: 1.0)
  - `WeightedMSELoss.square_time_weight`: Whether to square time weights (default: False)

- **`bond_length.gin`**: Bond length preservation loss
  - `BondLengthLoss.time_threshold`: Time threshold for applying loss (default: 1.0)
  - `BondLengthLoss.square_time_weight`: Whether to square time weights (default: False)

### Experiment Configurations

Located in `configs/experiments/`, these are complete configurations for specific experiments:

#### `default.gin`
Basic baseline experiment with:
- Linear noise schedule
- Weighted MSE loss
- Standard batch sizes (32 train, 64 validation)

#### `cosine_bond_loss.gin`
Advanced experiment with:
- Cosine noise schedule
- Weighted MSE loss with squared time weights
- Bond length preservation loss (threshold at t=0.5)

#### `overfit.gin`
Debugging/testing configuration:
- Trains on only 10 examples
- Small batch sizes (4)
- Reduced noise range for faster convergence

## Configurable Components

The following components are configured with `@gin.configurable`:

### Noise Schedules
- `Linear` - Linear noise schedule
- `CosineNoise` - Cosine noise schedule
- `CosineSqrNoise` - Squared cosine noise schedule
- `GeometricNoise` - Geometric noise schedule
- `BrownianBridgeNoise` - Brownian bridge noise schedule
- `LogLinearNoise` - Log-linear noise schedule

### Loss Functions
- `MSELoss` - Mean squared error loss
- `WeightedMSELoss` - Time-weighted MSE loss
- `NodeNLLLoss` - Node negative log-likelihood loss
- `EdgeNLLLoss` - Edge negative log-likelihood loss
- `BondLengthLoss` - Bond length preservation loss
- `PairwiseDistanceLoss` - Pairwise distance loss
- `SmoothLDDTLoss` - Smooth LDDT-based loss

### Data Components
- `get_graph_data_list()` - Function to load and split graph data
- `GraphDataset` - PyTorch Geometric dataset for graphs
- `get_coordinates()` - Function to load molecular coordinates

### Graph Components
- `BBRxnGraph` - Building block and reaction graph (single)
- `BatchedBBRxnGraph` - Batched building block and reaction graph

### Molecule Components
- `SyncogenMolecule` - Single molecule representation
- `SyncogenMoleculeBatch` - Batched molecule representation

## Creating Custom Configurations

### Creating a New Experiment Config

1. Create a new file in `configs/experiments/`:

```gin
# my_experiment.gin

# Include base settings and component configs
include "syncogen/configs/base.gin"
include "syncogen/configs/noise/linear.gin"
include "syncogen/configs/loss/mse.gin"

# Override specific parameters
Linear.sigma_min = 0.5
Linear.sigma_max = 15.0

TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 128

WeightedMSELoss.mse_coef = 2.0
WeightedMSELoss.square_time_weight = True
```

2. Use it in your code:

```python
gin.parse_config_file('syncogen/configs/experiments/my_experiment.gin')
```

### Overriding Parameters Programmatically

You can override parameters without modifying config files:

```python
import gin

# Load base config
gin.parse_config_file('syncogen/configs/experiments/default.gin')

# Override specific parameters
gin.bind_parameter('Linear.sigma_max', 15.0)
gin.bind_parameter('TRAIN_BATCH_SIZE', 64)

# Now instantiate with overridden values
noise = Linear()  # Will use sigma_max=15.0
```

### Creating Configurable Classes

To make your own class configurable:

```python
import gin

@gin.configurable
class MyModel:
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
```

Then in your config file:

```gin
MyModel.hidden_dim = 512
MyModel.dropout = 0.2
```

## Best Practices

1. **Use includes**: Build on top of `base.gin` and component configs rather than duplicating settings

2. **Use meaningful names**: Name your experiment configs descriptively (e.g., `cosine_bond_loss.gin` instead of `exp1.gin`)

3. **Document parameters**: Add comments to explain non-obvious parameter choices

4. **Keep base.gin stable**: Put experiment-specific settings in experiment configs, not base.gin

5. **Version control**: Commit all config files with your code for reproducibility

6. **Use constants**: Define constants at the top of base.gin (like `DATA_ROOT`, `TRAIN_BATCH_SIZE`) and reuse them

## Parameter Scoping

Gin uses the class/function name to scope parameters. For example:

```gin
# These are different parameters
Linear.sigma_max = 10.0           # For Linear noise schedule
GeometricNoise.sigma_max = 1.0    # For GeometricNoise schedule
```

When you instantiate a class, gin uses the parameter with the matching scope:

```python
linear = Linear()           # Uses Linear.sigma_max = 10.0
geometric = GeometricNoise() # Uses GeometricNoise.sigma_max = 1.0
```

## Advanced Features

### Multiple Includes

You can include multiple config files:

```gin
include "syncogen/configs/base.gin"
include "syncogen/configs/noise/cosine.gin"
include "syncogen/configs/loss/mse.gin"
include "syncogen/configs/loss/bond_length.gin"
```

Later includes override earlier ones if they set the same parameter.

### Configurable Factories

You can configure which class to use:

```gin
# In config
noise_schedule = @Linear
Linear.sigma_max = 10.0
```

```python
# In code
@gin.configurable
def train(noise_schedule):
    scheduler = noise_schedule()  # Will instantiate Linear
```

### Constants and References

Define constants and reference them:

```gin
SIGMA_MAX = 10.0
Linear.sigma_max = %SIGMA_MAX
```

## Debugging

To see what parameters are currently configured:

```python
import gin

# After parsing configs
print(gin.config_str())  # Prints all active configuration
```

To check a specific parameter:

```python
# Check if a parameter is configured
configured = gin.query_parameter('Linear.sigma_max')
print(f"Linear.sigma_max = {configured}")
```

## Common Issues

### Issue: "No parameter set for..."

**Cause**: You're trying to instantiate a class without configuring its parameters.

**Solution**: Make sure you've parsed a config file that sets the parameter, or provide it explicitly:
```python
noise = Linear(sigma_max=10.0)  # Explicit
# OR
gin.parse_config_file('configs/noise/linear.gin')  # Via config
noise = Linear()
```

### Issue: "Multiple values for parameter..."

**Cause**: The same parameter is set multiple times in different config files.

**Solution**: Check your includes and remove duplicate settings, or order includes so the desired value comes last.

### Issue: Config changes don't take effect

**Cause**: You might be parsing configs after instantiating objects.

**Solution**: Always parse config files before instantiating configured objects:
```python
# CORRECT
gin.parse_config_file('config.gin')
model = MyModel()

# INCORRECT
model = MyModel()
gin.parse_config_file('config.gin')  # Too late!
```

## References

- [Gin-config GitHub](https://github.com/google/gin-config)
- [Gin-config Documentation](https://gin-config.readthedocs.io/)

## Examples

See `experiments/` directory for complete working examples of gin configurations.

