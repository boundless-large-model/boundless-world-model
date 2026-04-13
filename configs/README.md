# Training Configurations

This directory contains YAML configuration files for training.

## Quick Start

```bash
# 1. Copy and edit a config
vim configs/train_template.yaml

# 2. Launch training
bash scripts/train.sh
```

## Configuration Markers

Template files use markers to indicate parameter importance:

| Marker | Meaning |
|--------|---------|
| `[REQUIRED]` | Must fill, training will fail if empty |
| `[KEY]` | Critical parameters affecting quality |
| `[TUNABLE]` | Hyperparameters to experiment with |
| `[OPTIONAL]` | Safe to keep as default |

