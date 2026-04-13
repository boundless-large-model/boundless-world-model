# Configurations

This directory contains training and model configuration files, organized by purpose.

## Directory Structure

```
configs/
├── train/          # Training configs (experiment hyperparameters)
│   ├── train_action_noise.yaml
│   ├── train_action_adaln.yaml
│   ├── train_noise_shortframe.yaml
│   ├── train_template_full.yaml
│   ├── train_template_base.yaml
│   └── ...
├── model/          # Model structure configs
│   ├── wan2_1_fun_1_3b_inp.yaml
│   ├── wan2_2_ti2v_5b.yaml
│   └── ...
└── README.md
```

## Quick Start

```bash
# Use training config
python scripts/train.py --config configs/train/train_action_noise.yaml
```

## Configuration Markers

Template files use markers to indicate parameter importance:

| Marker | Meaning |
|--------|---------|
| `[REQUIRED]` | Must fill, training will fail if empty |
| `[KEY]` | Critical parameters affecting quality |
| `[TUNABLE]` | Hyperparameters to experiment with |
| `[OPTIONAL]` | Safe to keep as default |