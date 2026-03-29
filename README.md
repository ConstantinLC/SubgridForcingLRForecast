# Subgrid Modelling Experiments

Machine learning models for subgrid parametrization and forecasting of quasi-geostrophic (QG) fluid dynamics. The project explores how to learn subgrid parametrizations that capture the influence of unresolved high-resolution (HR) features on coarse-grained low-resolution (LR) dynamics, combining data-driven forecasting with learned parametrizations.

## Overview

The core problem: simulating geophysical fluid dynamics at coarse resolution loses fine-scale turbulent structures that influence the large-scale dynamics. This project trains neural networks to:

1. **Forecast** low-resolution QG dynamics one or more timesteps ahead
2. **Learn subgrid parametrizations** that approximate the forcing effect of unresolved HR scales on LR dynamics
3. **Combine** forecasting and parametrization in joint training pipelines

Data comes from [PyQG](https://pyqg.readthedocs.io/) quasi-geostrophic simulations run at both low (64x64) and high (256x256) resolution.

## Project Structure

```
├── src/
│   ├── forecast/
│   │   ├── train.py                  # Forecasting training script
│   │   ├── module.py                 # ForecastModel Lightning module
│   │   ├── unet.py                   # UNet architecture
│   │   └── fno.py                    # Fourier Neural Operator (FNO)
│   ├── parametrization/
│   │   ├── train.py                  # Parametrization training script
│   │   ├── module.py                 # ParametrizationTeaching Lightning modules
│   │   └── subgrid_parametrization.py # Parametrization network architecture
│   ├── utils/
│   │   ├── utils_config.py           # YAML config parsing
│   │   ├── activations.py            # Activation function registry
│   │   └── generate_data.py          # Data generation utilities
│   ├── datamodule.py                 # PyTorch Lightning DataModule
│   ├── dataset.py                    # Dataset class definitions
│   └── dataset_highres.py            # High-res dataset implementation
├── configs/                          # YAML config files for experiments
├── checkpoints/
│   ├── forecast/                     # Saved forecasting model checkpoints
│   └── parametrization/              # Saved parametrization model checkpoints
└── pdearena/                         # PDEArena library (PDE surrogate learning)
```

## Experiment Configurations

| Config | Description |
|--------|-------------|
| `forecasting_raw.yaml` | Baseline LR forecasting |
| `forecasting_raw_ar.yaml` | Baseline with autoregressive rollout |
| `forecasting_with_forcing.yaml` | Forecasting with explicit forcing term |
| `forecasting_with_hr_encoding.yaml` | LR forecast conditioned on HR encoder output |
| `forecasting_with_hr_encoding_ar.yaml` | HR-encoding + autoregressive rollout |
| `hr_forecasting.yaml` | Direct high-resolution forecasting |
| `forecasting_with_parametrization_true_forcing.yaml` | Parametrization learned from true subgrid forcing |
| `forecasting_with_parametrization_learned_forcing.yaml` | Parametrization learned via HR-encoder teacher |
| `forecasting_with_pretrained_parametrization_and_forecast.yaml` | Combined pretrained models |

## Training

### Forecasting model

```bash
python src/forecast/train.py --config configs/forecasting_raw.yaml
```

### Parametrization model

```bash
python src/parametrization/train.py --config configs/forecasting_with_parametrization_true_forcing.yaml
```

Training uses PyTorch Lightning with:
- AdamW optimizer
- Early stopping (patience=10, monitors `val/MSE`)
- Best checkpoint saved to `checkpoints/forecast/` or `checkpoints/parametrization/`
- Experiment tracking via Weights & Biases

## Data

Expected data layout at `/mnt/SSD2/constantin/data/pyqg/`:

```
pyqg/
├── lowres/run_*.nc    # 64x64 coarse-grid simulations
└── highres/run_*.nc   # 256x256 fine-grid simulations
```

Each NetCDF file contains potential vorticity `q` (2 components) and optionally `q_forcing`. The 100 simulation runs are split 80/20 for training and validation.

## Key Config Parameters

```yaml
training:
  batch_size: 128
  max_epochs: 200

model:
  architecture: "UNet"   # or "FNO"
  learning_rate: 1e-3
  hidden_channels: 32
  n_input_scalar_components: 2
  n_output_scalar_components: 2

data:
  data_dir: "/mnt/SSD2/constantin/data/pyqg"
  lead_time: 1            # prediction horizon (hours)
  num_workers: 5

pipeline:
  add_parametrization: false
  add_highres_encoding: false
  add_forcing: false
  autoregressive: false
  noise_level: 0.0
  pretrained_forecast_path: null
  pretrained_parameterization_path: null
```

## Dependencies

- PyTorch + PyTorch Lightning
- xarray, h5py (NetCDF data loading)
- wandb (experiment tracking)
- numpy, scipy, matplotlib
- [PDEArena](https://microsoft.github.io/pdearena/) (included as submodule in `pdearena/`)
