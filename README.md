# Holod

Holod (package name **`holod`**) provides a command-line interface for training, evaluating, and visualizing models that estimate object depth from digital holograms. The project includes utilities for dataset preparation, logging, and post-processing such as amplitude/phase reconstruction.

## Features

- **Autofocus model training**: Classification or regression backbones (`efficientnet`, `vit`, `resnet50`, `focusnet`, and others) built with PyTorch.
- **Visualization utilities**: Plot depth‐prediction performance and generate publication-ready figures.
- **Hologram reconstruction**: Reconstruct amplitude and phase images using a trained model’s predicted depth.
- **Rich logging**: JSON log file and colorized console output.
- **Dataset tooling**: Generate metadata CSV files from `info.txt` descriptions and validate images.

## Installation

### Using `uv` (recommended)

```bash
git clone https://github.com/luxShrine/Holod.git
cd Holod
uv sync          # install dependencies into a virtual environment
uv run holod --help
```

Using native pip

```bash
python3 -m venv .venv
source .venv/bin/activate       # PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
holod --help
```

Dataset Preparation

Training expects a CSV in the dataset root with columns:

|  | Description
| ------------- | -------------- |
| path  | Path to each image relative to the CSV file
| Wavelength |  Laser wavelength in millimeters
| L_value | Distance from point source to sample (mm)
| z_value | Ground‑truth object distance from the sensor (mm)

If no CSV exists, create an `info.txt` file next to images containing:

```
Wavelength = 500
L_value = 18.96
z_value = 0.521
```

Then run `holod train DS_ROOT --create-csv` to auto‑generate the CSV. A small sample dataset is provided under `src/data/MW_Dataset_Sample`.

## Command‑line Usage

General entry point:

```bash
holod [OPTIONS] COMMAND [ARGS]...
```

Global options

- `-v / --verbose` – show debug messages.

- `--log-file PATH` – JSON log file (defaults to logs/holo_log.jsonl).

- `--help` – show command help.

## Commands

`train`

Train a model on a dataset.

```bash
holod train [OPTIONS] DS_ROOT
```

Key options:

`--csv-name PATH` – metadata CSV (default ODP-DLHM-Database.csv)

`--bins INT` – number of classification bins (1 for regression)

`--model [efficientnet|vit|resnet50|new|focusnet]`

`--crop INT` – image crop size

`--split FLOAT` – validation split

`--batch INT` – batch size

`--ep INT` – epochs

`--lr FLOAT` – learning rate

`--device [cuda|cpu]`

`--fixed-seed/--no-fixed-seed` – deterministic training

`--continue` – resume from a checkpoint

`--create-csv` – create metadata CSV from `info.txt`

`--sample` – train on provided sample data

Example quick test:

```bash
holod train src/data/MW_Dataset_Sample --bins 10 --crop 256 --ep 8 --lr 1e-4
```

`plot-train`

Visualize training and validation metrics.

```bash
holod plot-train [--display {show|save|both|meta}] [--classfiication]
```

`reconstruction`

Reconstruct amplitude and phase from a hologram using a trained model.

```bash
holod reconstruction [IMG_FILE_PATH] \
                    [--model_path PATH] [--crop_size 512] \
                    [--wavelength 5.3e-07] [--z 0.02] [--dx 1e-06] \
                    [--display {show|save|both}] \
                    [--amp_true PATH] [--phase_true PATH]
```

If `IMG_FILE_PATH` is empty, the sample hologram is used. The command outputs PSNR and NRMSE metrics and optionally saves plots.

# # Configuration

Default training parameters reside in `train_settings.toml`:

```bash
[paths]
dataset_root   = "MW-Dataset"
meta_csv_name  = "ODP-DLHM-Database.csv"

[train]
backbone       = "Enet"
batch_size     = 16
crop_size      = 224
device         = "cuda"
epoch_count    = 15
learning_rate  = 0.0001
num_classes    = 10
num_workers    = 6
optimizer_weight_decay = 1e-2
sch_factor     = 0.1
sch_patience   = 7
val_split      = 0.1

[flags]
checkpoint = false
create_csv = false
fixed_seed = true
sample     = false
```

CLI options override these defaults.

## Development & Testing

Common tasks are provided via the `makefile`:

```bash
make requirements   # install deps via uv
make format         # ruff lint + format
make typecheck      # mypy
make test           # run pytest
make check          # requirements + typecheck + test
make train          # shortcut to run 'holod train'
make plot           # shortcut to run 'holod plot-train'
make recon          # example reconstruction
```

Tests expect images in `src/tests/images` and can be run with `pytest -q src/tests/check_training.py`.

## License

Distributed under the MIT License
