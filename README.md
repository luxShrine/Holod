# Holod

Holod (package name **`holod`**) provides a command-line interface for training, evaluating, and visualizing models that estimate object depth from digital holograms. The project includes utilities for dataset preparation, logging, and post-processing such as amplitude/phase reconstruction.

## Features

- **Autofocus model training**: Classification or regression backbones (`efficientnet`, `vit`, `resnet50`, `focusnet`, and others) built with PyTorch.
- **Visualization utilities**: Plot depth‐prediction performance and generate publication-ready figures.
- **Hologram reconstruction**: Reconstruct amplitude and phase images using a trained model’s predicted depth.
- **Model comparison**: Benchmark backbones under one shared config (`compare`) or rank trained checkpoints on a single hologram (`compare-holo`) using the label-free gradient‑Tamura focus score — the sharpness of the reconstruction at the predicted depth.
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
| Wavelength |  Laser wavelength in micrometers (e.g. 0.405 for a 405 nm laser)
| L_value | Distance from point source to sensor/screen (mm)
| z_value | Ground‑truth distance from point source to sample (mm); the DLHM magnification is M = L/z

If no CSV exists, create an `info.txt` file next to images containing:

```
Wavelength = 0.405
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

`--soft-sigma FLOAT` – std dev (in bins) for soft ordinal classification labels
(SORD); 0 keeps hard one-hot labels

`--fixed-seed/--no-fixed-seed` – deterministic training

`--continue` – resume from a checkpoint

`--create-csv` – create metadata CSV from `info.txt`

`--sample` – train on provided sample data

Example quick test:

```bash
holod train src/data/MW_Dataset_Sample --bins 10 --crop 256 --ep 8 --lr 1e-4
```

`compare`

Compare model backbones under the shared configuration from
`train_settings.toml`. Collects architecture statistics (parameter counts,
tensor size, input channels), inference speed (latency and throughput), and
trains each backbone with the standard pipeline, reporting its losses and best
validation metric (accuracy for classification, MAE for regression). Results
are printed as a table and saved to
`reports/backbone_comparison_<timestamp>.{json,csv}`. Each trained backbone
also writes the same per-run artifacts an individual `train` run produces:
best-model checkpoints in `src/checkpoints`, a per-epoch loss/metric history
in `reports/loss/`, and a `reports/plot_info_*.json` with its per-sample
predictions (usable with `plot-train`).

```bash
holod compare [--model {efficientnet|vit|resnet50|focusnet|new}]... \
              [--bins INT] [--crop INT] [--batch INT] [--ep INT] \
              [--device {cuda|cpu}] [--soft-sigma FLOAT] [--sample] \
              [--display {save|show|both}]
```

Repeat `--model` to select a subset (default: every backbone configured in
`train_settings.toml`). Shared settings merge with CLI options, like `train`;
per-model settings (learning rate, weight decay, scheduler) come from each
backbone's own `[<backbone>.train]` section. Example:

```bash
holod compare --model focusnet --model resnet50 --bins 10 --ep 5 --sample
```

`compare-holo`

Evaluate already-trained checkpoints against a single hologram image. Each
model predicts a depth (timed over `--runs` repetitions), and the prediction is
scored **label-free with the gradient-Tamura focus score** of the
reconstruction at that depth — higher is sharper — replacing the earlier
NRMSE-based scoring. When `--z-true` is given, models are instead ranked by
absolute depth error and the report adds signed/relative error and, for
classifiers, the error in bin widths. Results are printed as a table and saved
to `reports/hologram_comparison.{json,csv}` alongside a bar-chart plot.

```bash
holod compare-holo IMG_FILE_PATH \
                   [--model-path PATH]... [--runs 5] [--crop_size 224] \
                   [--wavelength 5.3e-07] [--dx 3.8e-06] \
                   [--z-true MM] [--l-value MM] \
                   [--device {cuda|cpu}] [--display {save|show|both}]
```

`--model-path` defaults to every `.pth`/`.tar` under `src/checkpoints`. Pass
`--l-value` (the dataset's `L_value`, mm) whenever possible: DLHM holograms
refocus at the plane-wave-equivalent depth `M*(L - z)`, so without it the
focus score is computed at the raw predicted depth and is uninformative.

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
                    [--wavelength 5.3e-07] [--dx 3.8e-06] \
                    [--display {show|save|both}] \
                    [--amp_true PATH] [--phase_true PATH]
```

If `IMG_FILE_PATH` is empty, the sample hologram is used. The command logs the gradient‑Tamura focus score of the reconstruction (higher is sharper) and optionally saves plots.

# # Configuration

Default training parameters reside in `train_settings.toml`. The config is
**multi-model**: settings shared by every backbone (dataset, batch size, crop,
epochs, device, ...) live at the top level, while each backbone gets its own
`[<backbone>.train]` section carrying its optimizer and scheduler settings.
Only backbones with a section can be selected with `train --model` or included
in `compare`; valid section names are `enet`, `vit`, `resnet`, `pcnn`, and
`focusnet`.

```toml
# shared by every model
batch_size  = 8
crop_size   = 224
device      = "cuda"
epoch_count = 15
num_classes = 20     # set to 1 for regression training
num_workers = 2
val_split   = 0.2

[paths]
dataset_root  = "MW_Dataset_Sample"
meta_csv_name = "ODP-DLHM-Database.csv"

[flags]
checkpoint = false
create_csv = false
fixed_seed = true
sample     = false

# one section per model; omitted settings fall back to defaults
[enet.train]
backbone                = "enet"
learning_rate           = 1e-4
optimizer_weight_decay  = 1e-2
sch_factor              = 0.1
sch_patience            = 7

[focusnet.train]
backbone      = "focusnet"
learning_rate = 1e-4
```

CLI options override the shared settings; `train --lr` overrides the selected
model's learning rate.

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
