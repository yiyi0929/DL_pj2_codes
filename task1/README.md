# Task 1: CIFAR-10 Classification

This folder is a complete PyTorch implementation for Project-2 Task 1.
It trains a custom CNN on CIFAR-10 and produces the artifacts needed for the report:
model weights, training curves, confusion matrix, first-layer filter visualization,
and a comparison table for multiple experiment settings.

## Assignment coverage

The model in `models.py` includes:

- Fully connected layer: `nn.Linear` in the classifier.
- 2D convolutional layers: `nn.Conv2d` in every block.
- 2D pooling layers: `nn.MaxPool2d` and `nn.AdaptiveAvgPool2d`.
- Activations: ReLU, LeakyReLU, ELU, GELU, or SiLU.
- Extra components: BatchNorm, Dropout, and residual connections.

The experiment grid in `run_experiments.py` covers:

- Different numbers of filters: `base_channels=32` vs `base_channels=48`.
- Different regularization/loss settings: cross entropy with and without label smoothing, plus weight decay and dropout.
- Different activations: ReLU, GELU, and LeakyReLU.
- Different optimizers from `torch.optim`: AdamW and SGD.
- Insightful visualizations: loss/accuracy curves, confusion matrix, first-layer filters, and experiment comparison plot.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Quick smoke test

Run a tiny subset first to check that everything works:

```bash
python train.py --epochs 2 --subset-train 1024 --subset-test 512 --num-workers 0 --out-dir outputs/smoke_test
```

## Train the recommended final model

For a normal GPU run:

```bash
python train.py --epochs 100 --optimizer adamw --base-channels 48 --activation gelu --label-smoothing 0.1 --dropout 0.15 --weight-decay 5e-4 --amp --out-dir outputs/task1_best
```

If you do not have a CUDA GPU, remove `--amp`. On CPU, reduce epochs or use the smoke-test subset.

The script writes:

- `outputs/task1_best/best_model.pth`
- `outputs/task1_best/last_model.pth`
- `outputs/task1_best/history.csv`
- `outputs/task1_best/summary.json`
- `outputs/task1_best/loss_curve.png`
- `outputs/task1_best/accuracy_curve.png`
- `outputs/task1_best/confusion_matrix.png`
- `outputs/task1_best/first_layer_filters.png`

## Run the comparison experiments

Use this command for a compact experiment grid:

```bash
python run_experiments.py --epochs 30 --device cuda --amp
```

For a CPU debug run:

```bash
python run_experiments.py --epochs 2 --device cpu --num-workers 0 --subset-train 1024 --subset-test 512
```

The experiment grid writes:

- `outputs/task1_experiments/experiments_summary.csv`
- `outputs/task1_experiments/experiments_summary.json`
- `outputs/task1_experiments/experiment_test_error_comparison.png`

## What to put in the final report

1. Best test error from `summary.json`.
2. The model architecture from `models.py`.
3. Trainable parameter count from `summary.json`.
4. Training speed from `summary.json`.
5. The comparison table from `experiments_summary.csv`.
6. Loss curve, accuracy curve, confusion matrix, and filter visualization.
7. Links to GitHub code, CIFAR-10 dataset, and uploaded `best_model.pth` weights.

## Suggested final GitHub structure

```text
your-repo/
  task1_cifar10/
    README.md
    requirements.txt
    models.py
    train.py
    run_experiments.py
    report_task1_template.md
```

Do not commit `data/`, `outputs/`, or large `.pth` files to GitHub unless your repository is configured for large files. Upload weights to Google Drive or another netdisk and paste the link in the PDF report.
