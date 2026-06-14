"""Run a small experiment grid for Task 1.

This script covers the requested comparisons:
1. different numbers of filters,
2. different regularization/loss settings,
3. different activation functions,
4. different optimizers from torch.optim.

Example:
    python run_experiments.py --epochs 30 --device cuda --amp
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task-1 CIFAR-10 experiment grid.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./outputs/task1_experiments")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--subset-train", type=int, default=0)
    parser.add_argument("--subset-test", type=int, default=0)
    return parser.parse_args()


def experiment_definitions() -> list[dict[str, object]]:
    return [
        {
            "name": "A_base32_relu_adamw_ce_ls0",
            "base_channels": 32,
            "activation": "relu",
            "optimizer": "adamw",
            "label_smoothing": 0.0,
            "weight_decay": 5e-4,
            "dropout": 0.10,
            "note": "baseline; 32 base filters; ReLU; AdamW; CE without label smoothing",
        },
        {
            "name": "B_base48_relu_adamw_ce_ls0",
            "base_channels": 48,
            "activation": "relu",
            "optimizer": "adamw",
            "label_smoothing": 0.0,
            "weight_decay": 5e-4,
            "dropout": 0.10,
            "note": "more filters than baseline",
        },
        {
            "name": "C_base48_gelu_adamw_ce_ls01",
            "base_channels": 48,
            "activation": "gelu",
            "optimizer": "adamw",
            "label_smoothing": 0.10,
            "weight_decay": 5e-4,
            "dropout": 0.15,
            "note": "best-candidate setting; GELU plus label smoothing and dropout",
        },
        {
            "name": "D_base48_leakyrelu_adamw_ce_ls01",
            "base_channels": 48,
            "activation": "leaky_relu",
            "optimizer": "adamw",
            "label_smoothing": 0.10,
            "weight_decay": 5e-4,
            "dropout": 0.15,
            "note": "activation comparison against GELU/ReLU",
        },
        {
            "name": "E_base48_gelu_sgd_ce_ls01",
            "base_channels": 48,
            "activation": "gelu",
            "optimizer": "sgd",
            "label_smoothing": 0.10,
            "weight_decay": 5e-4,
            "dropout": 0.15,
            "note": "optimizer comparison using SGD momentum",
        },
    ]


def run_one(args: argparse.Namespace, exp: dict[str, object]) -> dict[str, object]:
    out_dir = Path(args.out_dir) / str(exp["name"])
    summary_path = out_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        summary["experiment"] = exp["name"]
        summary["note"] = exp["note"]
        return summary

    command = [
        sys.executable,
        "train.py",
        "--data-dir",
        args.data_dir,
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--base-channels",
        str(exp["base_channels"]),
        "--activation",
        str(exp["activation"]),
        "--optimizer",
        str(exp["optimizer"]),
        "--label-smoothing",
        str(exp["label_smoothing"]),
        "--weight-decay",
        str(exp["weight_decay"]),
        "--dropout",
        str(exp["dropout"]),
    ]
    if args.amp:
        command.append("--amp")
    if args.subset_train:
        command.extend(["--subset-train", str(args.subset_train)])
    if args.subset_test:
        command.extend(["--subset-test", str(args.subset_test)])

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    with summary_path.open() as f:
        summary = json.load(f)
    summary["experiment"] = exp["name"]
    summary["note"] = exp["note"]
    return summary


def write_summary(summaries: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "best_test_accuracy",
        "best_test_error",
        "best_epoch",
        "parameters",
        "seconds_per_epoch_mean",
        "model_path",
        "note",
    ]
    with (out_dir / "experiments_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fields})

    with (out_dir / "experiments_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    labels = [str(row["experiment"]) for row in summaries]
    errors = [float(row["best_test_error"]) for row in summaries]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, errors)
    plt.ylabel("Best test error (%)")
    plt.title("Task-1 experiment comparison on CIFAR-10")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "experiment_test_error_comparison.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    summaries: list[dict[str, object]] = []
    for exp in experiment_definitions():
        summaries.append(run_one(args, exp))
    write_summary(summaries, out_dir)
    print("All experiments finished.")
    print(f"Summary: {out_dir / 'experiments_summary.csv'}")
    print(f"Figure: {out_dir / 'experiment_test_error_comparison.png'}")


if __name__ == "__main__":
    main()
