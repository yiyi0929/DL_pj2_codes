"""Train the Task-1 CIFAR-10 network.

Example:
    python train.py --epochs 100 --optimizer adamw --base-channels 48 \
        --activation gelu --label-smoothing 0.1 --dropout 0.15 --amp
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from models import build_model, count_parameters

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 for task 1.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./outputs/task1_best")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "leaky_relu", "elu", "gelu", "silu"])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--no-bn", action="store_true", help="Disable BatchNorm for ablation.")
    parser.add_argument("--no-residual", action="store_true", help="Disable residual connections for ablation.")

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")

    parser.add_argument("--subset-train", type=int, default=0, help="Use only N training images for quick debug. 0 means all.")
    parser.add_argument("--subset-test", type=int, default=0, help="Use only N test images for quick debug. 0 means all.")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return torch.device(name)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=test_transform
    )

    if args.subset_train and args.subset_train > 0:
        train_set = Subset(train_set, list(range(min(args.subset_train, len(train_set)))))
    if args.subset_test and args.subset_test > 0:
        test_set = Subset(test_set, list(range(min(args.subset_test, len(test_set)))))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader



class CustomSGD:
    """手写支持动量和权重衰减的 SGD 优化器 (满足要求 5-c)"""
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=5e-4):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # 为每个参数初始化动量状态缓存
        self.velocities = [torch.zeros_like(p.data) for p in self.params]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.velocities):
            if p.grad is None:
                continue
            grad = p.grad.data
            
            # 1. 权重衰减 (L2 正则化)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
                
            # 2. 动量更新: v = momentum * v + grad
            if self.momentum != 0:
                v.mul_(self.momentum).add_(grad)
                grad = v
                
            # 3. 梯度下降更新参数
            p.data.add_(grad, alpha=-self.lr)


def make_optimizer(args: argparse.Namespace, model: nn.Module) -> optim.Optimizer:
    if args.lr is not None:
        lr = args.lr
    else:
        lr = 0.1 if args.optimizer == "sgd" else 1e-3

    if args.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    raise ValueError(args.optimizer)


def make_scheduler(args: argparse.Namespace, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler | None:
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "step":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[max(1, int(args.epochs * 0.5)), max(2, int(args.epochs * 0.75))],
            gamma=0.1,
        )
    raise ValueError(args.scheduler)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct, targets.numel()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    use_amp = scaler is not None and scaler.is_enabled()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        correct, seen = accuracy_from_logits(logits.detach(), targets)
        total_correct += correct
        total_seen += seen

    return total_loss / total_seen, 100.0 * total_correct / total_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    with_confusion: bool = False,
) -> tuple[float, float, torch.Tensor | None]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64) if with_confusion else None

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        correct, seen = accuracy_from_logits(logits, targets)
        total_correct += correct
        total_seen += seen

        if confusion is not None:
            preds = logits.argmax(dim=1).cpu()
            t_cpu = targets.cpu()
            for t, p in zip(t_cpu, preds):
                confusion[t.long(), p.long()] += 1

    return total_loss / total_seen, 100.0 * total_correct / total_seen, confusion


def save_history_csv(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["test_loss"] for row in history], label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CIFAR-10 training and test loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.with_name("loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_acc"] for row in history], label="train accuracy")
    plt.plot(epochs, [row["test_acc"] for row in history], label="test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CIFAR-10 training and test accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.with_name("accuracy_curve.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(confusion: torch.Tensor, path: Path) -> None:
    matrix = confusion.float()
    row_sum = matrix.sum(dim=1, keepdim=True).clamp(min=1)
    matrix = matrix / row_sum

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix.numpy(), interpolation="nearest")
    plt.title("Normalized confusion matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = list(range(len(CIFAR10_CLASSES)))
    plt.xticks(ticks, CIFAR10_CLASSES, rotation=45, ha="right")
    plt.yticks(ticks, CIFAR10_CLASSES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_first_layer_filters(model: nn.Module, path: Path, max_filters: int = 32) -> None:
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.weight.shape[1] == 3:
            first_conv = module
            break
    if first_conv is None:
        return

    weights = first_conv.weight.detach().cpu()[:max_filters]
    n_filters = weights.shape[0]
    cols = int(math.ceil(math.sqrt(n_filters)))
    rows = int(math.ceil(n_filters / cols))

    plt.figure(figsize=(cols * 1.4, rows * 1.4))
    for i in range(n_filters):
        filt = weights[i]
        filt = filt.permute(1, 2, 0)
        filt = filt - filt.min()
        denom = filt.max().clamp(min=1e-6)
        filt = filt / denom
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(filt.numpy())
        ax.axis("off")
    plt.suptitle("First-layer convolution filters")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = make_loaders(args)
    model = build_model(
        base_channels=args.base_channels,
        activation=args.activation,
        dropout=args.dropout,
        use_bn=not args.no_bn,
        use_residual=not args.no_residual,
    ).to(device)
    params = count_parameters(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_acc = 0.0
    best_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0)

    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["parameters"] = params
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Device: {device}")
    print(f"Trainable parameters: {params:,}")
    print(f"Output directory: {out_dir}")

    history: list[dict[str, float]] = []
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_error": 100.0 - test_acc,
            "lr": lr,
            "epoch_seconds": elapsed,
        }
        history.append(row)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "args": config,
                },
                out_dir / "best_model.pth",
            )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "args": config,
            },
            out_dir / "last_model.pth",
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc:.2f}% | "
            f"best {best_acc:.2f}% @ {best_epoch} | "
            f"lr {lr:.6g} | {elapsed:.1f}s"
        )

    total_seconds = time.time() - start_time
    save_history_csv(history, out_dir / "history.csv")
    plot_history(history, out_dir / "history.csv")

    best_path = out_dir / "best_model.pth"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model"])
    best_test_loss, best_test_acc, confusion = evaluate(
        model, test_loader, criterion, device, with_confusion=True
    )
    if confusion is not None:
        plot_confusion_matrix(confusion, out_dir / "confusion_matrix.png")
    plot_first_layer_filters(model, out_dir / "first_layer_filters.png")

    summary = {
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_acc,
        "best_test_error": 100.0 - best_test_acc,
        "best_test_loss": best_test_loss,
        "parameters": params,
        "total_seconds": total_seconds,
        "seconds_per_epoch_mean": total_seconds / max(1, len(history)),
        "model_path": str(best_path),
        "history_path": str(out_dir / "history.csv"),
        "loss_curve_path": str(out_dir / "loss_curve.png"),
        "accuracy_curve_path": str(out_dir / "accuracy_curve.png"),
        "confusion_matrix_path": str(out_dir / "confusion_matrix.png"),
        "filters_path": str(out_dir / "first_layer_filters.png"),
        "config": config,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Training finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

def plot_loss_landscape_1d(model, loader, criterion, device, best_state_dict, path):
    """在随机初始化和最优权重之间做线性插值，绘制 1D Loss Landscape"""
    # 1. 创建一个随机初始化的模型作为起点，best 模型作为终点
    init_model = build_model(base_channels=model.features[0].block[0].out_channels).to(device) # 简单实例化
    
    alphas = [i / 20.0 for i in range(-5, 26)] # alpha 从 -0.25 到 1.25
    losses = []
    
    init_p = {k: v.clone() for k, v in init_model.state_dict().items()}
    best_p = {k: v.clone() for k, v in best_state_dict.items()}
    
    model.eval()
    images, targets = next(iter(loader)) # 只取一个 batch 加快速度
    images, targets = images.to(device), targets.to(device)
    
    with torch.no_grad():
        for alpha in alphas:
            # 插值权重: P = (1 - alpha) * P_init + alpha * P_best
            new_state = {}
            for k in best_p.keys():
                if k in init_p and init_p[k].shape == best_p[k].shape:
                    new_state[k] = (1 - alpha) * init_p[k] + alpha * best_p[k]
                else:
                    new_state[k] = best_p[k]
            model.load_state_dict(new_state)
            logits = model(images)
            loss = criterion(logits, targets)
            losses.append(loss.item())
            
    plt.figure(figsize=(6, 4))
    plt.plot(alphas, losses, '-o', color='purple')
    plt.axvline(x=0.0, color='gray', linestyle='--', label='Initial Point')
    plt.axvline(x=1.0, color='green', linestyle='--', label='Trained Best Point')
    plt.xlabel('Interpolation Coefficient (alpha)')
    plt.ylabel('Loss')
    plt.title('1D Loss Landscape Trajectory')
    plt.legend()
    plt.savefig(path, dpi=200)
    plt.close()
