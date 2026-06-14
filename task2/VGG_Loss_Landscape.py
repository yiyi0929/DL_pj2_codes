import sys
import os

# 确保把 VGG_BatchNorm 加入 Python 搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import random
from tqdm import tqdm as tqdm
from IPython import display

# 从本地模块导入模型
from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm 
from data.loaders import get_cifar_loader

# =========================================================
# 1. 纯函数和常量定义（可以安全地放在外面，供子进程导入）
# =========================================================
device_id = 0 
num_workers = 0
batch_size = 128

# 路径配置
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

# 计算分类准确率
def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# 设置随机种子以保证实验可重复
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if 'cuda' in str(device): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 核心训练函数
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=20):
    model.to(device)
    learning_curve = [0.0] * epochs_n
    step_losses = []
    step_grads = []
    batches_n = len(train_loader)
    
    for epoch in range(epochs_n):
        if scheduler is not None:
            scheduler.step()
        model.train()

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_n}", leave=False)
        
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            
            step_losses.append(loss.item())
            running_loss += loss.item()

            if hasattr(model.classifier[4], 'weight') and model.classifier[4].weight.grad is not None:
                grad_norm = model.classifier[4].weight.grad.norm(2).item()
            else:
                grad_norm = 0.0
            step_grads.append(grad_norm)

            optimizer.step()

        learning_curve[epoch] = running_loss / batches_n
        val_acc = get_accuracy(model, val_loader)
        print(f"Epoch {epoch+1} ended -> Loss: {learning_curve[epoch]:.4f} | Val Acc: {val_acc:.4f}")

    return step_losses, step_grads

# 绘制并保存 Loss Landscape 填色对比图
def plot_loss_landscape(vgg_min, vgg_max, bn_min, bn_max):
    steps = np.arange(len(vgg_min))
    plt.figure(figsize=(11, 6))
    
    plt.plot(steps, vgg_min, color='crimson', alpha=0.4, linestyle='--')
    plt.plot(steps, vgg_max, color='crimson', alpha=0.4, linestyle='--')
    plt.fill_between(steps, vgg_min, vgg_max, color='crimson', alpha=0.15, label='Standard VGG-A (Without BN)')
    
    plt.plot(steps, bn_min, color='royalblue', alpha=0.4, linestyle='-')
    plt.plot(steps, bn_max, color='royalblue', alpha=0.4, linestyle='-')
    plt.fill_between(steps, bn_min, bn_max, color='royalblue', alpha=0.2, label='VGG-A with BatchNorm')
    
    plt.title('Loss Landscape: Impact of BatchNorm on Optimization Stability', fontsize=14, fontweight='bold')
    plt.xlabel('Steps (Batches over all Epochs)', fontsize=12)
    plt.ylabel('Loss Range (Max to Min across LRs)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    
    save_fig_path = os.path.join(figures_path, 'loss_landscape_comparison.png')
    plt.savefig(save_fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[Success] Final analysis image has been saved to: {save_fig_path}")


# =========================================================
# 2. 🔥 核心保护伞：所有“直接执行的代码”都塞进这里
# =========================================================
if __name__ == '__main__':
    print(f"Using device: {device}")

    # 载入 CIFAR-10 数据集
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)

    # 验证 Dataloader
    for X, y in train_loader:
        print(f" Successfully verified Dataloader. Batch shape: {X.shape}, labels: {y.shape}")
        break

    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    epo = 20  

    vgg_all_lr_losses = []
    vgg_bn_all_lr_losses = []

    # (1) 跑无 BN 的标准 VGG_A
    print("\n>>> Start training Standard VGG_A (Without BN)...")
    for lr in learning_rates:
        print(f"Training VGG_A with learning rate: {lr}")
        set_random_seeds(seed_value=2020, device=device)
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        losses, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
        vgg_all_lr_losses.append(losses)

    # (2) 跑带 BN 的 VGG_A_BatchNorm
    print("\n>>> Start training VGG_A_BatchNorm (With BN)...")
    for lr in learning_rates:
        print(f"Training VGG_A_BatchNorm with learning rate: {lr}")
        set_random_seeds(seed_value=2020, device=device)
        model_bn = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model_bn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        losses_bn, grads_bn = train(model_bn, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
        vgg_bn_all_lr_losses.append(losses_bn)

    # 转换为 numpy 矩阵
    vgg_all_lr_losses = np.array(vgg_all_lr_losses)
    vgg_bn_all_lr_losses = np.array(vgg_bn_all_lr_losses)

    # 计算极值
    vgg_min_curve = np.min(vgg_all_lr_losses, axis=0)
    vgg_max_curve = np.max(vgg_all_lr_losses, axis=0)
    vgg_bn_min_curve = np.min(vgg_bn_all_lr_losses, axis=0)
    vgg_bn_max_curve = np.max(vgg_bn_all_lr_losses, axis=0)

    # 保存数据
    np.savetxt(os.path.join(models_path, 'vgg_min_curve.txt'), vgg_min_curve)
    np.savetxt(os.path.join(models_path, 'vgg_max_curve.txt'), vgg_max_curve)

    # 执行绘图
    plot_loss_landscape(vgg_min_curve, vgg_max_curve, vgg_bn_min_curve, vgg_bn_max_curve)
