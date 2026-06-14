# Task 1: Train a Network on CIFAR-10

## 1. Objective

The goal of this task is to train a neural network for CIFAR-10 image classification and report the best test error, model structure, parameter count, training speed, and visual analysis.

## 2. Dataset and preprocessing

I used CIFAR-10, which contains 60,000 RGB images of size 32 x 32 in 10 classes. The training set has 50,000 images and the test set has 10,000 images.

Training transforms:

- Random crop with padding 4.
- Random horizontal flip.
- Tensor conversion.
- Normalization with CIFAR-10 channel mean and standard deviation.

Test transforms:

- Tensor conversion.
- Normalization with the same channel statistics.

Dataset link: [paste your CIFAR-10 link here]

## 3. Network architecture

I implemented a custom CNN named `CIFARConvNet`. It contains all required components:

| Requirement | Implementation |
|---|---|
| 2D convolution | `nn.Conv2d` in the stem and residual blocks |
| 2D pooling | `nn.MaxPool2d` and `nn.AdaptiveAvgPool2d` |
| Activation | ReLU, LeakyReLU, ELU, GELU, or SiLU |
| Fully connected layer | `nn.Linear` classifier |
| Extra components | BatchNorm, Dropout, and residual connections |

The final selected configuration was:

| Item | Value |
|---|---|
| Base channels | 48 |
| Activation | GELU |
| BatchNorm | Yes |
| Dropout | 0.15 |
| Residual connections | Yes |
| Optimizer | AdamW |
| Weight decay | 5e-4 |
| Label smoothing | 0.10 |
| Scheduler | Cosine annealing |
| Epochs | [fill in] |
| Trainable parameters | [fill in from summary.json] |

The model first extracts local visual features using convolutional blocks, reduces spatial size using pooling, increases the number of channels from 48 to 96 and 192, then uses global average pooling and a fully connected layer for classification.

## 4. Training setup

The loss function was cross entropy. I also tested label smoothing as a regularized version of cross entropy. The optimizer was selected from `torch.optim`. The final setting used AdamW because it gave stable training and good validation performance in my experiments.

Training command:

```bash
python train.py --epochs 100 --optimizer adamw --base-channels 48 --activation gelu --label-smoothing 0.1 --dropout 0.15 --weight-decay 5e-4 --amp --out-dir outputs/task1_best
```

## 5. Main result

| Metric | Result |
|---|---:|
| Best test accuracy | [fill in from summary.json] |
| Best test error | [fill in from summary.json] |
| Best epoch | [fill in from summary.json] |
| Number of trainable parameters | [fill in from summary.json] |
| Average seconds per epoch | [fill in from summary.json] |
| Trained weights link | [paste Google Drive or netdisk link] |
| GitHub code link | [paste GitHub link] |

## 6. Ablation and comparison experiments

I compared several design choices to satisfy the optimization requirements.

| Experiment | Filters | Activation | Optimizer | Loss/regularization | Best test error |
|---|---:|---|---|---|---:|
| A | 32 | ReLU | AdamW | CE, weight decay | [fill in] |
| B | 48 | ReLU | AdamW | CE, weight decay | [fill in] |
| C | 48 | GELU | AdamW | CE + label smoothing, weight decay, dropout | [fill in] |
| D | 48 | LeakyReLU | AdamW | CE + label smoothing, weight decay, dropout | [fill in] |
| E | 48 | GELU | SGD momentum | CE + label smoothing, weight decay, dropout | [fill in] |

Discussion:

- Increasing the number of base filters from 32 to 48 improved the representational capacity of the network, but also increased the parameter count and training time.
- GELU/LeakyReLU were compared with ReLU to test whether smoother or non-zero-negative activations improve optimization.
- Label smoothing and dropout acted as regularization, which reduced overfitting when the training accuracy became much higher than the test accuracy.
- AdamW was compared with SGD momentum. AdamW usually converges faster in early epochs, while SGD can become competitive with a well-tuned learning rate schedule.

## 7. Visualization and model insight

The training script generated the following figures:

1. `loss_curve.png`: training and test loss over epochs.
2. `accuracy_curve.png`: training and test accuracy over epochs.
3. `confusion_matrix.png`: per-class normalized confusion matrix.
4. `first_layer_filters.png`: visualization of learned first-layer convolution filters.
5. `experiment_test_error_comparison.png`: comparison of best test error across experiments.

The confusion matrix helps identify difficult class pairs. For example, animals such as cats and dogs may be confused more often than visually distinctive classes such as ships and airplanes. The first-layer filters show that the network learns local edge/color detectors, which are useful low-level features for CIFAR-10.

## 8. Conclusion

In Task 1, I implemented and trained a custom CNN with convolution, pooling, activation, fully connected layers, BatchNorm, Dropout, and residual connections. The best model achieved a test error of [fill in] on CIFAR-10. The ablation study showed that filter width, activation function, optimizer, and regularization all affected performance. The visualizations provided additional insight into training dynamics and model behavior.
