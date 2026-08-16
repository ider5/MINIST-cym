"""三种可切换模型：MLP、与旧 checkpoint 兼容的 CNN、带 Dropout 的 CNN。"""

from __future__ import annotations

import torch
import torch.nn as nn

from mnist_lab.config import IMAGE_SIZE, IN_CHANNELS, MODEL_NAMES, NUM_CLASSES


class MLP(nn.Module):
    """把 28×28 拉平后的两层全连接，用来对比「没有空间归纳偏置」时的表现。"""

    def __init__(self) -> None:
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    """
    与仓库最初 main.py 完全相同的结构，以便加载 cnn2.pkl：
    Conv(1→16, k=5, p=2) → ReLU → MaxPool2
    Conv(16→32, k=5, p=2) → ReLU → MaxPool2
    Linear(32*7*7 → 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32 * 7 * 7, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


class CNNDropout(nn.Module):
    """在两段卷积后插入 Dropout，用于过拟合对照实验。"""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.dropout1 = nn.Dropout(p)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.dropout2 = nn.Dropout(p)
        self.out = nn.Linear(32 * 7 * 7, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


def build_model(name: str, dropout_p: float = 0.5) -> nn.Module:
    """按名称构造模型。name ∈ {mlp, cnn, cnn_dropout}。"""
    key = name.strip().lower()
    if key == "mlp":
        return MLP()
    if key == "cnn":
        return CNN()
    if key == "cnn_dropout":
        return CNNDropout(p=dropout_p)
    raise ValueError(f"未知模型 {name!r}，可选：{MODEL_NAMES}")
