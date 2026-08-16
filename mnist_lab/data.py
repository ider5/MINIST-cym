"""数据管道：合成 toy 集（测试用）与 MNIST 加载、划分、增强。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torchvision import datasets, transforms

from mnist_lab.config import DEFAULT_SEED, IMAGE_SIZE, IN_CHANNELS, NUM_CLASSES


def mnist_transforms(*, augment: bool = False) -> transforms.Compose:
    """MNIST 默认已是白字黑底。增强只做小幅旋转/平移，便于观察过拟合。"""
    ops: list = []
    if augment:
        ops.extend(
            [
                transforms.RandomRotation(12),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            ]
        )
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def split_train_val(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> tuple[Subset, Subset]:
    """按比例把训练集再拆成 train / val。val_ratio=0 时验证集为空子集。"""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio 必须在 [0, 1) 内，收到 {val_ratio}")
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    if n_train <= 0:
        raise ValueError("划分后训练集为空，请减小 val_ratio 或增大数据量")
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return train_ds, val_ds


def take_subset(dataset: Dataset, n: Optional[int]) -> Dataset:
    """取前 n 条；n 为 None 时原样返回。教学实验用 subset 控制分钟级训练。"""
    if n is None:
        return dataset
    if n <= 0:
        raise ValueError("subset 必须为正整数")
    n = min(int(n), len(dataset))
    return Subset(dataset, list(range(n)))


def make_pattern_images(
    n: int,
    num_classes: int = NUM_CLASSES,
    image_size: int = IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    为每个数字构造可区分的条纹图案，让 toy 训练在几步内就能降 loss。
    不依赖 MNIST 下载。
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % num_classes
    x = torch.zeros(n, IN_CHANNELS, image_size, image_size)
    noise = torch.rand(n, IN_CHANNELS, image_size, image_size, generator=g) * 0.05
    for i in range(n):
        label = int(y[i])
        x[i, 0, label * 2 : label * 2 + 2, :] = 1.0
        x[i, 0, :, label * 2 : label * 2 + 2] = torch.clamp(
            x[i, 0, :, label * 2 : label * 2 + 2] + 0.6, max=1.0
        )
    x = torch.clamp(x + noise, 0.0, 1.0)
    return x, y


def make_toy_loaders(
    n: int = 64,
    batch_size: int = 16,
    num_classes: int = NUM_CLASSES,
    image_size: int = IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    val_ratio: float = 0.25,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """合成 28×28 数据。仅用于单元测试与无网络演示，绝不下载 MNIST。"""
    x, y = make_pattern_images(n, num_classes=num_classes, image_size=image_size, seed=seed)
    dataset = TensorDataset(x, y)
    n_test = max(1, n // 5)
    n_rest = n - n_test
    generator = torch.Generator().manual_seed(seed)
    rest, test_ds = random_split(dataset, [n_rest, n_test], generator=generator)
    train_ds, val_ds = split_train_val(rest, val_ratio=val_ratio, seed=seed)
    loader_kw = dict(batch_size=min(batch_size, n), shuffle=False)
    return (
        DataLoader(train_ds, **loader_kw),
        DataLoader(val_ds, **loader_kw),
        DataLoader(test_ds, **loader_kw),
    )


def make_mnist_loaders(
    batch_size: int = 50,
    val_ratio: float = 0.1,
    subset: Optional[int] = None,
    download: bool = True,
    root: str = "./data",
    augment: bool = False,
    seed: int = DEFAULT_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载 MNIST，从官方训练集再划出验证集。
    subset 只截断训练集（再划分 val），测试集始终用官方 test split（可同样截断以便快速课上演示）。
    """
    train_tf = mnist_transforms(augment=augment)
    test_tf = mnist_transforms(augment=False)
    train_full = datasets.MNIST(root=root, train=True, transform=train_tf, download=download)
    test_full = datasets.MNIST(root=root, train=False, transform=test_tf, download=download)
    train_full = take_subset(train_full, subset)
    test_data = take_subset(test_full, subset)
    train_ds, val_ds = split_train_val(train_full, val_ratio=val_ratio, seed=seed)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_data, batch_size=batch_size, shuffle=False),
    )


def load_digit_image(image_path: str | Path) -> torch.Tensor:
    """
    读入单张数字图 → [1, 1, 28, 28]。
    若平均亮度偏高（白底黑字），自动反色以贴近 MNIST 的白字黑底。
    """
    path = Path(image_path)
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if float(arr.mean()) > 0.5:
        arr = 1.0 - arr
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
