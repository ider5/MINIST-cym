"""课堂用的小份真实数字缓存，避免无网时只能看条纹 toy。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from mnist_lab.config import DEFAULT_SEED
from mnist_lab.data import split_train_val

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_PATH = REPO_ROOT / "fixtures" / "mnist_tiny.pt"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "cnn_cpu.pt"


def tiny_mnist_available(path: Path | None = None) -> bool:
    return Path(path or TINY_PATH).is_file()


def load_tiny_mnist(path: Path | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    p = Path(path or TINY_PATH)
    if not p.is_file():
        raise FileNotFoundError(
            f"找不到 {p}。请运行: python scripts/build_fixtures.py"
        )
    payload = torch.load(p, map_location="cpu")
    images = payload["images"]
    labels = payload["labels"]
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("fixture 中 images 应为 [N,1,28,28]")
    return images.float(), labels.long()


def make_classroom_loaders(
    batch_size: int = 32,
    subset: int | None = None,
    download: bool = True,
    root: str = "./data",
    toy: bool = False,
):
    """课堂优先真实数字：完整 MNIST → 仓库 fixture → 最后才是条纹 toy。"""
    from mnist_lab.data import make_mnist_loaders, make_toy_loaders

    if toy:
        n = max(subset or 128, 64)
        return make_toy_loaders(n=n, batch_size=batch_size), "toy"
    try:
        return (
            make_mnist_loaders(
                batch_size=batch_size, subset=subset, download=download, root=root
            ),
            "mnist",
        )
    except Exception:
        if tiny_mnist_available():
            return make_tiny_loaders(batch_size=batch_size), "tiny"
        n = max(subset or 128, 64)
        return make_toy_loaders(n=n, batch_size=batch_size), "toy"


def make_tiny_loaders(
    batch_size: int = 16,
    val_ratio: float = 0.2,
    seed: int = DEFAULT_SEED,
    path: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    images, labels = load_tiny_mnist(path)
    n = images.size(0)
    n_test = max(1, n // 5)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    test_idx, rest_idx = perm[:n_test], perm[n_test:]
    rest = TensorDataset(images[rest_idx], labels[rest_idx])
    test_ds = TensorDataset(images[test_idx], labels[test_idx])
    train_ds, val_ds = split_train_val(rest, val_ratio=val_ratio, seed=seed)
    kw = dict(batch_size=min(batch_size, n), shuffle=False)
    return (
        DataLoader(train_ds, **{**kw, "shuffle": True}),
        DataLoader(val_ds, **kw),
        DataLoader(test_ds, **kw),
    )
