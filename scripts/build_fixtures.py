"""下载一小份 MNIST 并短训 CPU 权重，供无网课堂使用。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms

from mnist_lab.models import build_model
from mnist_lab.tiny_data import DEFAULT_CHECKPOINT, TINY_PATH
from mnist_lab.train import train_model
from torch.utils.data import DataLoader, TensorDataset, random_split


def export_tiny(n: int, out: Path, data_root: str) -> None:
    ds = datasets.MNIST(root=data_root, train=True, transform=transforms.ToTensor(), download=True)
    images, labels = [], []
    per = max(1, n // 10)
    counts = {i: 0 for i in range(10)}
    for img, y in ds:
        if counts[int(y)] >= per:
            if sum(counts.values()) >= n:
                break
            continue
        images.append(img)
        labels.append(int(y))
        counts[int(y)] += 1
        if sum(counts.values()) >= n:
            break
    x = torch.stack(images, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": x, "labels": y}, out)
    print(f"wrote {out} shape={tuple(x.shape)} counts={counts}")


def train_checkpoint(fixture: Path, ckpt: Path, epochs: int) -> None:
    payload = torch.load(fixture, map_location="cpu")
    images, labels = payload["images"].float(), payload["labels"].long()
    n = images.size(0)
    n_val = max(1, n // 5)
    rest, val = random_split(
        TensorDataset(images, labels),
        [n - n_val, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    model = build_model("cnn")
    result = train_model(
        model,
        DataLoader(rest, batch_size=32, shuffle=True),
        DataLoader(val, batch_size=32),
        epochs=epochs,
        lr=0.01,
        device="cpu",
        run_dir=ckpt.parent / "_train_run",
    )
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.best_state, ckpt)
    print(f"wrote {ckpt} val_acc={result.history['val_acc'][-1]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--fixture", default=str(TINY_PATH))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    export_tiny(args.n, Path(args.fixture), args.data_root)
    train_checkpoint(Path(args.fixture), Path(args.checkpoint), args.epochs)


if __name__ == "__main__":
    main()
