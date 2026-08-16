"""无窗口可视化：全部写入文件，不调用 cv2.imshow。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch

from mnist_lab.config import OUTPUT_DIR


def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_learning_curves(history: dict, path: str | Path | None = None) -> Path:
    path = _ensure_parent(Path(path) if path else OUTPUT_DIR / "learning_curves.png")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].plot(history.get("train_loss", []), label="train_loss")
    ax[0].plot(history.get("val_loss", []), label="val_loss")
    ax[0].set_title("损失曲线")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history.get("val_acc", []), label="val_acc", color="tab:green")
    ax[1].set_title("验证准确率")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_confusion_matrix(cm: np.ndarray, path: str | Path | None = None) -> Path:
    path = _ensure_parent(Path(path) if path else OUTPUT_DIR / "confusion_matrix.png")
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵")
    fig.colorbar(im, ax=ax, fraction=0.046)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _to_hw(img: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def save_sample_grid(
    images: torch.Tensor | np.ndarray,
    labels: np.ndarray | torch.Tensor,
    preds: np.ndarray | torch.Tensor | None = None,
    path: str | Path | None = None,
    max_n: int = 32,
) -> Path:
    path = _ensure_parent(Path(path) if path else OUTPUT_DIR / "sample_grid.png")
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    n = min(max_n, len(images))
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.3))
    axes = np.atleast_1d(axes).ravel()
    labels = np.asarray(labels)
    preds_arr = None if preds is None else np.asarray(preds)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(_to_hw(images[i]), cmap="gray")
        title = str(int(labels[i]))
        if preds_arr is not None:
            title = f"y={int(labels[i])} ŷ={int(preds_arr[i])}"
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_error_gallery(
    images: torch.Tensor | np.ndarray,
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    path: str | Path | None = None,
    max_n: int = 16,
) -> Path:
    path = _ensure_parent(Path(path) if path else OUTPUT_DIR / "error_gallery.png")
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    idx = np.where(y_true != y_pred)[0]
    if len(idx) == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        ax.set_title("没有误分类样本")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    idx = idx[:max_n]
    if isinstance(images, torch.Tensor):
        subset = images[idx]
    else:
        subset = np.asarray(images)[idx]
    return save_sample_grid(subset, y_true[idx], y_pred[idx], path=path, max_n=max_n)


def save_heatmap(
    image: np.ndarray | torch.Tensor,
    heat: np.ndarray,
    path: str | Path | None = None,
    title: str = "heatmap",
) -> Path:
    path = _ensure_parent(Path(path) if path else OUTPUT_DIR / "heatmap.png")
    fig, ax = plt.subplots(1, 2, figsize=(5, 2.4))
    ax[0].imshow(_to_hw(image), cmap="gray")
    ax[0].set_title("输入")
    ax[0].axis("off")
    ax[1].imshow(_to_hw(image), cmap="gray")
    ax[1].imshow(heat, cmap="jet", alpha=0.5)
    ax[1].set_title(title)
    ax[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
