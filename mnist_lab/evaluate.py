"""评估指标：全部手写，不依赖 sklearn。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_lab.config import DEFAULT_DEVICE, NUM_CLASSES


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """行=真实类别，列=预测类别。"""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_report(
    cm: np.ndarray,
) -> dict:
    """由混淆矩阵计算 precision / recall / F1。"""
    num_classes = cm.shape[0]
    report: dict = {}
    precisions, recalls, f1s, supports = [], [], [], []
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        support = int(cm[k, :].sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        report[str(k)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    report["macro_avg"] = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "support": int(np.sum(supports)),
    }
    return report


@dataclass
class EvalResult:
    accuracy: float
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    report: dict
    loss: float = 0.0
    extras: dict = field(default_factory=dict)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = DEFAULT_DEVICE,
) -> EvalResult:
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        total_loss += float(loss_fn(logits, batch_y).item())
        n_batches += 1
        pred = torch.argmax(logits, dim=1)
        all_true.append(batch_y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    return EvalResult(
        accuracy=accuracy(y_true, y_pred),
        confusion_matrix=cm,
        y_true=y_true,
        y_pred=y_pred,
        report=per_class_report(cm),
        loss=total_loss / max(n_batches, 1),
    )
