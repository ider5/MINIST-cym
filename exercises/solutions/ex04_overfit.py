"""练习 4 参考答案。"""

from __future__ import annotations


def is_overfitting(train_loss: list[float], val_loss: list[float]) -> bool:
    if len(train_loss) < 2 or len(val_loss) < 2:
        return False
    return train_loss[-1] < train_loss[0] and val_loss[-1] > val_loss[0]


def choose_regularization(overfit: bool) -> str:
    return "dropout" if overfit else "none"
