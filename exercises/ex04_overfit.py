"""练习 4：从损失曲线判断过拟合，并选择正则手段。"""

from __future__ import annotations


def is_overfitting(train_loss: list[float], val_loss: list[float]) -> bool:
    """训练损失下降而验证损失相对起点上升时视为过拟合。"""
    raise NotImplementedError


def choose_regularization(overfit: bool) -> str:
    """过拟合时返回 'dropout' 或 'weight_decay'，否则返回 'none'。"""
    raise NotImplementedError
