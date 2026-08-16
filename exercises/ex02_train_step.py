"""练习 2：补全训练一步。必须调用 zero_grad → backward → step。"""

from __future__ import annotations

import torch
import torch.nn as nn


def train_one_step(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """返回标量 loss。"""
    raise NotImplementedError("请写出完整的前向、反向与参数更新")
