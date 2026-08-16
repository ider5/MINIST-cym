"""练习 2 参考答案。"""

from __future__ import annotations

import torch.nn as nn
import torch


def train_one_step(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = loss_fn(logits, batch_y)
    loss.backward()
    optimizer.step()
    return float(loss.item())
