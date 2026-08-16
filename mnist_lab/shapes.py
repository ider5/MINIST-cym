"""逐层张量形状：把 CNN 从「魔法」变成可核对的尺寸账本。"""

from __future__ import annotations

import torch
import torch.nn as nn

from mnist_lab.config import IMAGE_SIZE, IN_CHANNELS


def trace_shapes(
    model: nn.Module,
    x: torch.Tensor | None = None,
    batch: int = 2,
) -> list[tuple[str, tuple[int, ...]]]:
    """
    记录一次前向中各主要模块的输出形状。
    CNN 期望：input → conv1 14×14 → conv2 7×7 → logits 10。
    """
    if x is None:
        x = torch.zeros(batch, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    model.eval()
    rows: list[tuple[str, tuple[int, ...]]] = [("input", tuple(x.shape))]
    hooks = []

    def make_hook(name: str):
        def _hook(_m, _inp, out):
            if torch.is_tensor(out):
                rows.append((name, tuple(out.shape)))

        return _hook

    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            y = model(x)
        rows.append(("logits", tuple(y.shape)))
    finally:
        for h in hooks:
            h.remove()
    return rows


def format_trace(rows: list[tuple[str, tuple[int, ...]]]) -> str:
    return " → ".join(f"{name}{shape}" for name, shape in rows)


def cnn_expected_shapes(batch: int = 2) -> list[tuple[str, tuple[int, ...]]]:
    """本课 CNN 的标准尺寸，供课件 assert 对照。"""
    return [
        ("input", (batch, 1, 28, 28)),
        ("conv1", (batch, 16, 14, 14)),
        ("conv2", (batch, 32, 7, 7)),
        ("out", (batch, 10)),
        ("logits", (batch, 10)),
    ]
