"""可解释性：特征图、输入梯度显著性、Grad-CAM。"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mnist_lab.config import IMAGE_SIZE


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("输入应为 [B,1,28,28] 或 [1,28,28]")
    return x


def _target_conv(model: nn.Module) -> nn.Module:
    if not hasattr(model, "conv2"):
        raise ValueError("当前模型没有 conv2，Grad-CAM / 特征图请使用 cnn 或 cnn_dropout")
    return model.conv2


def feature_maps(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """返回 conv1 / conv2 的激活，形状 [B, C, H, W]。"""
    x = _as_batch(x)
    model.eval()
    maps: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str) -> Callable:
        def _hook(_m, _inp, out):
            maps[name] = out.detach().cpu()

        return _hook

    if hasattr(model, "conv1"):
        hooks.append(model.conv1.register_forward_hook(make_hook("conv1")))
    if hasattr(model, "conv2"):
        hooks.append(model.conv2.register_forward_hook(make_hook("conv2")))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
    if not maps:
        raise ValueError("模型不含 conv1/conv2，无法提取特征图")
    return maps


def saliency(model: nn.Module, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
    """输入空间的梯度绝对值，返回 [H, W]。"""
    model.eval()
    x = _as_batch(x).detach().clone().requires_grad_(True)
    logits = model(x)
    if class_idx is None:
        class_idx = int(torch.argmax(logits, dim=1)[0].item())
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()
    grad = x.grad.detach().abs()[0, 0]
    arr = grad.cpu().numpy()
    vmax = arr.max() if arr.max() > 0 else 1.0
    return arr / vmax


def gradcam(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
) -> np.ndarray:
    """在 conv2 上计算 Grad-CAM，上采样到 28×28，返回 [H, W] 且位于 [0, 1]。"""
    model.eval()
    x = _as_batch(x)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = _target_conv(model)

    def fwd(_m, _inp, out):
        activations.append(out)

    def bwd(_m, _gin, gout):
        gradients.append(gout[0])

    h_f = layer.register_forward_hook(fwd)
    h_b = layer.register_full_backward_hook(bwd)
    try:
        logits = model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1)[0].item())
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        act = activations[0][0]
        grad = gradients[0][0]
        weights = grad.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * act).sum(0))
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        vmax = cam.max() if cam.max() > 0 else 1.0
        return cam / vmax
    finally:
        h_f.remove()
        h_b.remove()
