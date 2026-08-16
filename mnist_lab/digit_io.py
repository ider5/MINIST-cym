"""从数组 / PIL 构造 MNIST 风格张量，可开关自动反色。"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from mnist_lab.config import IMAGE_SIZE


def prepare_digit_tensor(
    image: np.ndarray | Image.Image | torch.Tensor,
    *,
    auto_invert: bool = True,
) -> torch.Tensor:
    """返回 [1,1,28,28]，值域约 [0,1]。auto_invert 时白底会变成黑底白字。"""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float().numpy()
    elif isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=0) if arr.shape[0] <= 4 else arr.mean(axis=-1)
    if arr.max() > 1.5:
        arr = arr / 255.0
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if auto_invert and float(arr.mean()) > 0.5:
        arr = 1.0 - arr
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
