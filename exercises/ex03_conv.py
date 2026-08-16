"""练习 3：实现与 PyTorch 一致的互相关卷积。"""

from __future__ import annotations

import numpy as np


def conv2d_student(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    image: (N, C, H, W)
    kernel: (out_c, in_c, kH, kW)
    返回 (N, out_c, H', W')
    """
    raise NotImplementedError("提示：先 padding，再对每个窗口求 image_patch * kernel 的和")
