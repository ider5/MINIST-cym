"""练习 3 参考答案。"""

from __future__ import annotations

import numpy as np

from mnist_lab.numpy_conv import conv2d_numpy


def conv2d_student(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    return conv2d_numpy(image, kernel, stride=1, padding=padding)
