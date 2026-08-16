"""练习 3 参考答案：独立实现互相关，不调用教学库。"""

from __future__ import annotations

import numpy as np


def conv2d_student(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    ker = np.asarray(kernel, dtype=np.float64)
    n, c, h, w = img.shape
    out_c, in_c, kh, kw = ker.shape
    if c != in_c:
        raise ValueError("通道不匹配")
    if padding:
        img = np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, hp, wp = img.shape
    oh = hp - kh + 1
    ow = wp - kw + 1
    out = np.zeros((n, out_c, oh, ow), dtype=np.float64)
    for ni in range(n):
        for oc in range(out_c):
            for i in range(oh):
                for j in range(ow):
                    patch = img[ni, :, i : i + kh, j : j + kw]
                    out[ni, oc, i, j] = np.sum(patch * ker[oc])
    return out
