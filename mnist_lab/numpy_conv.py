"""从零实现二维卷积 / ReLU / 最大池化，用于对照 nn.Conv2d。"""

from __future__ import annotations

import numpy as np


def relu_numpy(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def _as_nchw(image: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    img = np.asarray(image)
    orig = img.shape
    if img.ndim == 2:
        img = img[None, None, ...]
    elif img.ndim == 3:
        img = img[None, ...]
    elif img.ndim != 4:
        raise ValueError("image 维数应为 2/3/4")
    return img, orig


def _as_oihw(kernel: np.ndarray) -> np.ndarray:
    ker = np.asarray(kernel)
    if ker.ndim == 2:
        ker = ker[None, None, ...]
    elif ker.ndim == 3:
        ker = ker[:, None, ...]
    elif ker.ndim != 4:
        raise ValueError("kernel 维数应为 2/3/4")
    return ker


def conv2d_numpy(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """
    互相关（与 PyTorch conv2d 一致，不翻转卷积核）。
    - image: (H, W) / (C, H, W) / (N, C, H, W)
    - kernel: (kH, kW) / (out_c, in_c, kH, kW)
    二维输入且 out_c=1 时返回二维；NCHW 输入返回 NCHW。
    """
    img, orig_shape = _as_nchw(np.asarray(image, dtype=np.float64))
    ker = _as_oihw(np.asarray(kernel, dtype=np.float64))
    n, c, _h, _w = img.shape
    out_c, in_c, kh, kw = ker.shape
    if c != in_c:
        raise ValueError(f"通道不匹配：image C={c}, kernel in_c={in_c}")
    if padding:
        img = np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, hp, wp = img.shape
    oh = (hp - kh) // stride + 1
    ow = (wp - kw) // stride + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("卷积输出尺寸非正，请检查 kernel / padding / stride")
    out = np.zeros((n, out_c, oh, ow), dtype=np.float64)
    for ni in range(n):
        for oc in range(out_c):
            for i in range(oh):
                for j in range(ow):
                    hs = i * stride
                    ws = j * stride
                    patch = img[ni, :, hs : hs + kh, ws : ws + kw]
                    out[ni, oc, i, j] = np.sum(patch * ker[oc])
    if len(orig_shape) == 2 and out_c == 1 and n == 1:
        return out[0, 0]
    if len(orig_shape) == 3 and n == 1:
        return out[0]
    return out


def max_pool2d_numpy(
    image: np.ndarray,
    kernel_size: int = 2,
    stride: int | None = None,
) -> np.ndarray:
    stride = kernel_size if stride is None else stride
    img, orig = _as_nchw(np.asarray(image, dtype=np.float64))
    n, c, h, w = img.shape
    oh = (h - kernel_size) // stride + 1
    ow = (w - kernel_size) // stride + 1
    out = np.zeros((n, c, oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            hs, ws = i * stride, j * stride
            out[:, :, i, j] = np.max(
                img[:, :, hs : hs + kernel_size, ws : ws + kernel_size], axis=(2, 3)
            )
    if len(orig) == 2:
        return out[0, 0]
    if len(orig) == 3:
        return out[0]
    return out
