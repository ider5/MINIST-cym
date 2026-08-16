"""练习 5：数值稳定的多分类交叉熵（logits + 类别下标）。"""

from __future__ import annotations

import numpy as np


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    logits: [N, C]，labels: [N] 整数。
    不要先 exp 再除——请用 log-sum-exp。
    """
    raise NotImplementedError
