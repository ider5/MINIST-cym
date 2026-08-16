"""练习 5 参考答案。"""

from __future__ import annotations

import numpy as np


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1).astype(int)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_z
    return float(-log_probs[np.arange(len(labels)), labels].mean())
