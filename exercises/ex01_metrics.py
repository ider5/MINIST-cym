"""练习 1：手写分类指标（禁止 import sklearn）。"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("请实现准确率 = 预测正确的比例")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    raise NotImplementedError("请返回形状 [num_classes, num_classes] 的整数矩阵，行=真实，列=预测")
