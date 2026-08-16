import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from exercises import (
    ex01_metrics,
    ex02_train_step,
    ex03_conv,
    ex04_overfit,
    ex05_cross_entropy,
)
from exercises.solutions import (
    ex01_metrics as sol01,
    ex02_train_step as sol02,
    ex03_conv as sol03,
    ex04_overfit as sol04,
    ex05_cross_entropy as sol05,
)


def test_student_stubs_raise():
    with pytest.raises(NotImplementedError):
        ex01_metrics.accuracy(np.array([0]), np.array([0]))
    with pytest.raises(NotImplementedError):
        ex02_train_step.train_one_step(None, None, None, None, None)
    with pytest.raises(NotImplementedError):
        ex03_conv.conv2d_student(np.zeros((1, 1, 3, 3)), np.ones((1, 1, 2, 2)))
    with pytest.raises(NotImplementedError):
        ex04_overfit.is_overfitting([1.0, 0.1], [1.0, 2.0])
    with pytest.raises(NotImplementedError):
        ex05_cross_entropy.cross_entropy(np.zeros((2, 3)), np.array([0, 1]))


def test_solution_metrics():
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 0, 2])
    assert sol01.accuracy(y_true, y_pred) == 0.75
    cm = sol01.confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm[1, 0] == 1


def test_solution_train_step_updates_params():
    model = nn.Linear(4, 3)
    x = torch.randn(5, 4)
    y = torch.tensor([0, 1, 2, 1, 0])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    before = model.weight.detach().clone()
    loss = sol02.train_one_step(model, x, y, opt, loss_fn)
    assert isinstance(loss, float)
    assert not torch.equal(before, model.weight.detach())


def test_solution_conv_matches_pytorch():
    rng = np.random.default_rng(1)
    image = rng.standard_normal((1, 1, 6, 6)).astype(np.float32)
    kernel = rng.standard_normal((2, 1, 3, 3)).astype(np.float32)
    out = sol03.conv2d_student(image, kernel, padding=1)
    ref = F.conv2d(torch.tensor(image), torch.tensor(kernel), padding=1).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_solution_overfit_detector():
    assert sol04.is_overfitting([1.0, 0.2], [1.0, 1.5]) is True
    assert sol04.is_overfitting([1.0, 0.8], [1.0, 0.7]) is False
    assert sol04.choose_regularization(True) in {"dropout", "weight_decay"}
    assert sol04.choose_regularization(False) == "none"


def test_solution_cross_entropy_stable():
    logits = np.array([[10.0, 0.0, -5.0], [0.0, 0.0, 0.0]])
    labels = np.array([0, 2])
    got = sol05.cross_entropy(logits, labels)
    ref = float(
        F.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()
    )
    assert abs(got - ref) < 1e-5
    huge = np.array([[1000.0, -1000.0]])
    assert np.isfinite(sol05.cross_entropy(huge, np.array([0])))
