import numpy as np

from mnist_lab.evaluate import accuracy, confusion_matrix, evaluate, per_class_report
from mnist_lab.models import build_model
from mnist_lab.data import make_toy_loaders


def test_accuracy_and_confusion_on_known_labels():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    assert abs(accuracy(y_true, y_pred) - 0.8) < 1e-9
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 2
    assert cm[2, 2] == 1
    report = per_class_report(cm)
    assert report["1"]["recall"] == 1.0
    assert report["0"]["recall"] == 0.5


def test_evaluate_runs_on_toy_model():
    model = build_model("mlp")
    _tr, _va, test = make_toy_loaders(n=32, batch_size=8)
    result = evaluate(model, test, device="cpu")
    assert result.confusion_matrix.shape == (10, 10)
    assert result.y_true.shape == result.y_pred.shape
    assert 0.0 <= result.accuracy <= 1.0
    assert "macro_avg" in result.report
