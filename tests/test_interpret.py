import numpy as np
import torch

from mnist_lab.interpret import feature_maps, gradcam, saliency
from mnist_lab.models import build_model


def test_feature_maps_and_explanations_are_finite():
    model = build_model("cnn")
    x = torch.rand(1, 1, 28, 28)
    maps = feature_maps(model, x)
    assert maps["conv1"].shape[0] == 1
    assert maps["conv2"].shape[1] == 32
    sal = saliency(model, x, class_idx=3)
    cam = gradcam(model, x, class_idx=3)
    assert sal.shape == (28, 28)
    assert cam.shape == (28, 28)
    assert np.isfinite(sal).all()
    assert np.isfinite(cam).all()
    assert cam.min() >= 0
    assert cam.max() <= 1 + 1e-6


def test_gradcam_rejects_mlp():
    model = build_model("mlp")
    x = torch.rand(1, 1, 28, 28)
    try:
        gradcam(model, x)
        assert False
    except ValueError:
        pass
