import numpy as np
import torch
import torch.nn.functional as F

from mnist_lab.numpy_conv import conv2d_numpy, max_pool2d_numpy, relu_numpy


def test_conv2d_matches_pytorch():
    rng = np.random.default_rng(0)
    image = rng.standard_normal((2, 1, 8, 8)).astype(np.float32)
    kernel = rng.standard_normal((3, 1, 3, 3)).astype(np.float32)
    out_np = conv2d_numpy(image, kernel, padding=1)
    out_pt = F.conv2d(torch.tensor(image), torch.tensor(kernel), padding=1).numpy()
    np.testing.assert_allclose(out_np, out_pt, atol=1e-4, rtol=1e-4)


def test_conv2d_2d_input():
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    kernel = np.ones((3, 3), dtype=np.float64)
    out = conv2d_numpy(image, kernel, padding=0)
    assert out.shape == (2, 2)


def test_relu_and_pool():
    x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    np.testing.assert_array_equal(relu_numpy(x), np.array([[0.0, 2.0], [3.0, 0.0]]))
    pooled = max_pool2d_numpy(np.arange(16, dtype=float).reshape(4, 4), kernel_size=2)
    assert pooled.shape == (2, 2)
    assert pooled[0, 0] == 5
