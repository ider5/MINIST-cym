from mnist_lab.digit_io import prepare_digit_tensor
import numpy as np


def test_prepare_inverts_white_and_keeps_black():
    white = np.full((32, 32), 255, dtype=np.uint8)
    white[8:24, 14:18] = 0
    t = prepare_digit_tensor(white, auto_invert=True)
    assert t.shape == (1, 1, 28, 28)
    assert float(t.mean()) < 0.5
    raw = prepare_digit_tensor(white, auto_invert=False)
    assert float(raw.mean()) > 0.5
