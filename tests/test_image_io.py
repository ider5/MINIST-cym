from pathlib import Path

import numpy as np
from PIL import Image

from mnist_lab.data import load_digit_image


def test_load_digit_inverts_white_background(tmp_path):
    path = Path(tmp_path) / "digit.png"
    arr = np.full((40, 40), 255, dtype=np.uint8)
    arr[10:30, 18:22] = 0
    Image.fromarray(arr).save(path)
    t = load_digit_image(path)
    assert t.shape == (1, 1, 28, 28)
    assert float(t.mean()) < 0.5
    assert float(t.max()) > 0.5
