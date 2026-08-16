from pathlib import Path

import numpy as np
import torch

from mnist_lab.visualize import (
    save_confusion_matrix,
    save_error_gallery,
    save_learning_curves,
    save_sample_grid,
)


def test_visualize_writes_nonempty_files(tmp_path):
    history = {"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.7], "val_acc": [0.2, 0.5]}
    p1 = save_learning_curves(history, tmp_path / "curves.png")
    p2 = save_confusion_matrix(np.eye(10, dtype=int), tmp_path / "cm.png")
    images = torch.rand(8, 1, 28, 28)
    labels = np.arange(8)
    preds = np.array([0, 1, 2, 3, 9, 5, 6, 7])
    p3 = save_sample_grid(images, labels, preds, tmp_path / "grid.png")
    p4 = save_error_gallery(images, labels, preds, tmp_path / "err.png")
    for p in (p1, p2, p3, p4):
        assert Path(p).is_file()
        assert Path(p).stat().st_size > 0
