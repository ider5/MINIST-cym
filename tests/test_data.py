from torch.utils.data import TensorDataset
import torch

from mnist_lab.data import make_toy_loaders, split_train_val, take_subset, make_pattern_images


def test_toy_loaders_shapes_and_no_download():
    train, val, test = make_toy_loaders(n=40, batch_size=8)
    bx, by = next(iter(train))
    assert bx.shape[1:] == (1, 28, 28)
    assert by.ndim == 1
    assert len(train.dataset) + len(val.dataset) + len(test.dataset) == 40


def test_split_train_val_ratio():
    x = torch.zeros(100, 1, 28, 28)
    y = torch.zeros(100, dtype=torch.long)
    ds = TensorDataset(x, y)
    train, val = split_train_val(ds, val_ratio=0.2, seed=0)
    assert len(train) == 80
    assert len(val) == 20


def test_take_subset():
    x, y = make_pattern_images(30)
    ds = TensorDataset(x, y)
    sub = take_subset(ds, 10)
    assert len(sub) == 10


def test_split_rejects_bad_ratio():
    ds = TensorDataset(torch.zeros(10, 1), torch.zeros(10))
    try:
        split_train_val(ds, val_ratio=1.0)
        assert False, "should have raised"
    except ValueError:
        pass
