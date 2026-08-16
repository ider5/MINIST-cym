from mnist_lab.models import build_model
from mnist_lab.tiny_data import DEFAULT_CHECKPOINT, TINY_PATH, load_tiny_mnist, make_tiny_loaders
import torch


def test_tiny_fixture_is_real_digits():
    images, labels = load_tiny_mnist()
    assert images.shape[0] >= 100
    assert images.shape[1:] == (1, 28, 28)
    assert set(labels.tolist()) == set(range(10))
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0


def test_tiny_loaders_and_checkpoint_forward():
    train, val, test = make_tiny_loaders(batch_size=8)
    bx, by = next(iter(train))
    assert bx.shape[1:] == (1, 28, 28)
    model = build_model("cnn")
    state = torch.load(DEFAULT_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    with torch.no_grad():
        y = model(bx)
    assert y.shape[0] == bx.shape[0]
    assert y.shape[1] == 10
    assert TINY_PATH.is_file()
