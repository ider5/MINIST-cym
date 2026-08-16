from mnist_lab.config import (
    CHECKPOINT_NAME,
    DEFAULT_DEVICE,
    IMAGE_SIZE,
    MODEL_NAMES,
    NUM_CLASSES,
    TrainConfig,
)


def test_defaults_are_cpu_and_mnist_shaped():
    assert DEFAULT_DEVICE == "cpu"
    assert NUM_CLASSES == 10
    assert IMAGE_SIZE == 28
    assert CHECKPOINT_NAME == "cnn2.pkl"


def test_model_names_match_plan():
    assert MODEL_NAMES == ("mlp", "cnn", "cnn_dropout")


def test_train_config_frozen_snapshot():
    cfg = TrainConfig(model="mlp", lr=0.01, subset=128)
    assert cfg.model == "mlp"
    assert cfg.device == "cpu"
    assert cfg.subset == 128
