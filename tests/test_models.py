import torch

from mnist_lab.models import CNN, build_model


def test_all_models_output_logits_shape():
    x = torch.randn(4, 1, 28, 28)
    for name in ("mlp", "cnn", "cnn_dropout"):
        model = build_model(name)
        y = model(x)
        assert y.shape == (4, 10), name


def test_unknown_model_raises():
    try:
        build_model("transformer")
        assert False
    except ValueError as exc:
        assert "transformer" in str(exc)


def test_cnn_matches_legacy_module_names():
    model = CNN()
    keys = set(model.state_dict())
    assert "conv1.0.weight" in keys
    assert "conv2.0.weight" in keys
    assert "out.weight" in keys
    assert model.conv1[0].weight.shape == (16, 1, 5, 5)
    assert model.conv2[0].weight.shape == (32, 16, 5, 5)
    assert model.out.weight.shape == (10, 32 * 7 * 7)


def test_cnn_checkpoint_roundtrip(tmp_path):
    src = build_model("cnn")
    path = tmp_path / "cnn2.pkl"
    torch.save(src.state_dict(), path)
    dst = build_model("cnn")
    dst.load_state_dict(torch.load(path, map_location="cpu"))
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        assert torch.allclose(src(x), dst(x))
