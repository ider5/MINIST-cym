from mnist_lab.data import make_toy_loaders
from mnist_lab.evaluate import evaluate
from mnist_lab.models import build_model
from mnist_lab.train import train_model


def test_train_loss_drops_after_few_steps(tmp_path):
    train_loader, val_loader, _ = make_toy_loaders(n=64, batch_size=16)
    model = build_model("mlp")
    before = evaluate(model, train_loader, device="cpu").loss
    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=3,
        lr=0.01,
        device="cpu",
        run_dir=tmp_path / "run",
    )
    after = evaluate(model, train_loader, device="cpu").loss
    assert after < before
    assert result.history["train_loss"][-1] <= result.history["train_loss"][0] + 0.5
    assert (tmp_path / "run" / "history.json").is_file()
    assert (tmp_path / "run" / "cnn2.pkl").is_file()
    assert "val_acc" in result.history
    assert len(result.history["train_loss"]) == 3
