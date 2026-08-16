from pathlib import Path


def test_nine_lesson_notebooks_exist():
    names = [
        "00_tensors.ipynb",
        "01_data.ipynb",
        "02_mlp_vs_cnn.ipynb",
        "03_train_loop.ipynb",
        "04_metrics.ipynb",
        "05_overfitting.ipynb",
        "06_hyperparams.ipynb",
        "07_interpret.ipynb",
        "08_error_analysis.ipynb",
    ]
    root = Path("notebooks")
    for name in names:
        path = root / name
        assert path.is_file(), name
        assert path.stat().st_size > 100
