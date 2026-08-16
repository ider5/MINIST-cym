from pathlib import Path
import json


def test_nine_lessons_are_full_sessions():
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
        nb = json.loads(path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        text = "\n".join("".join(c.get("source", [])) for c in cells)
        assert len(cells) >= 6, name
        assert "目标" in text, name
        assert "思考题" in text, name
        assert "会错的直觉" in text, name
