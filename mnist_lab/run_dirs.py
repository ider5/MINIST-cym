"""为每次训练分配不互相覆盖的 run 目录。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mnist_lab.config import RUNS_DIR


def new_run_dir(model: str, root: str | Path | None = None, stamp: str | None = None) -> Path:
    """例如 cnn_20260816-174801，便于实验室「超参对照」叠多条曲线。"""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model.strip()) or "run"
    ts = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(root) if root is not None else RUNS_DIR
    run_dir = path / f"{safe}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
