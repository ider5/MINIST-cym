"""全局教学配置：默认 CPU、固定类别数与路径约定。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

NUM_CLASSES = 10
IMAGE_SIZE = 28
IN_CHANNELS = 1

DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 50
DEFAULT_EPOCHS = 2
DEFAULT_LR = 0.001
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42

CHECKPOINT_NAME = "cnn2.pkl"
HISTORY_NAME = "history.json"
BEST_STATE_NAME = "best.pt"

DATA_ROOT = Path("./data")
OUTPUT_DIR = Path("./outputs")
RUNS_DIR = Path("./runs")

MODEL_NAMES = ("mlp", "cnn", "cnn_dropout")


@dataclass(frozen=True)
class TrainConfig:
    """一次训练实验的超参数快照，便于写入 runs/ 做对比。"""

    model: str = "cnn"
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    val_ratio: float = DEFAULT_VAL_RATIO
    weight_decay: float = 0.0
    subset: int | None = None
    seed: int = DEFAULT_SEED
    device: str = DEFAULT_DEVICE
