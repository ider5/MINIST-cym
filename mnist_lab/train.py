"""训练循环：记录 history、按验证集准确率保存最优权重。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_lab.config import (
    BEST_STATE_NAME,
    CHECKPOINT_NAME,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    HISTORY_NAME,
    OUTPUT_DIR,
    RUNS_DIR,
    TrainConfig,
)
from mnist_lab.evaluate import evaluate


@dataclass
class TrainResult:
    history: dict
    best_state: dict
    run_dir: Path


def _json_ready(cfg: TrainConfig) -> dict:
    data = asdict(cfg)
    return data


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    device: str | torch.device = DEFAULT_DEVICE,
    run_dir: str | Path | None = None,
    weight_decay: float = 0.0,
    max_steps: Optional[int] = None,
    config: TrainConfig | None = None,
) -> TrainResult:
    """
    标准监督学习循环。history 的每个元素对应一个 epoch
    （若因 max_steps 提前结束，则只有已完成的记录，至少会写入当前折损）。
    """
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    if run_dir is None:
        run_dir = RUNS_DIR / "latest"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    global_step = 0
    stop = False

    for _epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break
        train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        val = evaluate(model, val_loader, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val.loss))
        history["val_acc"].append(float(val.accuracy))
        if val.accuracy >= best_acc:
            best_acc = val.accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stop:
            break

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / BEST_STATE_NAME)
    torch.save(best_state, run_dir / CHECKPOINT_NAME)
    (run_dir / HISTORY_NAME).write_text(
        json.dumps(
            {
                "history": history,
                "config": _json_ready(config) if config is not None else {
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "device": str(device),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainResult(history=history, best_state=best_state, run_dir=run_dir)
