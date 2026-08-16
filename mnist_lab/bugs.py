"""故意写坏的训练病例：先看症状，再选病因。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_lab.config import DEFAULT_DEVICE
from mnist_lab.data import make_toy_loaders
from mnist_lab.models import build_model
from mnist_lab.train import train_model


@dataclass
class BugCase:
    id: str
    title: str
    symptom: str
    choices: dict[str, str]
    answer: str
    explanation: str


CASES: dict[str, BugCase] = {
    "exploding_lr": BugCase(
        id="exploding_lr",
        title="学习率过大",
        symptom="训练几个 step 后 train_loss 变成很大的数，曲线向上飞。",
        choices={
            "A": "模型层数不够",
            "B": "学习率过大，参数一步跳出合理范围",
            "C": "验证集太大",
            "D": "ReLU 必须换成 Sigmoid",
        },
        answer="B",
        explanation="lr=10 时 Adam 的更新幅度远超权重尺度，损失爆炸。对照实验用 lr=1e-3。",
    ),
    "shuffled_labels": BugCase(
        id="shuffled_labels",
        title="标签被打乱",
        symptom="训练损失在降，但准确率一直在 10% 附近徘徊。",
        choices={
            "A": "过拟合",
            "B": "学习率过小所以学不动",
            "C": "标签与图像没有对应关系，只能猜类别",
            "D": "Dropout 开太大",
        },
        answer="C",
        explanation="十类随机乱猜的期望准确率是 1/10。图像还在，监督信号是噪声。",
    ),
    "forgot_zero_grad": BugCase(
        id="forgot_zero_grad",
        title="忘记 zero_grad",
        symptom="同样的 batch 反复训练，loss 很快变得不稳定或异常大。",
        choices={
            "A": "batch size 必须是 2 的幂",
            "B": "梯度在 step 之间累加，等效于越来越大的更新",
            "C": "交叉熵不能用于十分类",
            "D": "需要更多 Dropout",
        },
        answer="B",
        explanation="PyTorch 默认累加梯度。每步都要 optimizer.zero_grad()，否则等于把历史梯度叠在一起。",
    ),
    "tune_on_test": BugCase(
        id="tune_on_test",
        title="用测试集调参",
        symptom="「验证准确率」看起来很高，换一份从没看过的数据就掉下来。",
        choices={
            "A": "模型结构太简单",
            "B": "把测试集当成验证集，数字被偷看过所以虚高",
            "C": "CPU 比 GPU 更准",
            "D": "epoch 太少",
        },
        answer="B",
        explanation="验证集可以反复看；测试集只能最后用一次。偷看测试集等于把答案当练习。",
    ),
    "no_invert": BugCase(
        id="no_invert",
        title="白底图没有反色",
        symptom="自己拍的数字怎么看都像 7，模型却很自信地猜成别的。预览图底色是白的。",
        choices={
            "A": "必须用 GPU",
            "B": "MNIST 是白字黑底，白底黑字分布相反",
            "C": "卷积核一定坏了",
            "D": "图片分辨率必须是 224",
        },
        answer="B",
        explanation="像素均值 >0.5 通常是白底。load_digit_image 会自动反色；关掉它就会错。",
    ),
}


def list_cases() -> list[BugCase]:
    return list(CASES.values())


def _clone_xy(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def _shuffled_loader(loader: DataLoader, seed: int = 0) -> DataLoader:
    x, y = _clone_xy(loader)
    g = torch.Generator().manual_seed(seed)
    y = y[torch.randperm(y.size(0), generator=g)]
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=loader.batch_size or 16, shuffle=True)


def _train_skipping_zero_grad(
    model: nn.Module,
    loader: DataLoader,
    steps: int,
    lr: float,
    device: str,
) -> list[float]:
    device_t = torch.device(device)
    model = model.to(device_t)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    model.train()
    it = iter(loader)
    for _ in range(steps):
        try:
            bx, by = next(it)
        except StopIteration:
            it = iter(loader)
            bx, by = next(it)
        bx, by = bx.to(device_t), by.to(device_t)
        loss = loss_fn(model(bx), by)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def run_bug_case(
    case_id: str,
    run_dir: str | Path,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """在 toy 数据上复现症状（快、不下载）。返回可供实验室展示的数字。"""
    if case_id not in CASES:
        raise ValueError(f"未知病例 {case_id}")
    train_loader, val_loader, test_loader = make_toy_loaders(n=64, batch_size=16)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"id": case_id, "title": CASES[case_id].title}

    if case_id == "exploding_lr":
        model = build_model("mlp")
        result = train_model(
            model,
            train_loader,
            val_loader,
            epochs=2,
            lr=10.0,
            device=device,
            run_dir=run_dir,
            max_steps=6,
        )
        payload["history"] = result.history
        payload["highlight"] = f"train_loss = {result.history['train_loss']}"

    elif case_id == "shuffled_labels":
        model = build_model("mlp")
        bad = _shuffled_loader(train_loader)
        result = train_model(
            model, bad, val_loader, epochs=3, lr=0.05, device=device, run_dir=run_dir
        )
        payload["history"] = result.history
        payload["highlight"] = f"val_acc = {result.history['val_acc']}（十类随机≈0.10）"

    elif case_id == "forgot_zero_grad":
        model = build_model("mlp")
        losses = _train_skipping_zero_grad(model, train_loader, steps=8, lr=0.05, device=device)
        payload["history"] = {"train_loss": losses, "val_loss": [], "val_acc": []}
        payload["highlight"] = f"无 zero_grad 的逐步 loss = {[round(v, 3) for v in losses]}"

    elif case_id == "tune_on_test":
        model = build_model("mlp")
        cheated = train_model(
            model,
            train_loader,
            test_loader,
            epochs=3,
            lr=0.05,
            device=device,
            run_dir=run_dir / "cheated",
        )
        honest_model = build_model("mlp")
        honest = train_model(
            honest_model,
            train_loader,
            val_loader,
            epochs=3,
            lr=0.05,
            device=device,
            run_dir=run_dir / "honest",
        )
        payload["history"] = cheated.history
        payload["highlight"] = (
            f"偷看测试当验证 val_acc={cheated.history['val_acc'][-1]:.2f}；"
            f"诚实验证 val_acc={honest.history['val_acc'][-1]:.2f}。"
            "前者用的其实是测试集。"
        )

    elif case_id == "no_invert":
        white = torch.ones(1, 1, 28, 28)
        white[0, 0, 6:22, 12:16] = 0.0
        inverted = 1.0 - white
        payload["history"] = {"train_loss": [], "val_loss": [], "val_acc": []}
        payload["white_mean"] = float(white.mean())
        payload["inverted_mean"] = float(inverted.mean())
        payload["images"] = {"raw": white, "inverted": inverted}
        payload["highlight"] = (
            f"白底均值 {payload['white_mean']:.2f}（像纸），"
            f"反色后均值 {payload['inverted_mean']:.2f}（像 MNIST）。"
        )

    return payload


def grade_diagnosis(case_id: str, choice: str) -> bool:
    case = CASES[case_id]
    return str(choice).strip().upper() == case.answer
