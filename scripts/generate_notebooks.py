"""生成 notebooks/ 下的中文课件。"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(source: str) -> dict:
    return nbf.v4.new_markdown_cell(source)


def code(source: str) -> dict:
    return nbf.v4.new_code_cell(source)


def write(name: str, title: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb["cells"] = [md(f"# {title}")] + cells
    path = OUT / name
    path.write_text(nbf.writes(nb), encoding="utf-8")
    print("wrote", path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    write(
        "00_tensors.ipynb",
        "00 张量与计算图",
        [
            md(
                "机器学习把数据表示成**张量**（多维数组）。MNIST 一张图是 `[1, 28, 28]`："
                "通道、高、宽。批量训练时再加 batch 维变成 `[B, 1, 28, 28]`。\n\n"
                "PyTorch 用计算图记录运算，`loss.backward()` 沿图反向传播梯度。"
            ),
            code(
                "import torch\n"
                "x = torch.randn(2, 1, 28, 28, requires_grad=True)\n"
                "w = torch.randn(1, 1, 3, 3, requires_grad=True)\n"
                "y = torch.nn.functional.conv2d(x, w, padding=1).mean()\n"
                "y.backward()\n"
                "print('x 的形状', x.shape)\n"
                "print('w.grad 是否有限', torch.isfinite(w.grad).all().item())"
            ),
            md("练习：把 `requires_grad=True` 去掉再 backward，观察报错。这就是「没有计算图」。"),
        ],
    )

    write(
        "01_data.ipynb",
        "01 数据管道、划分与增强",
        [
            md(
                "三个集合职责不同：\n"
                "- **train**：更新权重\n"
                "- **val**：选超参、发现过拟合\n"
                "- **test**：最终报告，平时不要反复偷看\n\n"
                "下面用合成 toy 数据走通 API（不下载 MNIST）。有网络时可把 `make_toy_loaders` 换成 `make_mnist_loaders(subset=512)`。"
            ),
            code(
                "from mnist_lab.data import make_toy_loaders, split_train_val, mnist_transforms\n"
                "train, val, test = make_toy_loaders(n=80, batch_size=16)\n"
                "bx, by = next(iter(train))\n"
                "print(bx.shape, by[:8])\n"
                "print('增强流水线', mnist_transforms(augment=True))"
            ),
        ],
    )

    write(
        "02_mlp_vs_cnn.ipynb",
        "02 从全连接到卷积",
        [
            md(
                "MLP 把 784 个像素当独立特征，空间邻近关系要重新学。"
                "CNN 用同一个 5×5 核在图上滑动（权值共享），先检测局部笔画，再组合成数字。\n\n"
                "本课 CNN 与旧版 `cnn2.pkl` 结构相同：16 通道 → 32 通道 → `Linear(32*7*7, 10)`。"
            ),
            code(
                "import torch\n"
                "from mnist_lab.models import build_model\n"
                "x = torch.randn(3, 1, 28, 28)\n"
                "for name in ('mlp', 'cnn', 'cnn_dropout'):\n"
                "    m = build_model(name)\n"
                "    print(name, m(x).shape, '参数量', sum(p.numel() for p in m.parameters()))"
            ),
        ],
    )

    write(
        "03_train_loop.ipynb",
        "03 训练循环与反向传播",
        [
            md(
                "每个 batch：\n"
                "1. `optimizer.zero_grad()` 清掉上一步梯度\n"
                "2. `logits = model(x)` 前向\n"
                "3. `loss = CrossEntropy(logits, y)`\n"
                "4. `loss.backward()` 反传\n"
                "5. `optimizer.step()` 用梯度更新参数\n\n"
                "`epoch` 是把训练集过一遍；`batch` 是一次更新用的小堆样本。"
            ),
            code(
                "from mnist_lab.data import make_toy_loaders\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.train import train_model\n"
                "from mnist_lab.evaluate import evaluate\n"
                "tr, va, te = make_toy_loaders(n=64, batch_size=16)\n"
                "model = build_model('cnn')\n"
                "before = evaluate(model, tr).loss\n"
                "result = train_model(model, tr, va, epochs=2, lr=0.05, run_dir='runs/nb03', max_steps=8)\n"
                "print('loss 前/后', round(before, 3), result.history['train_loss'])\n"
                "print('val_acc', result.history['val_acc'])"
            ),
        ],
    )

    write(
        "04_metrics.ipynb",
        "04 评估指标",
        [
            md(
                "准确率会被类别不均衡误导。混淆矩阵能看出「把 4 当成 9」这种结构错误。"
                "Precision = 预测为该类里有多少真是；Recall = 该类里找回了多少。"
            ),
            code(
                "import numpy as np\n"
                "from mnist_lab.evaluate import accuracy, confusion_matrix, per_class_report\n"
                "y_true = np.array([0, 0, 1, 1, 2, 2])\n"
                "y_pred = np.array([0, 1, 1, 1, 2, 0])\n"
                "print('acc', accuracy(y_true, y_pred))\n"
                "cm = confusion_matrix(y_true, y_pred, num_classes=3)\n"
                "print(cm)\n"
                "print(per_class_report(cm)['macro_avg'])"
            ),
        ],
    )

    write(
        "05_overfitting.ipynb",
        "05 过拟合与正则化",
        [
            md(
                "过拟合：训练集越来越好，验证集变差。常用对策：\n"
                "- **更多数据 / 增强**\n"
                "- **Dropout**（`cnn_dropout`）\n"
                "- **weight_decay**（Adam 的 L2）\n"
                "- **早停**：验证准确率不再升就停下（本包按 val_acc 存 best.pt）"
            ),
            code(
                "from mnist_lab.data import make_toy_loaders\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.train import train_model\n"
                "tr, va, te = make_toy_loaders(n=64, batch_size=16)\n"
                "plain = train_model(build_model('cnn'), tr, va, epochs=2, lr=0.05, run_dir='runs/nb05a', max_steps=6)\n"
                "reg = train_model(build_model('cnn_dropout'), tr, va, epochs=2, lr=0.05, weight_decay=1e-3, run_dir='runs/nb05b', max_steps=6)\n"
                "print('无正则 val_acc', plain.history['val_acc'])\n"
                "print('dropout+wd val_acc', reg.history['val_acc'])"
            ),
        ],
    )

    write(
        "06_hyperparams.ipynb",
        "06 超参数实验",
        [
            md(
                "一次只改一个量：学习率、batch size、epoch、模型。"
                "学习率太大损失会炸；太小则学不动。结果写在 `runs/*/history.json`，实验室「超参对照」页可叠加曲线。"
            ),
            code(
                "from mnist_lab.data import make_toy_loaders\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.train import train_model\n"
                "tr, va, _ = make_toy_loaders(n=64, batch_size=16)\n"
                "for lr in (1e-1, 1e-3, 1e-5):\n"
                "    r = train_model(build_model('mlp'), tr, va, epochs=1, lr=lr, run_dir=f'runs/nb06_{lr}', max_steps=4)\n"
                "    print('lr', lr, 'train_loss', r.history['train_loss'], 'val_acc', r.history['val_acc'])"
            ),
        ],
    )

    write(
        "07_interpret.ipynb",
        "07 模型可解释性",
        [
            md(
                "黑盒也可以部分打开：\n"
                "- 特征图：中间层每个通道的激活\n"
                "- saliency：对输入求梯度\n"
                "- Grad-CAM：最后一层卷积的加权激活，上采样回 28×28"
            ),
            code(
                "import torch\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.interpret import feature_maps, saliency, gradcam\n"
                "from mnist_lab.data import make_toy_loaders\n"
                "from mnist_lab.train import train_model\n"
                "tr, va, te = make_toy_loaders(n=48, batch_size=16)\n"
                "model = build_model('cnn')\n"
                "train_model(model, tr, va, epochs=1, lr=0.05, run_dir='runs/nb07', max_steps=6)\n"
                "x, y = next(iter(te))\n"
                "x0 = x[:1]\n"
                "print({k: v.shape for k, v in feature_maps(model, x0).items()})\n"
                "print('saliency', saliency(model, x0).shape, 'gradcam', gradcam(model, x0).shape)"
            ),
        ],
    )

    write(
        "08_error_analysis.ipynb",
        "08 错误分析与改进",
        [
            md(
                "看完准确率不要停。把误分类图册摊开：是不是 4/9、3/8 分不清？"
                "然后针对性地加增强、加深模型或加 Dropout。\n\n"
                "从零卷积：`conv2d_numpy` 与 `F.conv2d` 应对齐（atol=1e-4）。"
            ),
            code(
                "import numpy as np, torch, torch.nn.functional as F\n"
                "from mnist_lab.numpy_conv import conv2d_numpy, relu_numpy, max_pool2d_numpy\n"
                "from mnist_lab.data import make_toy_loaders\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.train import train_model\n"
                "from mnist_lab.evaluate import evaluate\n"
                "from mnist_lab.visualize import save_error_gallery\n"
                "img = np.random.randn(1, 1, 8, 8).astype(np.float32)\n"
                "ker = np.random.randn(1, 1, 3, 3).astype(np.float32)\n"
                "print('max |numpy-torch|', np.max(np.abs(conv2d_numpy(img, ker, padding=1) - F.conv2d(torch.tensor(img), torch.tensor(ker), padding=1).numpy())))\n"
                "tr, va, te = make_toy_loaders(n=48, batch_size=16)\n"
                "model = build_model('cnn')\n"
                "train_model(model, tr, va, epochs=1, lr=0.05, run_dir='runs/nb08', max_steps=6)\n"
                "ev = evaluate(model, te)\n"
                "xs, ys = next(iter(te))\n"
                "with torch.no_grad():\n"
                "    pred = torch.argmax(model(xs), 1).numpy()\n"
                "print('acc', ev.accuracy, 'saved', save_error_gallery(xs, ys.numpy(), pred, 'outputs/nb08_err.png'))"
            ),
        ],
    )


if __name__ == "__main__":
    main()
