"""生成可上满一节课的 notebooks。"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
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
    print("wrote", path, "cells", len(nb["cells"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    write(
        "00_tensors.ipynb",
        "00 张量与计算图",
        [
            md("## 目标\n学完你能口述：一张 MNIST 图为什么是 `[1,28,28]`；`backward` 为什么需要 `requires_grad`。"),
            md("## 会错的直觉\n「张量就是 numpy，和梯度无关。」——一旦要 `backward`，就必须在计算图上。"),
            md("### 形状\n- 一张灰度图：`[C,H,W] = [1,28,28]`\n- 一个 batch：`[B,1,28,28]`\n- 十分类 logits：`[B,10]`"),
            code(
                "import torch\n"
                "x = torch.zeros(4, 1, 28, 28)\n"
                "print('batch 图像', tuple(x.shape))\n"
                "assert x.shape == (4, 1, 28, 28)\n"
                "w = torch.randn(8, 1, 3, 3, requires_grad=True)\n"
                "y = torch.nn.functional.conv2d(x, w, padding=1)\n"
                "print('卷积输出', tuple(y.shape))\n"
                "assert y.shape == (4, 8, 28, 28)"
            ),
            md("课内练习：下面这格必须成功。若你改了 padding，先想清楚输出高宽。"),
            code(
                "x = torch.randn(2, 1, 28, 28, requires_grad=True)\n"
                "loss = x.pow(2).mean()\n"
                "loss.backward()\n"
                "assert x.grad is not None\n"
                "print('grad 均值', float(x.grad.abs().mean()))"
            ),
            md(
                "## 思考题\n"
                "1. 把 `requires_grad=True` 去掉再 backward，报错信息里哪几个词最关键？\n"
                "2. 为什么我们几乎从不把整份 60000 张一次放进一个张量里训练？\n"
                "3. `view` / `reshape` 会不会切断计算图？"
            ),
        ],
    )

    write(
        "01_data.ipynb",
        "01 数据管道、划分与增强",
        [
            md("## 目标\n能解释 train / val / test 各自只能干什么；知道增强发生在训练期。"),
            md("## 会错的直觉\n「准确率最高的那个划分方式最好。」——如果拿测试集反复试，数字会虚高。"),
            code(
                "from mnist_lab.data import make_toy_loaders, split_train_val, mnist_transforms\n"
                "from torch.utils.data import TensorDataset\n"
                "import torch\n"
                "train, val, test = make_toy_loaders(n=80, batch_size=16)\n"
                "print(len(train.dataset), len(val.dataset), len(test.dataset))\n"
                "bx, by = next(iter(train))\n"
                "assert bx.shape[1:] == (1, 28, 28)\n"
                "print('一个 batch', tuple(bx.shape), by[:4].tolist())"
            ),
            md("真实数字请用仓库里的 `fixtures/mnist_tiny.pt`（条纹 toy 只适合测 API）："),
            code(
                "from mnist_lab.tiny_data import make_tiny_loaders\n"
                "tr, va, te = make_tiny_loaders(batch_size=16)\n"
                "x, y = next(iter(tr))\n"
                "print('tiny batch', tuple(x.shape), 'labels', y[:8].tolist())\n"
                "assert x.min() >= 0 and x.max() <= 1"
            ),
            code(
                "print(mnist_transforms(augment=False))\n"
                "print(mnist_transforms(augment=True))"
            ),
            md(
                "## 思考题\n"
                "1. 为什么验证集要从**训练集**再划，而不是从测试集划？\n"
                "2. 增强如果也打在测试集上，评估还公平吗？\n"
                "3. `subset=512` 上课够用，但报告最终准确率时应该用什么数据？"
            ),
        ],
    )

    write(
        "02_mlp_vs_cnn.ipynb",
        "02 从全连接到卷积",
        [
            md("## 目标\n对照 MLP 与 CNN 的参数量与特征形状；能默写本课 CNN 的尺寸账本。"),
            md("## 会错的直觉\n「卷积会先翻转核再滑窗。」——PyTorch 的 conv2d 是互相关，不翻转。"),
            md(
                "本课 CNN：`Conv 1→16 k=5 p=2` → ReLU → Pool2 → "
                "`Conv 16→32 k=5 p=2` → ReLU → Pool2 → `Linear(32*7*7, 10)`。\n\n"
                "尺寸：`[B,1,28,28] → [B,16,14,14] → [B,32,7,7] → [B,10]`。"
            ),
            code(
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.shapes import trace_shapes, format_trace, cnn_expected_shapes\n"
                "cnn = build_model('cnn')\n"
                "rows = trace_shapes(cnn, batch=2)\n"
                "print(format_trace(rows))\n"
                "expect = dict(cnn_expected_shapes(2))\n"
                "got = dict(rows)\n"
                "for name, shape in expect.items():\n"
                "    assert got[name] == shape, (name, got.get(name), shape)\n"
                "print('尺寸账本核对通过')"
            ),
            code(
                "mlp = build_model('mlp')\n"
                "def nparams(m):\n"
                "    return sum(p.numel() for p in m.parameters())\n"
                "print('mlp', nparams(mlp), 'cnn', nparams(cnn))\n"
                "assert nparams(cnn) < nparams(mlp)"
            ),
            md(
                "## 思考题\n"
                "1. 为什么 CNN 参数更少，却通常更适合图像？\n"
                "2. padding=2、kernel=5 时，池化前空间尺寸为什么仍是 28？\n"
                "3. 若去掉两层 MaxPool，全连接输入还是 `32*7*7` 吗？"
            ),
        ],
    )

    write(
        "03_train_loop.ipynb",
        "03 训练循环与反向传播",
        [
            md("## 目标\n默写五步：zero_grad → 前向 → loss → backward → step。分清 batch 与 epoch。"),
            md("## 会错的直觉\n「忘了 zero_grad 也没关系，梯度会自动覆盖。」——默认是累加。"),
            code(
                "from mnist_lab.data import make_toy_loaders\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.train import train_model\n"
                "from mnist_lab.evaluate import evaluate\n"
                "tr, va, te = make_toy_loaders(n=64, batch_size=16)\n"
                "model = build_model('mlp')\n"
                "before = evaluate(model, tr).loss\n"
                "result = train_model(model, tr, va, epochs=3, lr=0.05, run_dir='runs/nb03')\n"
                "after = evaluate(model, tr).loss\n"
                "print('loss', round(before, 3), '→', round(after, 3), 'val_acc', result.history['val_acc'])\n"
                "assert after < before"
            ),
            md("对照：故意不 zero_grad（见实验室「诊断台」）。"),
            code(
                "from mnist_lab.bugs import run_bug_case\n"
                "payload = run_bug_case('forgot_zero_grad', 'runs/nb03_bug')\n"
                "print(payload['highlight'])"
            ),
            md(
                "## 思考题\n"
                "1. 一个 epoch 里 step 的次数和什么有关？\n"
                "2. 为什么我们按 **val_acc** 存 best.pt，而不是最后一步权重？\n"
                "3. Adam 的 lr=0.05 对 MLP toy 很猛，换成 CNN 全量 MNIST 还合适吗？"
            ),
        ],
    )

    write(
        "04_metrics.ipynb",
        "04 评估指标",
        [
            md("## 目标\n能手算 3 类混淆矩阵的 precision / recall；知道准确率会掩盖「总把 4 看成 9」。"),
            md("## 会错的直觉\n「准确率 90% 就很好。」——若 90% 样本都是 1，全猜 1 也有 90%。"),
            code(
                "import numpy as np\n"
                "from mnist_lab.evaluate import accuracy, confusion_matrix, per_class_report\n"
                "y_true = np.array([0, 0, 1, 1, 2, 2])\n"
                "y_pred = np.array([0, 1, 1, 1, 2, 0])\n"
                "cm = confusion_matrix(y_true, y_pred, num_classes=3)\n"
                "print(cm)\n"
                "assert cm[0, 0] == 1 and cm[0, 1] == 1\n"
                "print('acc', accuracy(y_true, y_pred))\n"
                "print(per_class_report(cm)['1'])"
            ),
            md("课内填空：类别 0 的 recall 是多少？用断言钉死。"),
            code(
                "r0 = per_class_report(cm)['0']['recall']\n"
                "print('class0 recall', r0)\n"
                "assert abs(r0 - 0.5) < 1e-9"
            ),
            md(
                "## 思考题\n"
                "1. 行=真实、列=预测时，对角线之外的大数字意味着什么？\n"
                "2. macro-F1 和 accuracy 何时差得最明显？\n"
                "3. 为什么本课禁止用 sklearn 交练习 1？"
            ),
        ],
    )

    write(
        "05_overfitting.ipynb",
        "05 过拟合与正则化",
        [
            md("## 目标\n看见「训练变好、验证变差」能说出过拟合；知道 Dropout 与 weight_decay 的使用阶段。"),
            md("## 会错的直觉\n「再多训几个 epoch 总会更好。」——过了某点只是在背训练集。"),
            code(
                "from mnist_lab.tiny_data import make_tiny_loaders\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.train import train_model\n"
                "tr, va, te = make_tiny_loaders(batch_size=32)\n"
                "plain = train_model(build_model('cnn'), tr, va, epochs=4, lr=0.01, run_dir='runs/nb05_plain')\n"
                "reg = train_model(build_model('cnn_dropout'), tr, va, epochs=4, lr=0.01, weight_decay=1e-3, run_dir='runs/nb05_reg')\n"
                "print('plain val', plain.history['val_acc'])\n"
                "print('dropout+wd val', reg.history['val_acc'])\n"
                "assert 'val_loss' in plain.history"
            ),
            code(
                "from exercises.solutions.ex04_overfit import is_overfitting\n"
                "assert is_overfitting([1.0, 0.2], [1.0, 1.6]) is True\n"
                "assert is_overfitting([1.0, 0.7], [1.0, 0.8]) is False\n"
                "print('过拟合检测器工作正常')"
            ),
            md(
                "## 思考题\n"
                "1. Dropout 在 `model.eval()` 时还随机丢神经元吗？\n"
                "2. 早停看的是哪条曲线？\n"
                "3. 数据增强算正则吗？它和 Dropout 惩罚的是同一件事吗？"
            ),
        ],
    )

    write(
        "06_hyperparams.ipynb",
        "06 超参数实验",
        [
            md("## 目标\n一次只改一个量；会读 `runs/<model>_<时间戳>/history.json`。"),
            md("## 会错的直觉\n「同时改 lr、batch、模型，谁涨了就是谁的功劳。」——你分不清。"),
            code(
                "from mnist_lab.data import make_toy_loaders\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.train import train_model\n"
                "from mnist_lab.run_dirs import new_run_dir\n"
                "tr, va, _ = make_toy_loaders(n=64, batch_size=16)\n"
                "for lr in (1e-1, 1e-3, 1e-5):\n"
                "    rd = new_run_dir(f'mlp_lr{lr}', root='runs', stamp=str(lr).replace('.', 'p'))\n"
                "    r = train_model(build_model('mlp'), tr, va, epochs=2, lr=lr, run_dir=rd)\n"
                "    print('lr', lr, 'train', r.history['train_loss'], 'val_acc', r.history['val_acc'])"
            ),
            md("课内练习：确认三次实验写进了三个不同目录。"),
            code(
                "from pathlib import Path\n"
                "dirs = list(Path('runs').glob('mlp_lr*'))\n"
                "print([p.name for p in dirs])\n"
                "assert len(dirs) >= 3"
            ),
            md(
                "## 思考题\n"
                "1. lr=1e-5 时 val_acc 几乎不动，你下一步改什么？\n"
                "2. 为什么实验室必须用时间戳目录，而不是永远覆盖 `lab_cnn`？\n"
                "3. 用测试集选最好的 lr 犯了哪条纪律？"
            ),
        ],
    )

    write(
        "07_interpret.ipynb",
        "07 模型可解释性",
        [
            md("## 目标\n能区分特征图、saliency、Grad-CAM；知道课堂必须看**真数字**而不是条纹。"),
            md("## 会错的直觉\n「热力图亮的地方就是像素本身最亮。」——Grad-CAM 亮的是对**当前类别**有贡献的区域。"),
            code(
                "from mnist_lab.tiny_data import make_tiny_loaders, DEFAULT_CHECKPOINT\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.interpret import feature_maps, saliency, gradcam\n"
                "import torch\n"
                "model = build_model('cnn')\n"
                "model.load_state_dict(torch.load(DEFAULT_CHECKPOINT, map_location='cpu'))\n"
                "_, _, te = make_tiny_loaders(batch_size=8)\n"
                "x, y = next(iter(te))\n"
                "x0 = x[:1]\n"
                "maps = feature_maps(model, x0)\n"
                "print({k: tuple(v.shape) for k, v in maps.items()})\n"
                "cam = gradcam(model, x0)\n"
                "sal = saliency(model, x0)\n"
                "assert cam.shape == (28, 28) and sal.shape == (28, 28)\n"
                "print('label', int(y[0]), 'cam max', float(cam.max()))"
            ),
            md("课内练习：CAM 必须是 28×28 且落在 [0,1]。"),
            code(
                "assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0 + 1e-6\n"
                "print('CAM 范围合法')"
            ),
            md(
                "## 思考题\n"
                "1. MLP 为什么没有 Grad-CAM（在本课实现里）？\n"
                "2. 若热力图糊在边上而不是笔画上，可能是模型差还是 CAM 层选错？\n"
                "3. 可解释性高亮能证明模型「理解数字」了吗？"
            ),
        ],
    )

    write(
        "08_error_analysis.ipynb",
        "08 错误分析、从零卷积与坏实验",
        [
            md("## 目标\n会看误分类图册；能让 NumPy 卷积对齐 PyTorch；能诊断五种经典事故。"),
            md("## 会错的直觉\n「准确率已经 85%，课就结束了。」——剩下的 15% 才告诉你下一步改数据还是改模型。"),
            code(
                "import numpy as np, torch, torch.nn.functional as F\n"
                "from mnist_lab.numpy_conv import conv2d_numpy\n"
                "img = np.random.randn(1, 1, 8, 8).astype(np.float32)\n"
                "ker = np.random.randn(2, 1, 3, 3).astype(np.float32)\n"
                "diff = np.max(np.abs(conv2d_numpy(img, ker, padding=1) - F.conv2d(torch.tensor(img), torch.tensor(ker), padding=1).numpy()))\n"
                "print('max |numpy-torch|', diff)\n"
                "assert diff < 1e-4"
            ),
            code(
                "from mnist_lab.tiny_data import make_tiny_loaders, DEFAULT_CHECKPOINT\n"
                "from mnist_lab.models import build_model\n"
                "from mnist_lab.evaluate import evaluate\n"
                "from mnist_lab.visualize import save_error_gallery\n"
                "import torch\n"
                "model = build_model('cnn')\n"
                "model.load_state_dict(torch.load(DEFAULT_CHECKPOINT, map_location='cpu'))\n"
                "_, _, te = make_tiny_loaders(batch_size=32)\n"
                "ev = evaluate(model, te)\n"
                "xs, ys = next(iter(te))\n"
                "with torch.no_grad():\n"
                "    pred = torch.argmax(model(xs), 1).numpy()\n"
                "print('acc', round(ev.accuracy, 3), 'saved', save_error_gallery(xs, ys.numpy(), pred, 'outputs/nb08_err.png'))"
            ),
            code(
                "from mnist_lab.bugs import list_cases, run_bug_case, grade_diagnosis\n"
                "print([c.id for c in list_cases()])\n"
                "p = run_bug_case('exploding_lr', 'runs/nb08_lr')\n"
                "print(p['highlight'])\n"
                "assert grade_diagnosis('exploding_lr', 'B')"
            ),
            md(
                "## 思考题\n"
                "1. 4 和 9 经常对调时，增强或加数据，哪件事更先做？\n"
                "2. 白底照片预测全错，你先检查模型还是先看预处理？\n"
                "3. 把测试集准确率写进训练循环当「验证」，报告还能信吗？"
            ),
        ],
    )


if __name__ == "__main__":
    main()
