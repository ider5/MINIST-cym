# MNIST Lab · 卷积神经网络教学实验

用 **28×28 手写数字** 这一条线，把数据划分、MLP vs CNN、训练循环、指标、过拟合、超参、可解释性和错误分析讲完。面向第一次上 CNN 实验课的同学；默认 **CPU**，图都写到 `outputs/`，不弹窗。

预置了 250 张真实数字（`fixtures/mnist_tiny.pt`）和一份小权重（`checkpoints/cnn_cpu.pt`），装好依赖就能预测，不必先下完整 MNIST。

## 五分钟上手

需要 Python 3.10+。

```bash
pip install -r requirements.txt
# 若上面装的是 GPU 版 PyTorch、而你只有 CPU：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

然后任选一条路：

| 你想先做什么 | 命令 |
| --- | --- |
| 打开交互实验室（推荐） | `streamlit run app/streamlit_app.py` |
| 立刻猜一张图 | `python -m mnist_lab predict --image-path your_digit.png` |
| 按课顺序读 | `python -m mnist_lab lesson` 然后打开 `notebooks/00_tensors.ipynb` |
| 自己短训一版 | `python -m mnist_lab train --tiny --epochs 3 --model cnn` |

没有 `your_digit.png` 时，进实验室选「手写预测」：画板点涂、上传、摄像头或从 fixture 里挑一张。

## 三种数据，别用错

| 开关 | 是什么 | 什么时候用 |
| --- | --- | --- |
| `--tiny`（默认课堂） | 仓库里的真实手写数字，250 张 | 无网上课、看 Grad-CAM、讲错例 |
| 不加快捷开关 | 完整 MNIST（需下载） | 有网、想看更像样的准确率；可加 `--subset 2048` 控制时间 |
| `--toy` | 合成**条纹**，不是数字 | 只给 `pytest` 和「坏实验」快速复现 |

**不要用 `--toy` 讲特征图 / Grad-CAM。** 条纹热力图会让人以为模型在看横杠。

## 学习路径（00 → 08）

每课都有：目标、一个会错的直觉、带 `assert` 的代码格、三道思考题。

| 课 | 笔记本 | 学完你能说出 |
| --- | --- | --- |
| 00 | [notebooks/00_tensors.ipynb](notebooks/00_tensors.ipynb) | 为什么是 `[B,1,28,28]`，`backward` 为什么要计算图 |
| 01 | [notebooks/01_data.ipynb](notebooks/01_data.ipynb) | train / val / test 各自只能干什么 |
| 02 | [notebooks/02_mlp_vs_cnn.ipynb](notebooks/02_mlp_vs_cnn.ipynb) | `[28×28] → 14×14 → 7×7 → 10` |
| 03 | [notebooks/03_train_loop.ipynb](notebooks/03_train_loop.ipynb) | zero_grad → 前向 → loss → backward → step |
| 04 | [notebooks/04_metrics.ipynb](notebooks/04_metrics.ipynb) | 混淆矩阵比准确率多告诉你什么 |
| 05 | [notebooks/05_overfitting.ipynb](notebooks/05_overfitting.ipynb) | 训练变好、验证变差该怎么办 |
| 06 | [notebooks/06_hyperparams.ipynb](notebooks/06_hyperparams.ipynb) | 一次只改一个量，结果在 `runs/` |
| 07 | [notebooks/07_interpret.ipynb](notebooks/07_interpret.ipynb) | 特征图、saliency、Grad-CAM 的差别 |
| 08 | [notebooks/08_error_analysis.ipynb](notebooks/08_error_analysis.ipynb) | 误分类图册 + 五种坏实验 |

完整大纲：[docs/curriculum.md](docs/curriculum.md) · 90 分钟课表：[docs/teacher.md](docs/teacher.md)

## 实验室页面

```bash
streamlit run app/streamlit_app.py
```

数据探查 → 模型与训练 → 手写预测 → 评估 → 可解释性 → **坏实验诊断** → 超参对照 → 测验。

每次短训写到 `runs/<模型>_<时间戳>/`，对照页可以叠多条曲线。模型可选 `mlp` / `cnn` / `cnn_dropout`。

## 命令行

```bash
python -m mnist_lab --help
python -m mnist_lab lesson                          # 列出课件
python -m mnist_lab train --tiny --epochs 3         # 真实小样本短训
python -m mnist_lab train --subset 2048 --epochs 2  # 完整 MNIST 的课上演示
python -m mnist_lab evaluate --tiny --split test    # 准确率 + 混淆矩阵
python -m mnist_lab visualize --kind gradcam --tiny
python -m mnist_lab quiz                            # 交互测验（13 题）
```

`visualize --kind`：`curves` | `confusion` | `errors` | `samples` | `gradcam` | `saliency` | `features`。

预测单张图时，若当前目录没有 `cnn2.pkl`，会自动用 `checkpoints/cnn_cpu.pt`（小样本短训，大约八成，不是论文数字）。白底照片会自动反色成 MNIST 的白字黑底。

旧入口仍可用：`python main.py --mode train --tiny`。

## 练习

在 [exercises/](exercises/) 填空，**不要** `import mnist_lab` 或 `sklearn` 交现成答案。

```bash
pytest tests/test_exercises.py -v
```

空实现会 `NotImplementedError`。参考答案在 `exercises/solutions/`，请先自己写。

## 测试

```bash
pytest
```

只用 toy 和仓库 fixture，**不下载完整 MNIST、不跑全量训练**。

## 目录

```
mnist_lab/           教学代码（中文注释）
app/streamlit_app.py 实验室
notebooks/           课件 00–08
fixtures/            真实数字小样本
checkpoints/         CPU 预置权重
exercises/           学生填空
tests/               快测
docs/                大纲与教师手册
outputs/  runs/      运行产物（已 gitignore）
```

重做小样本和权重：`PYTHONPATH=. python scripts/build_fixtures.py --n 256 --epochs 20`

## 常见问题

- **预测全错**：先看图是不是白底黑字却关掉了自动反色，再怀疑模型。
- **没有桌面 / 远程服务器**：不要找 `cv2.imshow`，看 `outputs/` 里的 png。
- **对照曲线只有一条**：旧版会覆盖同一目录；现在每次训练一个时间戳文件夹。
- **`cnn2.pkl` 加载失败**：`build_model("cnn")` 与最初两层卷积结构相同，可以加载旧权重。

本课不覆盖：线性回归、树模型、Transformer、多机多卡。
