# MNIST Lab 课程大纲

围绕 **手写数字 CNN** 一条主线，而不是全科机器学习百科。

## 学习路径

| 序号 | 课件 | 你将能做到 | 代码入口 |
| --- | --- | --- | --- |
| 00 | [notebooks/00_tensors.ipynb](../notebooks/00_tensors.ipynb) | 读懂 `[B,1,28,28]`，知道 backward 需要计算图 | `torch.Tensor` |
| 01 | [notebooks/01_data.ipynb](../notebooks/01_data.ipynb) | 划分 train/val/test，解释增强 | `mnist_lab.data` |
| 02 | [notebooks/02_mlp_vs_cnn.ipynb](../notebooks/02_mlp_vs_cnn.ipynb) | 对比 MLP 与 CNN 的归纳偏置 | `build_model` |
| 03 | [notebooks/03_train_loop.ipynb](../notebooks/03_train_loop.ipynb) | 默写五步训练循环 | `train_model` |
| 04 | [notebooks/04_metrics.ipynb](../notebooks/04_metrics.ipynb) | 手算混淆矩阵与 F1 | `mnist_lab.evaluate` |
| 05 | [notebooks/05_overfitting.ipynb](../notebooks/05_overfitting.ipynb) | 用曲线判断过拟合并选正则 | `cnn_dropout`, `weight_decay` |
| 06 | [notebooks/06_hyperparams.ipynb](../notebooks/06_hyperparams.ipynb) | 做对照实验并保存 runs | `runs/*/history.json` |
| 07 | [notebooks/07_interpret.ipynb](../notebooks/07_interpret.ipynb) | 看特征图 / saliency / Grad-CAM | `mnist_lab.interpret` |
| 08 | [notebooks/08_error_analysis.ipynb](../notebooks/08_error_analysis.ipynb) | 读误分类图册；NumPy 卷积对齐 PyTorch | `numpy_conv`, `visualize` |

## 课堂活动

1. 跑通 `python -m mnist_lab lesson`
2. 无网用 `--tiny` 或预置权重；`--toy` 只测 API，不要用来讲 Grad-CAM
3. `streamlit run app/streamlit_app.py`：短训、手写预测、坏实验诊断
4. 完成 `exercises/` 五题，`pytest tests/test_exercises.py`
5. `python -m mnist_lab quiz`

## 建议课时

- 实验课：按 00→08 顺序，每课配合实验室对应页面。节奏见 [teacher.md](teacher.md)。
- 作业：练习 1–5；选做把自己的数字照片用 `predict` 识别（注意白底会自动反色）。

## 不会覆盖

线性回归、树模型、Transformer、多机多卡训练。
