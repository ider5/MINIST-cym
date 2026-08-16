# MNIST Lab · 卷积神经网络教学实验

把原来的单文件 MNIST CNN，改成一套可讲、可看、可练、可改的教学实验。

主线仍然是 **28×28 手写数字识别**，用它讲清数据划分、MLP vs CNN、训练循环、指标、过拟合、超参、可解释性与错误分析。

## 你能做什么

- **训练 / 预测**：兼容旧命令 `python main.py --mode train`，并提供 `python -m mnist_lab ...`
- **看见训练**：损失曲线、混淆矩阵、误分类图册（全部写入 `outputs/`，不弹窗）
- **看见卷积**：特征图、梯度显著性、Grad-CAM；NumPy 手写卷积对照 `nn.Conv2d`
- **对比实验**：`mlp` / `cnn` / `cnn_dropout`，调节 lr、batch、epoch、weight_decay
- **课件**：`notebooks/00`–`08`
- **测验 + 练习**：`python -m mnist_lab quiz`；`exercises/` 由 pytest 批改
- **实验室**：`streamlit run app/streamlit_app.py`

旧权重文件名仍为 `cnn2.pkl`。`build_model("cnn")` 的结构与最初脚本一致，可以直接 `load_state_dict`。

## 安装

```bash
pip install -r requirements.txt
```

需要 PyTorch CPU 版时：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 快速开始

```bash
# 列出课件
python -m mnist_lab lesson

# 合成数据短训（不下载 MNIST，适合无网 / CI）
python -m mnist_lab train --toy --epochs 2 --model cnn

# 有 MNIST 时的课上演示（subset 控制分钟级）
python -m mnist_lab train --subset 2048 --epochs 2 --model cnn

# 测试批次预测（保存网格图，不再 cv2.imshow）
python -m mnist_lab predict-test --toy

# 单张图片
python -m mnist_lab predict --image-path your_digit.png

# 评估 + 混淆矩阵
python -m mnist_lab evaluate --toy --split test

# 导出 Grad-CAM
python -m mnist_lab visualize --kind gradcam --toy

# 测验（非交互：准备 JSON 答案）
python -m mnist_lab quiz
```

兼容入口：

```bash
python main.py --mode train --toy
python main.py --mode predict_test --toy
python main.py --mode predict --image_path your_digit.png
```

交互实验室：

```bash
streamlit run app/streamlit_app.py
```

## 目录

```
mnist_lab/          教学代码包（中文注释）
app/streamlit_app.py  实验室
notebooks/          00–08 课件
exercises/          学生填空；solutions/ 为参考答案
tests/              快测，只用 toy / 手工张量
docs/curriculum.md  课程大纲
```

## 运行测试

```bash
pytest
```

测试默认 **不下载 MNIST、不跑全量训练**。

## 课程大纲

见 [docs/curriculum.md](docs/curriculum.md)。
