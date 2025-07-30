# MNIST 手写数字识别 CNN

这是一个基于 PyTorch 实现的卷积神经网络（CNN），用于识别 MNIST 手写数字。

## 功能

*   **训练**：训练一个新的 CNN 模型并将其保存到 `cnn2.pkl`。
*   **测试集预测**：加载预训练的模型，并在 MNIST 测试集上进行预测。
*   **单张图片预测**：加载预训练的模型，并对用户提供的单张手写数字图片进行预测。

## 安装

1.  克隆或下载此项目。
2.  安装所需的 Python 库：

    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

通过命令行参数可以控制不同的功能模式。

### 训练模型

运行以下命令来训练一个新的模型：

```bash
python main.py --mode train
```

训练完成后，模型将被保存为 `cnn2.pkl`。

### 在测试集上预测

使用预训练的模型在 MNIST 测试集上进行预测：

```bash
python main.py --mode predict_test
```

### 预测您自己的图片

您可以对自己提供的图片进行预测。请确保图片中的数字清晰，背景简洁。

```bash
python main.py --mode predict --image_path /path/to/your/image.png
```

将 `/path/to/your/image.png` 替换为您的图片路径。
