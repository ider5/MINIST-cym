# 教师手册（90 分钟）

对象：第一次接触 CNN 的实验课。默认 CPU。不要把测试集当验证集。

## 时间盒

| 分钟 | 做什么 | 入口 |
| --- | --- | --- |
| 0–10 | 环境、`python -m mnist_lab lesson`、打开实验室 | README |
| 10–25 | 课件 00–02：形状账本 `[1,28,28]→14×14→7×7→10` | notebooks/00–02 |
| 25–45 | 短训 cnn（subset=256 或 tiny fixture）+ 看曲线 | 实验室「模型与训练」 |
| 45–60 | 上传/点选一张图，看 softmax 与 Grad-CAM | 「手写预测」「可解释性」 |
| 60–75 | 坏实验：lr=10、打乱标签、忘记 zero_grad | 「坏实验诊断」 |
| 75–90 | 练习 1 或 2 当场开写；测验 13 题作收尾 | exercises/、quiz |

作业：练习 1–5；选做自己拍一张数字（注意白底会自动反色）。

## 常见卡点

- **弹窗 / 无桌面**：本课不使用 `cv2.imshow`，图在 `outputs/`。
- **无网络**：不要强下 MNIST。使用 `fixtures/mnist_tiny.pt` 与 `checkpoints/cnn_cpu.pt`。
- **条纹 toy 拿来讲 CAM**：学生会以为模型在看横条。toy 只给 pytest 和诊断台复现速度。
- **预测全错**：先看预处理是不是白底没反色，再怀疑模型。
- **练习 3 直接 `from mnist_lab.numpy_conv import`**：`pytest tests/test_exercise_imports.py` 会抓。
- **对照曲线只有一条**：旧版把结果写进固定 `lab_cnn`。现在是 `runs/<model>_<时间戳>/`。

## 预置权重

`checkpoints/cnn_cpu.pt` 在 250 张平衡小样本上短训，验证准确率大约八成，**不是**论文数字。课堂用来「立刻猜一张图」。需要重做：

```bash
PYTHONPATH=. python scripts/build_fixtures.py --n 256 --epochs 20
```

## 练习答案要点

1. 准确率 = 相等比例；混淆矩阵行=真实、列=预测。
2. 顺序必须是 zero_grad → backward → step。
3. 互相关：窗口与核逐元素相乘再求和，与 `F.conv2d` 对齐。
4. 训练损失下降且验证损失相对起点上升 → 过拟合 → dropout 或 weight_decay。
5. 交叉熵用 log-sum-exp，禁止先 `exp` 再除（会炸）。

## 诊断台标准答案

- exploding_lr → B
- shuffled_labels → C
- forgot_zero_grad → B
- tune_on_test → B
- no_invert → B
