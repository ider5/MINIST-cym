"""MNIST Lab 交互实验室。启动：streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnist_lab.data import make_mnist_loaders, make_toy_loaders, mnist_transforms
from mnist_lab.evaluate import evaluate
from mnist_lab.interpret import feature_maps, gradcam, saliency
from mnist_lab.models import build_model
from mnist_lab.quiz import grade_answers, load_questions
from mnist_lab.train import train_model
from mnist_lab.visualize import (
    save_confusion_matrix,
    save_error_gallery,
    save_heatmap,
    save_learning_curves,
    save_sample_grid,
)

st.set_page_config(page_title="MNIST Lab 教学实验室", layout="wide")


def _try_mnist(batch_size: int, subset: int | None):
    try:
        return make_mnist_loaders(
            batch_size=batch_size, subset=subset, download=True, root=str(ROOT / "data")
        ), False
    except Exception as exc:  # noqa: BLE001 — 教学环境可能无网
        st.warning(f"无法加载 MNIST（{exc}）。已切换为合成 toy 数据，功能仍可演示。")
        return make_toy_loaders(n=max(subset or 128, 64), batch_size=batch_size), True


def page_data():
    st.header("数据探查")
    st.markdown(
        "MNIST 是 28×28 灰度手写数字。训练前要划分 **train / val / test**："
        "训练集更新参数，验证集调超参、看过拟合，测试集只在最后报告。"
    )
    subset = st.slider("预览样本数 subset", 32, 512, 128, 32)
    loaders, toy = _try_mnist(batch_size=32, subset=subset)
    train_loader, val_loader, test_loader = loaders
    st.write(
        f"当前数据: {'toy 合成图案' if toy else 'MNIST'}；"
        f"train batches={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}"
    )
    xs, ys = next(iter(train_loader))
    counts = np.bincount(ys.numpy(), minlength=10)
    st.subheader("当前 batch 标签分布")
    st.bar_chart(counts)
    grid = save_sample_grid(xs, ys.numpy(), path=ROOT / "outputs" / "lab_samples.png")
    st.image(str(grid), caption="随机样本（标签写在图上）")
    st.subheader("数据增强前后")
    st.caption("旋转 + 平移。增强相当于告诉模型：数字稍微歪一点仍然是同一个字。")
    from torchvision.transforms import ToPILImage

    raw = xs[0]
    aug = mnist_transforms(augment=True)
    # ToTensor 输入需要 PIL。将 tensor 转 PIL 再增强。
    pil = ToPILImage()(raw)
    aug_t = aug(pil)
    col1, col2 = st.columns(2)
    col1.image(raw.squeeze().numpy(), caption="原始", clamp=True)
    col2.image(aug_t.squeeze().numpy(), caption="增强后", clamp=True)


def page_train():
    st.header("模型与训练")
    st.markdown(
        "一次参数更新 = `zero_grad` → 前向算 loss → `backward` 得到梯度 → `step`。"
        "CNN 用局部卷积核；MLP 把像素当独立特征。Dropout 只在 `model.train()` 时随机丢神经元。"
    )
    model_name = st.selectbox("模型", ["cnn", "mlp", "cnn_dropout"])
    col = st.columns(4)
    lr = col[0].number_input("学习率 lr", 1e-4, 1.0, 1e-3, format="%.4f")
    epochs = col[1].slider("epochs", 1, 5, 1)
    batch_size = col[2].slider("batch_size", 8, 64, 32)
    subset = col[3].slider("subset", 64, 1024, 256, 64)
    weight_decay = st.slider("weight_decay（L2 正则）", 0.0, 0.05, 0.0, 0.001)
    if st.button("开始短训", type="primary"):
        loaders, toy = _try_mnist(batch_size=batch_size, subset=subset)
        train_loader, val_loader, test_loader = loaders
        model = build_model(model_name)
        run_dir = ROOT / "runs" / f"lab_{model_name}"
        with st.spinner("训练中（CPU 短训）…"):
            result = train_model(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                lr=float(lr),
                device="cpu",
                run_dir=run_dir,
                weight_decay=float(weight_decay),
            )
        st.session_state["model"] = model
        st.session_state["result"] = result
        st.session_state["test_loader"] = test_loader
        st.session_state["toy"] = toy
        st.success(f"完成。最后验证准确率 {result.history['val_acc'][-1]:.3f}")
        curve = save_learning_curves(result.history, ROOT / "outputs" / "lab_curves.png")
        st.image(str(curve))
        st.json(result.history)
    elif "result" in st.session_state:
        st.info("使用上一次训练结果。")
        st.image(str(ROOT / "outputs" / "lab_curves.png"))


def page_eval():
    st.header("评估与错误分析")
    if "model" not in st.session_state:
        st.info("请先到「模型与训练」跑一次短训。")
        return
    model = st.session_state["model"]
    test_loader = st.session_state["test_loader"]
    result = evaluate(model, test_loader, device="cpu")
    st.metric("测试准确率", f"{result.accuracy:.3f}")
    st.write("宏平均", result.report["macro_avg"])
    cm_path = save_confusion_matrix(result.confusion_matrix, ROOT / "outputs" / "lab_cm.png")
    st.image(str(cm_path), caption="行=真实类别，列=预测类别")
    xs, ys = next(iter(test_loader))
    with torch.no_grad():
        preds = torch.argmax(model(xs), dim=1).cpu().numpy()
    err = save_error_gallery(xs, ys.numpy(), preds, ROOT / "outputs" / "lab_err.png")
    st.image(str(err), caption="误分类图册：看模型经常把谁当成谁")
    with st.expander("每类 precision / recall / F1"):
        st.json({k: v for k, v in result.report.items() if k != "macro_avg"})


def page_xai():
    st.header("可解释性")
    st.markdown(
        "**特征图**看卷积通道激活了什么结构；**saliency** 是输入梯度；"
        "**Grad-CAM** 用最后一层卷积的梯度加权，高亮对当前类别重要的区域。"
    )
    if "model" not in st.session_state:
        st.info("请先训练一个 cnn / cnn_dropout（MLP 没有 conv2）。")
        return
    model = st.session_state["model"]
    if not hasattr(model, "conv2"):
        st.error("当前模型是 MLP，请改用 cnn 再训一次。")
        return
    test_loader = st.session_state["test_loader"]
    xs, ys = next(iter(test_loader))
    idx = st.slider("选择 batch 内第几张", 0, len(xs) - 1, 0)
    x = xs[idx : idx + 1]
    st.image(x.squeeze().numpy(), caption=f"真实标签 {int(ys[idx])}", clamp=True, width=160)
    maps = feature_maps(model, x)
    conv1 = maps["conv1"][0, :8].unsqueeze(1)
    feat_path = save_sample_grid(
        conv1, list(range(conv1.size(0))), path=ROOT / "outputs" / "lab_feat.png"
    )
    st.subheader("conv1 前 8 个通道")
    st.image(str(feat_path))
    cam = gradcam(model, x)
    sal = saliency(model, x)
    p1 = save_heatmap(x[0], cam, ROOT / "outputs" / "lab_cam.png", title="Grad-CAM")
    p2 = save_heatmap(x[0], sal, ROOT / "outputs" / "lab_sal.png", title="saliency")
    c1, c2 = st.columns(2)
    c1.image(str(p1))
    c2.image(str(p2))


def page_quiz():
    st.header("随堂测验")
    questions = load_questions()
    answers = {}
    with st.form("quiz"):
        for q in questions:
            answers[q["id"]] = st.radio(
                f"{q['id']}. {q['prompt']}",
                list(q["choices"].keys()),
                format_func=lambda k, q=q: f"{k}. {q['choices'][k]}",
                index=None,
                key=f"quiz_{q['id']}",
            )
        submitted = st.form_submit_button("交卷")
    if submitted:
        cleaned = {k: (v or "") for k, v in answers.items()}
        score = grade_answers(cleaned, questions)
        st.success(f"得分 {score.correct}/{score.total}（{score.percent:.0f}%）")
        for d, q in zip(score.details, questions):
            icon = "✅" if d["ok"] else "❌"
            st.markdown(f"{icon} **{d['id']}** 正确答案 {d['answer']}。{q['explanation']}")


def page_compare():
    st.header("超参数 / 模型对照")
    st.markdown("把多次短训的 `runs/*/history.json` 画在一起，体会 lr、模型选择的影响。")
    run_root = ROOT / "runs"
    if not run_root.exists():
        st.info("还没有 runs/，请先训练。")
        return
    files = sorted(run_root.glob("*/history.json"))
    if not files:
        st.info("没有 history.json。")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        ax.plot(payload["history"].get("val_acc", []), label=f.parent.name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_acc")
    ax.legend()
    out = ROOT / "outputs" / "lab_compare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    st.image(str(out))


PAGES = {
    "数据探查": page_data,
    "模型与训练": page_train,
    "评估与错误分析": page_eval,
    "可解释性": page_xai,
    "超参对照": page_compare,
    "随堂测验": page_quiz,
}


def main():
    st.title("MNIST Lab · 卷积神经网络教学实验室")
    st.caption("默认 CPU。单元测试不会下载 MNIST；本页在有网络时会尝试下载数据集。")
    name = st.sidebar.radio("页面", list(PAGES))
    PAGES[name]()


if __name__ == "__main__":
    main()
