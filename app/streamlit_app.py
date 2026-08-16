"""MNIST Lab 交互实验室。启动：streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnist_lab.bugs import grade_diagnosis, list_cases, run_bug_case
from mnist_lab.data import mnist_transforms
from mnist_lab.digit_io import prepare_digit_tensor
from mnist_lab.evaluate import evaluate
from mnist_lab.interpret import feature_maps, gradcam, saliency
from mnist_lab.models import build_model
from mnist_lab.quiz import grade_answers, load_questions
from mnist_lab.run_dirs import new_run_dir
from mnist_lab.tiny_data import (
    DEFAULT_CHECKPOINT,
    load_tiny_mnist,
    make_classroom_loaders,
    tiny_mnist_available,
)
from mnist_lab.train import train_model
from mnist_lab.visualize import (
    save_confusion_matrix,
    save_error_gallery,
    save_heatmap,
    save_learning_curves,
    save_sample_grid,
)
from torchvision.transforms import ToPILImage

st.set_page_config(page_title="MNIST Lab 教学实验室", layout="wide")


def _loaders(batch_size: int, subset: int | None, toy: bool = False):
    loaders, source = make_classroom_loaders(
        batch_size=batch_size, subset=subset, download=True, root=str(ROOT / "data"), toy=toy
    )
    if source == "toy":
        st.warning("当前是条纹 toy 数据：只能测 API。看数字/Grad-CAM 请用 fixtures 或完整 MNIST。")
    elif source == "tiny":
        st.info("使用仓库内置 fixtures/mnist_tiny.pt（真实手写数字小样本）。")
    return loaders, source


def _load_pretrained():
    model = build_model("cnn")
    if not DEFAULT_CHECKPOINT.is_file():
        return None
    model.load_state_dict(torch.load(DEFAULT_CHECKPOINT, map_location="cpu"))
    model.eval()
    return model


def page_data():
    st.header("数据探查")
    st.markdown(
        "MNIST 是 28×28 灰度手写数字。**train** 更新参数，**val** 调超参、看过拟合，**test** 只在最后报告。"
    )
    subset = st.slider("预览样本数 subset", 32, 512, 128, 32)
    loaders, source = _loaders(batch_size=32, subset=subset)
    train_loader, val_loader, test_loader = loaders
    st.write(
        f"来源={source}；train batches={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}"
    )
    xs, ys = next(iter(train_loader))
    st.bar_chart(np.bincount(ys.numpy(), minlength=10))
    grid = save_sample_grid(xs, ys.numpy(), path=ROOT / "outputs" / "lab_samples.png")
    st.image(str(grid), caption="随机样本")
    raw = xs[0]
    aug_t = mnist_transforms(augment=True)(ToPILImage()(raw))
    c1, c2 = st.columns(2)
    c1.image(raw.squeeze().numpy(), caption="原始", clamp=True)
    c2.image(aug_t.squeeze().numpy(), caption="增强后", clamp=True)


def page_train():
    st.header("模型与训练")
    st.markdown("一次更新 = `zero_grad` → 前向 → `backward` → `step`。每次训练写入带时间戳的 `runs/` 目录。")
    model_name = st.selectbox("模型", ["cnn", "mlp", "cnn_dropout"])
    col = st.columns(4)
    lr = col[0].number_input("学习率 lr", 1e-4, 1.0, 1e-3, format="%.4f")
    epochs = col[1].slider("epochs", 1, 8, 2)
    batch_size = col[2].slider("batch_size", 8, 64, 32)
    subset = col[3].slider("subset", 64, 1024, 256, 64)
    weight_decay = st.slider("weight_decay（L2）", 0.0, 0.05, 0.0, 0.001)
    if st.button("开始短训", type="primary"):
        loaders, source = _loaders(batch_size=batch_size, subset=subset)
        train_loader, val_loader, test_loader = loaders
        model = build_model(model_name)
        run_dir = new_run_dir(model_name, root=ROOT / "runs")
        with st.spinner("训练中…"):
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
        st.session_state["source"] = source
        st.success(f"完成。val_acc={result.history['val_acc'][-1]:.3f}  目录 {run_dir.name}")
        curve = save_learning_curves(result.history, ROOT / "outputs" / "lab_curves.png")
        st.image(str(curve))
        st.json(result.history)
    elif "result" in st.session_state:
        st.info(f"使用上一次训练：{st.session_state['result'].run_dir}")
        p = ROOT / "outputs" / "lab_curves.png"
        if p.is_file():
            st.image(str(p))


def page_eval():
    st.header("评估与错误分析")
    model = st.session_state.get("model") or _load_pretrained()
    if model is None:
        st.info("请先训练，或确认 checkpoints/cnn_cpu.pt 存在。")
        return
    if "test_loader" in st.session_state:
        test_loader = st.session_state["test_loader"]
    else:
        loaders, _ = _loaders(32, 256)
        test_loader = loaders[2]
    result = evaluate(model, test_loader, device="cpu")
    st.metric("准确率", f"{result.accuracy:.3f}")
    st.write("宏平均", result.report["macro_avg"])
    st.image(str(save_confusion_matrix(result.confusion_matrix, ROOT / "outputs" / "lab_cm.png")))
    xs, ys = next(iter(test_loader))
    with torch.no_grad():
        preds = torch.argmax(model(xs), dim=1).cpu().numpy()
    st.image(str(save_error_gallery(xs, ys.numpy(), preds, ROOT / "outputs" / "lab_err.png")))


def page_xai():
    st.header("可解释性")
    model = st.session_state.get("model") or _load_pretrained()
    if model is None or not hasattr(model, "conv2"):
        st.info("需要带 conv2 的 CNN。请短训 cnn，或使用预置权重。")
        return
    if tiny_mnist_available():
        images, labels = load_tiny_mnist()
        idx = st.slider("fixture 第几张", 0, int(images.size(0)) - 1, 0)
        x = images[idx : idx + 1]
        y = int(labels[idx])
    else:
        loaders, src = _loaders(8, 64)
        if src == "toy":
            st.error("可解释性不要用条纹 toy。请保留 fixtures/mnist_tiny.pt。")
            return
        xs, ys = next(iter(loaders[2]))
        idx = st.slider("batch 内", 0, len(xs) - 1, 0)
        x = xs[idx : idx + 1]
        y = int(ys[idx])
    st.image(x.squeeze().numpy(), caption=f"标签 {y}", clamp=True, width=160)
    conv1 = feature_maps(model, x)["conv1"][0, :8].unsqueeze(1)
    st.image(str(save_sample_grid(conv1, list(range(8)), path=ROOT / "outputs" / "lab_feat.png")))
    c1, c2 = st.columns(2)
    c1.image(str(save_heatmap(x[0], gradcam(model, x), ROOT / "outputs" / "lab_cam.png", "Grad-CAM")))
    c2.image(str(save_heatmap(x[0], saliency(model, x), ROOT / "outputs" / "lab_sal.png", "saliency")))


def page_draw():
    st.header("手写 / 上传预测")
    st.markdown(
        "用画图软件写一个数字并上传，或从 fixture 挑一张。"
        "右侧会显示自动反色后的 28×28、softmax 条形图和 Grad-CAM。"
        "没有自己训的模型时，使用 `checkpoints/cnn_cpu.pt`（小样本短训，不是 SOTA）。"
    )
    model = st.session_state.get("model") or _load_pretrained()
    if model is None:
        st.error("找不到权重。")
        return
    auto_invert = st.checkbox("白底自动反色（建议开启）", value=True)
    uploaded = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])
    x = None
    caption = ""
    if uploaded is not None:
        pil = Image.open(uploaded).convert("L")
        raw = prepare_digit_tensor(pil, auto_invert=False)
        x = prepare_digit_tensor(pil, auto_invert=auto_invert)
        caption = "上传图"
        c1, c2 = st.columns(2)
        c1.image(raw.squeeze().numpy(), caption="未反色", clamp=True, width=140)
        c2.image(x.squeeze().numpy(), caption="送进模型", clamp=True, width=140)
    elif tiny_mnist_available():
        images, labels = load_tiny_mnist()
        idx = st.slider("或选 fixture", 0, int(images.size(0)) - 1, 7)
        x = images[idx : idx + 1]
        caption = f"fixture 标签 {int(labels[idx])}"
        st.image(x.squeeze().numpy(), caption=caption, clamp=True, width=140)
    if x is None:
        return
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
    st.metric("预测", f"{pred}  置信度 {probs[pred]:.2f}")
    st.bar_chart({"softmax": probs})
    if hasattr(model, "conv2"):
        heat = gradcam(model, x, class_idx=pred)
        st.image(str(save_heatmap(x[0], heat, ROOT / "outputs" / "lab_pred_cam.png", "Grad-CAM")))


def page_bugs():
    st.header("坏实验诊断台")
    st.markdown("先看症状，再选病因。不要先翻答案。病例在 toy 数据上快速复现（这里是测诊断，不是看笔画）。")
    case = st.selectbox("病例", list_cases(), format_func=lambda c: f"{c.id} · {c.title}")
    if st.button("复现症状"):
        payload = run_bug_case(case.id, ROOT / "runs" / f"bug_{case.id}")
        st.session_state["bug_payload"] = payload
        st.session_state["bug_id"] = case.id
    payload = st.session_state.get("bug_payload")
    if payload and st.session_state.get("bug_id") == case.id:
        st.write(case.symptom)
        st.code(payload.get("highlight", ""))
        hist = payload.get("history") or {}
        if hist.get("train_loss"):
            st.line_chart({"train_loss": hist["train_loss"]})
        if hist.get("val_acc"):
            st.line_chart({"val_acc": hist["val_acc"]})
        if "images" in payload:
            c1, c2 = st.columns(2)
            c1.image(payload["images"]["raw"].squeeze().numpy(), caption="白底", clamp=True, width=120)
            c2.image(payload["images"]["inverted"].squeeze().numpy(), caption="反色后", clamp=True, width=120)
    choice = st.radio("病因是？", list(case.choices.keys()), format_func=lambda k: f"{k}. {case.choices[k]}")
    if st.button("提交诊断"):
        ok = grade_diagnosis(case.id, choice)
        st.success("诊断正确") if ok else st.error("不对，再看看曲线")
        st.info(case.explanation)


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
        score = grade_answers({k: (v or "") for k, v in answers.items()}, questions)
        st.success(f"得分 {score.correct}/{score.total}（{score.percent:.0f}%）")
        for d, q in zip(score.details, questions):
            icon = "对" if d["ok"] else "错"
            st.markdown(f"**{icon} {d['id']}** 答案 {d['answer']}。{q['explanation']}")


def page_compare():
    st.header("超参数 / 模型对照")
    st.markdown("每次短训一个时间戳目录，这里把 `runs/*/history.json` 叠在一起。")
    files = sorted((ROOT / "runs").glob("*/history.json")) if (ROOT / "runs").exists() else []
    files = [f for f in files if f.parent.name != "_train_run"]
    if not files:
        st.info("还没有 runs/，请先训练至少两次（改 lr 或模型）。")
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
    ax.legend(fontsize=7)
    out = ROOT / "outputs" / "lab_compare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    st.image(str(out))


PAGES = {
    "数据探查": page_data,
    "模型与训练": page_train,
    "手写预测": page_draw,
    "评估与错误分析": page_eval,
    "可解释性": page_xai,
    "坏实验诊断": page_bugs,
    "超参对照": page_compare,
    "随堂测验": page_quiz,
}


def main():
    st.title("MNIST Lab · 卷积神经网络教学实验室")
    st.caption("预置权重 checkpoints/cnn_cpu.pt；真实小样本 fixtures/mnist_tiny.pt。")
    PAGES[st.sidebar.radio("页面", list(PAGES))]()


if __name__ == "__main__":
    main()
