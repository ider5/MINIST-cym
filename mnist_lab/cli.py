"""命令行入口：train / predict / evaluate / visualize / quiz / lesson。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from mnist_lab.config import (
    CHECKPOINT_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    OUTPUT_DIR,
    RUNS_DIR,
    TrainConfig,
)
from mnist_lab.data import load_digit_image, make_mnist_loaders, make_toy_loaders
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

NOTEBOOKS = [
    "00_tensors.ipynb — 张量与计算图直觉",
    "01_data.ipynb — 数据管道、划分与增强",
    "02_mlp_vs_cnn.ipynb — 从全连接到卷积",
    "03_train_loop.ipynb — 训练循环与反向传播",
    "04_metrics.ipynb — 评估指标",
    "05_overfitting.ipynb — 过拟合与正则化",
    "06_hyperparams.ipynb — 超参数实验",
    "07_interpret.ipynb — 可解释性",
    "08_error_analysis.ipynb — 错误分析与改进",
]


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到权重 {path}。请先运行: python -m mnist_lab train"
        )
    model.load_state_dict(_torch_load(path))


def _add_data_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--subset", type=int, default=None, help="截断样本数，课上演示用")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--download", action="store_true", default=True)
    p.add_argument("--no-download", action="store_false", dest="download")
    p.add_argument("--toy", action="store_true", help="使用合成数据，不下载 MNIST")


def cmd_train(args: argparse.Namespace) -> int:
    cfg = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        weight_decay=args.weight_decay,
        subset=args.subset,
        device=args.device,
    )
    if args.toy:
        train_loader, val_loader, _test = make_toy_loaders(
            n=max(args.subset or 128, 64),
            batch_size=args.batch_size,
        )
    else:
        train_loader, val_loader, _test = make_mnist_loaders(
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            subset=args.subset,
            download=args.download,
            root=args.data_root,
            augment=args.augment,
        )
    model = build_model(args.model)
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / args.model
    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        run_dir=run_dir,
        weight_decay=args.weight_decay,
        config=cfg,
    )
    cwd_ckpt = Path(CHECKPOINT_NAME)
    torch.save(result.best_state, cwd_ckpt)
    print(f"训练完成。最优验证准确率: {result.history['val_acc'][-1]:.4f}")
    print(f"权重已写入 {run_dir / CHECKPOINT_NAME} 以及 ./{CHECKPOINT_NAME}")
    save_learning_curves(result.history, OUTPUT_DIR / "learning_curves.png")
    print(f"损失曲线: {OUTPUT_DIR / 'learning_curves.png'}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = build_model(args.model)
    _load_checkpoint(model, Path(args.checkpoint))
    model.eval()
    x = load_digit_image(args.image_path)
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1)[0].item())
        prob = torch.softmax(logits, dim=1)[0, pred].item()
    print(f"图片 {args.image_path} 的预测结果是: {pred}（置信度 {prob:.3f}）")
    out = Path(args.out) if args.out else OUTPUT_DIR / "predict.png"
    save_sample_grid(x, [pred], path=out, max_n=1)
    print(f"已保存预览: {out}")
    return 0


def cmd_predict_test(args: argparse.Namespace) -> int:
    model = build_model(args.model)
    _load_checkpoint(model, Path(args.checkpoint))
    if args.toy:
        _tr, _va, test_loader = make_toy_loaders(n=64, batch_size=32)
    else:
        _tr, _va, test_loader = make_mnist_loaders(
            batch_size=32,
            subset=args.subset or 32,
            download=args.download,
            root=args.data_root,
        )
    batch_x, batch_y = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(batch_x), dim=1).cpu().numpy()
    print(f"被测数字: {batch_y.numpy()}")
    print(f"测试结果: {pred}")
    out = OUTPUT_DIR / "predict_test.png"
    save_sample_grid(batch_x, batch_y.numpy(), pred, path=out)
    print(f"已保存网格图: {out}（不再弹出窗口）")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = build_model(args.model)
    _load_checkpoint(model, Path(args.checkpoint))
    if args.toy:
        loaders = make_toy_loaders(n=64, batch_size=args.batch_size)
    else:
        loaders = make_mnist_loaders(
            batch_size=args.batch_size,
            subset=args.subset,
            download=args.download,
            root=args.data_root,
        )
    split = {"train": 0, "val": 1, "test": 2}[args.split]
    result = evaluate(model, loaders[split], device=args.device)
    print(f"{args.split} 准确率: {result.accuracy:.4f}  损失: {result.loss:.4f}")
    print(json.dumps(result.report["macro_avg"], ensure_ascii=False, indent=2))
    save_confusion_matrix(result.confusion_matrix, OUTPUT_DIR / "confusion_matrix.png")
    print(f"混淆矩阵: {OUTPUT_DIR / 'confusion_matrix.png'}")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.kind == "curves":
        hist_path = Path(args.run_dir) / "history.json"
        payload = json.loads(hist_path.read_text(encoding="utf-8"))
        path = save_learning_curves(payload["history"], OUTPUT_DIR / "learning_curves.png")
        print(path)
        return 0
    model = build_model(args.model)
    _load_checkpoint(model, Path(args.checkpoint))
    if args.toy:
        _tr, _va, test_loader = make_toy_loaders(n=64, batch_size=32)
    else:
        _tr, _va, test_loader = make_mnist_loaders(
            batch_size=32, subset=args.subset or 64, download=args.download, root=args.data_root
        )
    xs, ys = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(xs), dim=1).cpu().numpy()
    y_np = ys.numpy()
    if args.kind == "confusion":
        from mnist_lab.evaluate import confusion_matrix

        cm = confusion_matrix(y_np, preds)
        print(save_confusion_matrix(cm, OUTPUT_DIR / "confusion_matrix.png"))
    elif args.kind == "errors":
        print(save_error_gallery(xs, y_np, preds, OUTPUT_DIR / "error_gallery.png"))
    elif args.kind == "samples":
        print(save_sample_grid(xs, y_np, preds, OUTPUT_DIR / "sample_grid.png"))
    elif args.kind == "gradcam":
        heat = gradcam(model, xs[:1])
        print(save_heatmap(xs[0], heat, OUTPUT_DIR / "gradcam.png", title="Grad-CAM"))
    elif args.kind == "saliency":
        heat = saliency(model, xs[:1])
        print(save_heatmap(xs[0], heat, OUTPUT_DIR / "saliency.png", title="saliency"))
    elif args.kind == "features":
        maps = feature_maps(model, xs[:1])
        conv1 = maps["conv1"][0, :8]
        print(save_sample_grid(conv1.unsqueeze(1), list(range(len(conv1))), path=OUTPUT_DIR / "feature_maps.png"))
    return 0


def cmd_quiz(args: argparse.Namespace) -> int:
    questions = load_questions()
    if args.answers:
        try:
            answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"无法读取答案 JSON: {exc}")
            return 2
        score = grade_answers(answers, questions)
        print(f"得分 {score.correct}/{score.total} ({score.percent:.0f}%)")
        for d in score.details:
            mark = "✓" if d["ok"] else "✗"
            print(f"  {mark} {d['id']}: 你的答案 {d['given'] or '空'} / 正确 {d['answer']}")
            if not d["ok"]:
                print(f"     {d['explanation']}")
        return 0 if score.correct == score.total else 1
    answers = {}
    for q in questions:
        print(f"\n[{q['id']}] {q['prompt']}")
        for k, v in q["choices"].items():
            print(f"  {k}. {v}")
        answers[q["id"]] = input("你的选择: ").strip()
    score = grade_answers(answers, questions)
    print(f"\n得分 {score.correct}/{score.total} ({score.percent:.0f}%)")
    return 0


def cmd_lesson(_args: argparse.Namespace) -> int:
    print("MNIST Lab 课件（见 notebooks/ 目录）：")
    for line in NOTEBOOKS:
        print(f"  - {line}")
    print("交互实验室: streamlit run app/streamlit_app.py")
    print("编程练习: exercises/README.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST Lab — 卷积神经网络教学实验")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--model", default="cnn", choices=["mlp", "cnn", "cnn_dropout"])
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument("--lr", type=float, default=DEFAULT_LR)
    p_train.add_argument("--weight-decay", type=float, default=0.0)
    p_train.add_argument("--device", default=DEFAULT_DEVICE)
    p_train.add_argument("--run-dir", default=None)
    p_train.add_argument("--augment", action="store_true")
    _add_data_flags(p_train)
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="预测单张图片")
    p_pred.add_argument("--image-path", required=True)
    p_pred.add_argument("--model", default="cnn")
    p_pred.add_argument("--checkpoint", default=CHECKPOINT_NAME)
    p_pred.add_argument("--out", default=None)
    p_pred.set_defaults(func=cmd_predict)

    p_pt = sub.add_parser("predict-test", help="在测试批次上预测并保存网格图")
    p_pt.add_argument("--model", default="cnn")
    p_pt.add_argument("--checkpoint", default=CHECKPOINT_NAME)
    _add_data_flags(p_pt)
    p_pt.set_defaults(func=cmd_predict_test)

    p_ev = sub.add_parser("evaluate", help="计算准确率与混淆矩阵")
    p_ev.add_argument("--model", default="cnn")
    p_ev.add_argument("--checkpoint", default=CHECKPOINT_NAME)
    p_ev.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_ev.add_argument("--device", default=DEFAULT_DEVICE)
    _add_data_flags(p_ev)
    p_ev.set_defaults(func=cmd_evaluate)

    p_vz = sub.add_parser("visualize", help="导出曲线 / 混淆矩阵 / 误分类 / 解释图")
    p_vz.add_argument(
        "--kind",
        required=True,
        choices=["curves", "confusion", "errors", "samples", "gradcam", "saliency", "features"],
    )
    p_vz.add_argument("--model", default="cnn")
    p_vz.add_argument("--checkpoint", default=CHECKPOINT_NAME)
    p_vz.add_argument("--run-dir", default=str(RUNS_DIR / "cnn"))
    _add_data_flags(p_vz)
    p_vz.set_defaults(func=cmd_visualize)

    p_quiz = sub.add_parser("quiz", help="随堂测验")
    p_quiz.add_argument("--answers", default=None, help="JSON 答案文件，非交互判分")
    p_quiz.set_defaults(func=cmd_quiz)

    p_lesson = sub.add_parser("lesson", help="列出课件")
    p_lesson.set_defaults(func=cmd_lesson)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
