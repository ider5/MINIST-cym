"""兼容旧入口：python main.py --mode train|predict|predict_test"""

from __future__ import annotations

import argparse

from mnist_lab.cli import main as lab_main


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST CNN - 训练和预测（兼容入口）")
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["train", "predict", "predict_test"],
        help="选择模式: train (训练), predict (预测单张图片), or predict_test (在测试集上预测)",
    )
    parser.add_argument("--image_path", type=str, help="需要预测的图片路径")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--toy", action="store_true", help="使用合成数据，便于无网络环境")
    args, extra = parser.parse_known_args()

    if args.mode == "train":
        argv = ["train", "--model", args.model]
        if args.toy:
            argv.append("--toy")
        raise SystemExit(lab_main(argv + extra))
    if args.mode == "predict":
        if not args.image_path:
            print("错误: 使用 'predict' 模式时必须提供 --image_path 参数。")
            raise SystemExit(2)
        argv = ["predict", "--image-path", args.image_path, "--model", args.model]
        raise SystemExit(lab_main(argv + extra))
    argv = ["predict-test", "--model", args.model]
    if args.toy:
        argv.append("--toy")
    raise SystemExit(lab_main(argv + extra))


if __name__ == "__main__":
    main()
