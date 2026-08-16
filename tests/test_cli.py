import json
import subprocess
import sys
from pathlib import Path

from mnist_lab.quiz import grade_answers, load_questions


def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "mnist_lab", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "train" in proc.stdout


def test_cli_lesson_and_quiz_file(tmp_path):
    lesson = subprocess.run(
        [sys.executable, "-m", "mnist_lab", "lesson"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert lesson.returncode == 0
    assert "00_tensors" in lesson.stdout

    questions = load_questions()
    answers = {q["id"]: q["answer"] for q in questions}
    ans_path = tmp_path / "ans.json"
    ans_path.write_text(json.dumps(answers), encoding="utf-8")
    quiz = subprocess.run(
        [sys.executable, "-m", "mnist_lab", "quiz", "--answers", str(ans_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert quiz.returncode == 0
    assert "13/13" in quiz.stdout


def test_cli_train_toy(tmp_path):
    import os

    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mnist_lab",
            "train",
            "--toy",
            "--epochs",
            "1",
            "--subset",
            "64",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "run" / "cnn2.pkl").is_file()
    assert (tmp_path / "run" / "history.json").is_file()


def test_cli_predict_uses_bundled_checkpoint(tmp_path):
    from PIL import Image
    import numpy as np
    import os

    img = np.zeros((28, 28), dtype=np.uint8)
    img[4:24, 12:16] = 255
    path = tmp_path / "one.png"
    Image.fromarray(img).save(path)
    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    proc = subprocess.run(
        [sys.executable, "-m", "mnist_lab", "predict", "--image-path", str(path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "预测结果" in proc.stdout


def test_legacy_main_help():
    proc = subprocess.run(
        [sys.executable, str(Path("main.py")), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0


def test_grade_partial_answers():
    questions = load_questions()
    score = grade_answers({"q1": "A"}, questions)
    assert score.correct < score.total
    assert score.total == 13
