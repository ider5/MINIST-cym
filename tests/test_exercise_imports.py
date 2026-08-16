"""学生练习不得直接调用教学库里的现成答案。"""

from __future__ import annotations

import ast
from pathlib import Path

STUDENT_FILES = [
    Path("exercises/ex01_metrics.py"),
    Path("exercises/ex02_train_step.py"),
    Path("exercises/ex03_conv.py"),
    Path("exercises/ex04_overfit.py"),
    Path("exercises/ex05_cross_entropy.py"),
]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_student_files_do_not_import_mnist_lab_or_sklearn():
    for path in STUDENT_FILES:
        mods = imported_modules(path)
        assert "mnist_lab" not in mods, path
        assert "sklearn" not in mods, path


def test_ex03_solution_does_not_call_library_conv():
    text = Path("exercises/solutions/ex03_conv.py").read_text(encoding="utf-8")
    assert "mnist_lab" not in text
    assert "conv2d_numpy" not in text
