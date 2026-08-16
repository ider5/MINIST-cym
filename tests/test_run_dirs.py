from pathlib import Path

from mnist_lab.run_dirs import new_run_dir


def test_new_run_dir_uses_model_and_stamp(tmp_path):
    a = new_run_dir("cnn", root=tmp_path, stamp="20260816-010203")
    b = new_run_dir("cnn", root=tmp_path, stamp="20260816-010204")
    assert a.name == "cnn_20260816-010203"
    assert b != a
    assert a.is_dir() and b.is_dir()
    assert list(tmp_path.iterdir())
    assert Path(a).parent == tmp_path
