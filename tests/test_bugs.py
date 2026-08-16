from mnist_lab.bugs import CASES, grade_diagnosis, list_cases, run_bug_case


def test_all_cases_registered():
    ids = {c.id for c in list_cases()}
    assert ids == {"exploding_lr", "shuffled_labels", "forgot_zero_grad", "tune_on_test", "no_invert"}


def test_grade_diagnosis():
    assert grade_diagnosis("exploding_lr", "B") is True
    assert grade_diagnosis("exploding_lr", "A") is False


def test_exploding_lr_loss_not_tiny(tmp_path):
    payload = run_bug_case("exploding_lr", tmp_path / "exp")
    losses = payload["history"]["train_loss"]
    assert max(losses) > 1.0


def test_shuffled_labels_acc_near_chance(tmp_path):
    payload = run_bug_case("shuffled_labels", tmp_path / "shuf")
    accs = payload["history"]["val_acc"]
    assert max(accs) < 0.5


def test_forgot_zero_grad_records_steps(tmp_path):
    payload = run_bug_case("forgot_zero_grad", tmp_path / "zg")
    assert len(payload["history"]["train_loss"]) == 8


def test_no_invert_means(tmp_path):
    payload = run_bug_case("no_invert", tmp_path / "inv")
    assert payload["white_mean"] > 0.5
    assert payload["inverted_mean"] < 0.5
