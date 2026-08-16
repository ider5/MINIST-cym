from mnist_lab.models import build_model
from mnist_lab.shapes import cnn_expected_shapes, format_trace, trace_shapes


def test_cnn_shape_trace_matches_lesson_ledger():
    model = build_model("cnn")
    batch = 2
    rows = trace_shapes(model, batch=batch)
    expected = cnn_expected_shapes(batch=batch)
    by_name = dict(rows)
    for name, shape in expected:
        assert by_name[name] == shape, (name, by_name.get(name), shape)


def test_mlp_trace_ends_with_10_logits():
    rows = trace_shapes(build_model("mlp"), batch=3)
    assert rows[0] == ("input", (3, 1, 28, 28))
    assert rows[-1] == ("logits", (3, 10))
    text = format_trace(rows)
    assert "logits(3, 10)" in text.replace(" ", "") or "logits(3, 10)" in text
