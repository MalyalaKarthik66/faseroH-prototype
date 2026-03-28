from evaluation.semantic_metrics import symbolic_equivalent
from evaluation.syntactic_metrics import compute_syntactic_metrics


def test_metric_correctness():
    preds = ["x + 1", "sin(x)"]
    refs = ["x + 1", "sin(x)"]
    m = compute_syntactic_metrics(preds, refs)
    assert m["sequence_accuracy"] == 1.0
    assert m["token_accuracy"] == 1.0


def test_symbolic_equivalence():
    assert symbolic_equivalent("x + x", "2*x")
    assert not symbolic_equivalent("x + 1", "x + 2")
