from evaluation.coefficient_metrics import coefficient_mse


def test_coefficient_mse_perfect_match_is_zero():
    preds = ["1 + x + x**2", "sin(x)"]
    refs = ["1 + x + x**2", "sin(x)"]
    assert coefficient_mse(preds, refs, order=4) == 0.0


def test_coefficient_mse_skips_invalid_rows():
    preds = ["1 + x", "INVALID_EXPR"]
    refs = ["1 + x", "x + 2"]
    mse = coefficient_mse(preds, refs, order=4)
    assert mse >= 0.0
