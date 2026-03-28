from typing import List

import numpy as np
import sympy as sp


x = sp.Symbol("x")


def _coeff_vector(expr_text: str, order: int = 4):
    try:
        expr = sp.sympify(expr_text)
        series = sp.series(expr, x, 0, order + 1).removeO().expand()
    except Exception:
        return None
    coeffs = []
    for p in range(order + 1):
        try:
            coeff = sp.N(series.coeff(x, p))
            coeffs.append(float(coeff))
        except Exception:
            return None
    return np.asarray(coeffs, dtype=np.float64)


def coefficient_mse(preds: List[str], refs: List[str], order: int = 4) -> float:
    if not preds:
        return 0.0
    errors = []
    for pred, ref in zip(preds, refs):
        pred_vec = _coeff_vector(pred, order=order)
        ref_vec = _coeff_vector(ref, order=order)
        if pred_vec is None or ref_vec is None:
            continue
        if np.any(~np.isfinite(pred_vec)) or np.any(~np.isfinite(ref_vec)):
            continue
        errors.append(float(np.mean((pred_vec - ref_vec) ** 2)))
    if not errors:
        return 0.0
    return float(np.mean(errors))