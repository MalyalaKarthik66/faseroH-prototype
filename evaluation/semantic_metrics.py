from typing import List
import warnings

import numpy as np
import sympy as sp


x = sp.Symbol("x")


def _to_sympy(expr_text: str):
    try:
        return sp.sympify(expr_text)
    except Exception:
        return None


def symbolic_equivalent(pred: str, target: str) -> bool:
    p = _to_sympy(pred)
    t = _to_sympy(target)
    if p is None or t is None:
        return False
    try:
        if sp.simplify(p - t) == 0:
            return True
    except Exception:
        pass
    return numeric_equivalent(pred, target)


def numeric_equivalent(pred: str, target: str, tol: float = 1e-4) -> bool:
    p = _to_sympy(pred)
    t = _to_sympy(target)
    if p is None or t is None:
        return False
    pts = np.linspace(-1.0, 1.0, 100)
    pts = pts[np.abs(pts) >= 1e-3]
    if pts.size == 0:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                pf = np.asarray(sp.lambdify(x, p, modules=["numpy"])(pts), dtype=np.float64)
                tf = np.asarray(sp.lambdify(x, t, modules=["numpy"])(pts), dtype=np.float64)
    except Exception:
        return False
    if np.any(~np.isfinite(pf)) or np.any(~np.isfinite(tf)):
        return False
    return float(np.max(np.abs(pf - tf))) < tol


def semantic_equivalence_rate(preds: List[str], refs: List[str]) -> float:
    if not preds:
        return 0.0
    ok = [symbolic_equivalent(p, r) for p, r in zip(preds, refs)]
    return float(sum(ok)) / len(ok)
