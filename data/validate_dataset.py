import argparse

import numpy as np
import pandas as pd
import sympy as sp


x = sp.Symbol("x")


def validate_row(expr_text: str, taylor_text: str) -> bool:
    try:
        expr = sp.sympify(expr_text)
        taylor = sp.sympify(taylor_text)
    except Exception:
        return False
    if expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return False
    if taylor.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return False
    pts = np.linspace(-0.4, 0.4, 9)
    expr_fn = sp.lambdify(x, expr, modules=["numpy"])
    taylor_fn = sp.lambdify(x, taylor, modules=["numpy"])
    try:
        ev = np.asarray(expr_fn(pts), dtype=np.float64)
        tv = np.asarray(taylor_fn(pts), dtype=np.float64)
    except Exception:
        return False
    if np.any(~np.isfinite(ev)) or np.any(~np.isfinite(tv)):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated Taylor dataset")
    parser.add_argument("--input", type=str, default="data/taylor_dataset.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    valid = df.apply(lambda r: validate_row(r["expression"], r["taylor"]), axis=1)
    total = len(df)
    valid_count = int(valid.sum())
    print(f"valid_rows={valid_count}/{total}")
    if valid_count != total:
        invalid_idx = df.index[~valid].tolist()[:20]
        print("invalid_indices_preview=", invalid_idx)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
