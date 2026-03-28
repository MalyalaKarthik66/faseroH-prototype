import argparse
import json
from typing import Dict, List

import numpy as np
import sympy as sp


x = sp.Symbol("x")


def expression_to_histogram(
    expression: str,
    num_samples: int = 2000,
    bins: int = 64,
    x_min: float = -2.0,
    x_max: float = 2.0,
) -> List[float]:
    expr = sp.sympify(expression)
    fn = sp.lambdify(x, expr, modules=["numpy"])
    xs = np.linspace(x_min, x_max, num_samples)
    ys = np.asarray(fn(xs), dtype=np.float64)
    ys = ys[np.isfinite(ys)]
    if ys.size == 0:
        return [0.0] * bins
    hist, _ = np.histogram(ys, bins=bins, density=True)
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist.tolist()


def build_histogram_record(expression: str, **kwargs) -> Dict[str, object]:
    return {
        "expression": expression,
        "histogram": expression_to_histogram(expression, **kwargs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate histogram input from symbolic expression")
    parser.add_argument("--expression", type=str, required=True)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=2000)
    args = parser.parse_args()
    record = build_histogram_record(args.expression, bins=args.bins, num_samples=args.num_samples)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
