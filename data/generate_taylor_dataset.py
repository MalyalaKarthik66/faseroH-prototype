import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate_expression_trees import ExpressionGeneratorConfig, generate_expressions


x = sp.Symbol("x")


def is_valid_expr(expr: sp.Expr) -> bool:
    if expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return False
    return True


def taylor_expand(expr: sp.Expr, max_order: int = 4) -> sp.Expr:
    return sp.series(expr, x, 0, max_order + 1).removeO()


def _safe_numeric_eval(expr: sp.Expr, points: np.ndarray) -> bool:
    fn = sp.lambdify(x, expr, modules=["numpy"])
    try:
        with np.errstate(all="ignore"):
            values = np.asarray(fn(points), dtype=np.float64)
        if np.any(~np.isfinite(values)):
            return False
        return True
    except Exception:
        return False


def build_dataset(size: int, max_order: int, seed: int) -> pd.DataFrame:
    cfg = ExpressionGeneratorConfig(seed=seed)
    expressions = generate_expressions(size, cfg)
    rows: List[Dict[str, str]] = []
    points = np.linspace(-0.5, 0.5, 11)
    for expr in tqdm(expressions, desc="Generating Taylor data"):
        try:
            series_expr = taylor_expand(expr, max_order=max_order)
        except Exception:
            continue
        if not is_valid_expr(expr) or not is_valid_expr(series_expr):
            continue
        if not _safe_numeric_eval(expr, points):
            continue
        if not _safe_numeric_eval(series_expr, points):
            continue
        rows.append(
            {
                "expression": str(expr),
                "taylor": str(series_expr),
            }
        )
    return pd.DataFrame(rows)


def split_dataset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, shuffle=True)
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate symbolic Taylor expansion dataset")
    parser.add_argument("--size", type=int, default=20000)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/taylor_dataset.csv")
    parser.add_argument("--meta", type=str, default="data/taylor_dataset_meta.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset(size=args.size, max_order=args.max_order, seed=args.seed)
    if df.empty:
        raise RuntimeError("Generated dataset is empty; adjust generation parameters.")
    df = split_dataset(df, seed=args.seed)
    df.to_csv(output_path, index=False)
    meta = {
        "size": int(len(df)),
        "max_order": int(args.max_order),
        "seed": int(args.seed),
        "split_counts": df["split"].value_counts().to_dict(),
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved dataset to {output_path} with {len(df)} samples")


if __name__ == "__main__":
    main()
