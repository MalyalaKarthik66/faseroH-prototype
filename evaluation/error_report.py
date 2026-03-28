import argparse
import json
from pathlib import Path

import sympy as sp


x = sp.Symbol("x")


def classify_error(predicted_taylor: str, target_taylor: str) -> str:
    try:
        p = sp.series(sp.sympify(predicted_taylor), x, 0, 5).removeO().expand()
        t = sp.series(sp.sympify(target_taylor), x, 0, 5).removeO().expand()
    except Exception:
        return "wrong_function_structure"

    powers = list(range(5))
    p_coeff = [sp.N(p.coeff(x, i)) for i in powers]
    t_coeff = [sp.N(t.coeff(x, i)) for i in powers]

    missing = any(float(abs(t_coeff[i])) > 1e-9 and float(abs(p_coeff[i])) <= 1e-9 for i in powers)
    extra = any(float(abs(t_coeff[i])) <= 1e-9 and float(abs(p_coeff[i])) > 1e-9 for i in powers)

    if missing:
        return "missing_term"
    if extra:
        return "extra_term"

    coeff_mismatch = any(float(abs(p_coeff[i] - t_coeff[i])) > 1e-6 for i in powers)
    if coeff_mismatch:
        return "wrong_coefficient"

    if sp.simplify(p - t) != 0:
        return "wrong_function_structure"

    return "correct"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate error analysis report from sample predictions")
    parser.add_argument("--input", type=str, default="results/sample_predictions.json")
    parser.add_argument("--output", type=str, default="results/error_analysis.md")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    rows = json.loads(in_path.read_text(encoding="utf-8"))

    counts = {
        "wrong_coefficient": 0,
        "missing_term": 0,
        "extra_term": 0,
        "wrong_function_structure": 0,
        "correct": 0,
    }

    for row in rows:
        label = classify_error(row.get("predicted_taylor", ""), row.get("target_taylor", ""))
        counts[label] = counts.get(label, 0) + 1

    total = max(1, len(rows))

    lines = [
        "# Error Analysis Report",
        "",
        f"Total analyzed samples: {len(rows)}",
        "",
        "| Error Type | Count | Percentage |",
        "|---|---:|---:|",
    ]

    for key in ["wrong_coefficient", "missing_term", "extra_term", "wrong_function_structure", "correct"]:
        count = counts.get(key, 0)
        pct = 100.0 * count / total
        lines.append(f"| {key} | {count} | {pct:.2f}% |")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved error analysis report to {out}")


if __name__ == "__main__":
    main()
