import argparse
import json
from pathlib import Path


def esc(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative markdown examples from sample predictions")
    parser.add_argument("--input", type=str, default="results/sample_predictions.json")
    parser.add_argument("--output", type=str, default="results/qualitative_examples.md")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing sample prediction file: {data_path}")

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    rows = payload[: max(1, args.limit)]

    lines = [
        "# Qualitative Prediction Examples",
        "",
        "| Input Function | Target Taylor | Predicted Taylor | Correct? |",
        "|---|---|---|---|",
    ]

    for row in rows:
        correct = "Yes" if bool(row.get("semantic_equivalent", False)) else "No"
        lines.append(
            "| "
            + esc(row.get("input_function", ""))
            + " | "
            + esc(row.get("target_taylor", ""))
            + " | "
            + esc(row.get("predicted_taylor", ""))
            + " | "
            + correct
            + " |"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved qualitative examples to {out}")


if __name__ == "__main__":
    main()
