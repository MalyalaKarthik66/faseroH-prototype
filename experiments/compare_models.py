import argparse
import json
from pathlib import Path


def load_metrics(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def row(model_name: str, payload):
    if not payload:
        return f"| {model_name} | N/A | N/A | N/A | N/A | N/A |"
    test = payload.get("test", {})
    return (
        f"| {model_name} | {fmt(test.get('token_accuracy'))} | {fmt(test.get('sequence_accuracy'))} | "
        f"{fmt(test.get('bleu'))} | {fmt(test.get('edit_distance'))} | {fmt(test.get('semantic_equivalence'))} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LSTM and Transformer metrics")
    parser.add_argument("--lstm", type=str, default="results/lstm/metrics.json")
    parser.add_argument("--transformer", type=str, default="results/transformer/metrics.json")
    parser.add_argument("--output", type=str, default="results/model_comparison.md")
    args = parser.parse_args()

    lstm_metrics = load_metrics(Path(args.lstm))
    transformer_metrics = load_metrics(Path(args.transformer))

    lines = [
        "# Model Comparison",
        "",
        "| Model | Token Accuracy | Sequence Accuracy | BLEU | Edit Distance | Semantic Equivalence |",
        "|---|---:|---:|---:|---:|---:|",
        row("LSTM", lstm_metrics),
        row("Transformer", transformer_metrics),
        "",
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison table to {out}")


if __name__ == "__main__":
    main()
