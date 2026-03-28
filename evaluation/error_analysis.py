import json
from pathlib import Path
from typing import Dict, List

from evaluation.semantic_metrics import symbolic_equivalent


def classify_error(pred: str, target: str) -> str:
    if pred.strip() == target.strip():
        return "exact_match"
    if symbolic_equivalent(pred, target):
        return "semantic_match"
    return "semantic_mismatch"


def build_error_report(predictions: List[str], targets: List[str]) -> Dict[str, object]:
    rows = []
    counts = {"exact_match": 0, "semantic_match": 0, "semantic_mismatch": 0}
    for p, t in zip(predictions, targets):
        label = classify_error(p, t)
        counts[label] += 1
        rows.append({"prediction": p, "target": t, "label": label})
    return {"summary": counts, "rows": rows}


def save_error_report(path: str, predictions: List[str], targets: List[str]) -> None:
    report = build_error_report(predictions, targets)
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
