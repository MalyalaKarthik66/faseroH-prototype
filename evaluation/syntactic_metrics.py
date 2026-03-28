from collections import Counter
from typing import Dict, List

from tokenizer.prefix_converter import infix_to_prefix


def _tokenize(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if " " in stripped:
        return stripped.split()
    try:
        return infix_to_prefix(stripped)
    except Exception:
        return stripped.split()


def token_accuracy(preds: List[str], refs: List[str]) -> float:
    correct = 0
    total = 0
    for p, r in zip(preds, refs):
        pt = _tokenize(p)
        rt = _tokenize(r)
        for i in range(min(len(pt), len(rt))):
            correct += int(pt[i] == rt[i])
        total += max(len(rt), 1)
    return correct / total if total else 0.0


def sequence_accuracy(preds: List[str], refs: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(int(p.strip() == r.strip()) for p, r in zip(preds, refs)) / len(preds)


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def bleu_score(preds: List[str], refs: List[str], max_n: int = 4) -> float:
    if not preds:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        match = 0
        total = 0
        for p, r in zip(preds, refs):
            pt = _tokenize(p)
            rt = _tokenize(r)
            p_counts = _ngram_counts(pt, n)
            r_counts = _ngram_counts(rt, n)
            match += sum(min(cnt, r_counts[ng]) for ng, cnt in p_counts.items())
            total += max(sum(p_counts.values()), 1)
        precisions.append((match + 1) / (total + 1))
    score = 1.0
    for p in precisions:
        score *= p
    score = score ** (1 / max_n)
    pred_len = sum(len(_tokenize(p)) for p in preds)
    ref_len = sum(len(_tokenize(r)) for r in refs)
    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else pow(2.718281828, 1 - ref_len / pred_len)
    return bp * score


def edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def mean_edit_distance(preds: List[str], refs: List[str]) -> float:
    if not preds:
        return 0.0
    d = [edit_distance(_tokenize(p), _tokenize(r)) for p, r in zip(preds, refs)]
    return sum(d) / len(d)


def compute_syntactic_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    return {
        "token_accuracy": token_accuracy(preds, refs),
        "sequence_accuracy": sequence_accuracy(preds, refs),
        "bleu": bleu_score(preds, refs),
        "edit_distance": mean_edit_distance(preds, refs),
    }
