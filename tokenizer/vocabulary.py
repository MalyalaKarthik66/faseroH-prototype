import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast


SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


class Vocabulary:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: List[str] = []
        for token in SPECIAL_TOKENS:
            self.add_token(token)

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx["<PAD>"]

    @property
    def sos_idx(self) -> int:
        return self.token_to_idx["<SOS>"]

    @property
    def eos_idx(self) -> int:
        return self.token_to_idx["<EOS>"]

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx["<UNK>"]

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def add_token(self, token: str) -> None:
        if token not in self.token_to_idx:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)

    def build(self, token_sequences: Iterable[Iterable[str]]) -> None:
        counter = Counter()
        for seq in token_sequences:
            counter.update(seq)
        budget = self.max_size - len(self.idx_to_token)
        for token, _ in counter.most_common(max(budget, 0)):
            self.add_token(token)

    def encode(self, tokens: List[str], add_boundaries: bool = True) -> List[int]:
        ids = [self.token_to_idx.get(tok, self.unk_idx) for tok in tokens]
        if add_boundaries:
            return [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, token_ids: List[int], remove_boundaries: bool = True) -> List[str]:
        tokens = [self.idx_to_token[i] if i < len(self.idx_to_token) else "<UNK>" for i in token_ids]
        if not remove_boundaries:
            return tokens
        return [t for t in tokens if t not in {"<SOS>", "<EOS>", "<PAD>", "<UNK>"}]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "idx_to_token": self.idx_to_token,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Vocabulary":
        max_size_raw = payload.get("max_size", 1000)
        idx_to_token_raw = payload.get("idx_to_token", [])
        vocab = cls(max_size=int(cast(int, max_size_raw)))
        vocab.idx_to_token = [str(tok) for tok in cast(List[Any], idx_to_token_raw)]
        vocab.token_to_idx = {t: i for i, t in enumerate(vocab.idx_to_token)}
        return vocab

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
