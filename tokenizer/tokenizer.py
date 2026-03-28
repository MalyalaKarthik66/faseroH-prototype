from typing import List

from tokenizer.prefix_converter import infix_to_prefix, prefix_to_infix
from tokenizer.vocabulary import Vocabulary


class SymbolicTokenizer:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def tokenize(self, expression: str) -> List[str]:
        return infix_to_prefix(expression)

    def detokenize(self, tokens: List[str]) -> str:
        return prefix_to_infix(tokens)

    def encode(self, expression: str) -> List[int]:
        return self.vocab.encode(self.tokenize(expression), add_boundaries=True)

    def decode(self, token_ids: List[int]) -> str:
        tokens = self.vocab.decode(token_ids, remove_boundaries=True)
        if not tokens:
            return "0"
        return self.detokenize(tokens)
