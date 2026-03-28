import math
import random
from typing import Dict, List, TypedDict

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from tokenizer.tokenizer import SymbolicTokenizer
from tokenizer.vocabulary import Vocabulary


class BatchItem(TypedDict):
    src_ids: torch.Tensor
    tgt_ids: torch.Tensor
    src_len: int
    tgt_len: int
    expression: str
    target: str


class TaylorDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str,
        tokenizer: SymbolicTokenizer,
        max_input_len: int = 80,
        max_target_len: int = 120,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> BatchItem:
        row = self.df.iloc[idx]
        expr = str(row["expression"])
        tgt = str(row["taylor"])
        src_ids = self.tokenizer.encode(expr)[: self.max_input_len]
        tgt_ids = self.tokenizer.encode(tgt)[: self.max_target_len]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_len": len(src_ids),
            "tgt_len": len(tgt_ids),
            "expression": expr,
            "target": tgt,
        }


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, lengths: List[int], batch_size: int, num_buckets: int = 8, shuffle: bool = True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_buckets = max(1, num_buckets)
        self.shuffle = shuffle
        self.indices = list(range(len(lengths)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        sorted_indices = sorted(self.indices, key=lambda i: self.lengths[i])
        bucket_size = max(1, math.ceil(len(sorted_indices) / self.num_buckets))
        buckets = [sorted_indices[i : i + bucket_size] for i in range(0, len(sorted_indices), bucket_size)]
        if self.shuffle:
            random.shuffle(buckets)
        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i : i + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def collate_batch(batch: List[BatchItem], pad_idx: int) -> Dict[str, object]:
    src = [item["src_ids"] for item in batch]
    tgt = [item["tgt_ids"] for item in batch]
    src_pad = pad_sequence(src, batch_first=True, padding_value=pad_idx)
    tgt_pad = pad_sequence(tgt, batch_first=True, padding_value=pad_idx)
    src_len = torch.tensor([item["src_len"] for item in batch], dtype=torch.long)
    tgt_len = torch.tensor([item["tgt_len"] for item in batch], dtype=torch.long)
    return {
        "src_ids": src_pad,
        "tgt_ids": tgt_pad,
        "src_len": src_len,
        "tgt_len": tgt_len,
        "expression": [item["expression"] for item in batch],
        "target": [item["target"] for item in batch],
    }


def build_vocab_from_csv(csv_path: str, max_size: int = 1000) -> Vocabulary:
    df = pd.read_csv(csv_path)
    tokenizer_vocab = Vocabulary(max_size=max_size)
    token_sequences: List[List[str]] = []
    temp_tokenizer = SymbolicTokenizer(tokenizer_vocab)
    for _, row in df.iterrows():
        token_sequences.append(temp_tokenizer.tokenize(row["expression"]))
        token_sequences.append(temp_tokenizer.tokenize(row["taylor"]))
    tokenizer_vocab.build(token_sequences)
    return tokenizer_vocab


def build_dataloaders(
    csv_path: str,
    batch_size: int,
    max_vocab_size: int = 1000,
    max_input_len: int = 80,
    max_target_len: int = 120,
    num_buckets: int = 8,
) -> tuple[Dict[str, DataLoader], SymbolicTokenizer, Vocabulary]:
    vocab = build_vocab_from_csv(csv_path, max_size=max_vocab_size)
    tokenizer = SymbolicTokenizer(vocab)

    datasets = {
        split: TaylorDataset(
            csv_path=csv_path,
            split=split,
            tokenizer=tokenizer,
            max_input_len=max_input_len,
            max_target_len=max_target_len,
        )
        for split in ["train", "val", "test"]
    }

    loaders: Dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        lengths = [int(ds[i]["src_len"]) for i in range(len(ds))]
        sampler = BucketBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            num_buckets=num_buckets,
            shuffle=(split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=lambda b, pad_idx=vocab.pad_idx: collate_batch(b, pad_idx),
        )
    return loaders, tokenizer, vocab
