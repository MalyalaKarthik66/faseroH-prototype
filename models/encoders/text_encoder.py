from typing import Tuple

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj_h = nn.Linear(out_dim, hidden_dim)
        self.proj_c = nn.Linear(out_dim, hidden_dim)

    def forward(self, src_ids: torch.Tensor, src_len: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(src_ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (h_n, c_n) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            c_last = torch.cat([c_n[-2], c_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
            c_last = c_n[-1]
        h0 = torch.tanh(self.proj_h(h_last)).unsqueeze(0)
        c0 = torch.tanh(self.proj_c(c_last)).unsqueeze(0)
        return outputs, (h0, c0)
