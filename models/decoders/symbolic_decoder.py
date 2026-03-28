from typing import Tuple

import torch
from torch import nn


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 256):
        super().__init__()
        self.w_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.w_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # dec_hidden: [B, H], enc_outputs: [B, S, E]
        score = self.v(torch.tanh(self.w_enc(enc_outputs) + self.w_dec(dec_hidden).unsqueeze(1))).squeeze(-1)
        # Use smaller mask value for float16 compatibility
        mask_value = torch.finfo(score.dtype).min / 2
        score = score.masked_fill(~src_mask, mask_value)
        attn_weights = torch.softmax(score, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights


class SymbolicDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        enc_output_dim: int,
        pad_idx: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(enc_dim=enc_output_dim, dec_dim=hidden_dim, attn_dim=hidden_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim + enc_output_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim + enc_output_dim + embedding_dim, vocab_size)

    def forward_step(
        self,
        token_t: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        emb = self.dropout(self.embedding(token_t)).unsqueeze(1)
        dec_hidden = hidden[0][-1]
        context, attn = self.attention(dec_hidden, enc_outputs, src_mask)
        lstm_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, hidden = self.lstm(lstm_in, hidden)
        out = out.squeeze(1)
        logits = self.out(torch.cat([out, context, emb.squeeze(1)], dim=-1))
        return logits, hidden, attn
