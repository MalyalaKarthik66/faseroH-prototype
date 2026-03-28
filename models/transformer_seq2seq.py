import math
from typing import List

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, src_ids: torch.Tensor, src_len: torch.Tensor, tgt_ids: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        del src_len, teacher_forcing_ratio
        src_key_padding_mask = src_ids.eq(self.pad_idx)
        tgt_in = tgt_ids[:, :-1]
        src = self.pos(self.src_emb(src_ids) * math.sqrt(self.d_model))
        tgt = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        tgt_mask = self._causal_mask(tgt_in.size(1), src_ids.device)
        out = self.transformer(
            src,
            tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_in.eq(self.pad_idx),
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        logits = self.proj(out)
        # Align to target length for shared trainer interface.
        pad = torch.zeros(logits.size(0), 1, logits.size(-1), device=logits.device)
        return torch.cat([pad, logits], dim=1)

    @torch.no_grad()
    def greedy_decode(self, src_ids: torch.Tensor, src_len: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 120) -> List[List[int]]:
        del src_len
        batch_size = src_ids.size(0)
        src_key_padding_mask = src_ids.eq(self.pad_idx)
        memory = self.transformer.encoder(self.pos(self.src_emb(src_ids) * math.sqrt(self.d_model)), src_key_padding_mask=src_key_padding_mask)
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src_ids.device)
        for _ in range(max_len - 1):
            tgt = self.pos(self.tgt_emb(ys) * math.sqrt(self.d_model))
            tgt_mask = self._causal_mask(ys.size(1), src_ids.device)
            out = self.transformer.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=ys.eq(self.pad_idx),
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_tok = self.proj(out[:, -1]).argmax(dim=-1)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | next_tok.eq(eos_idx)
            if finished.all():
                break
        return ys.tolist()

    @torch.no_grad()
    def beam_decode(
        self,
        src_ids: torch.Tensor,
        src_len: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 120,
        beam_width: int = 5,
        length_penalty: float = 0.6,
    ) -> List[List[int]]:
        del src_len
        def normalized_score(total_log_prob: float, seq_len: int, alpha: float = length_penalty) -> float:
            return total_log_prob / (max(seq_len, 1) ** alpha)

        out = []
        for i in range(src_ids.size(0)):
            src = src_ids[i : i + 1]
            src_key_padding_mask = src.eq(self.pad_idx)
            memory = self.transformer.encoder(
                self.pos(self.src_emb(src) * math.sqrt(self.d_model)),
                src_key_padding_mask=src_key_padding_mask,
            )
            beams = [([sos_idx], 0.0)]
            completed = []
            for _ in range(max_len - 1):
                candidates = []
                for seq, score in beams:
                    if seq[-1] == eos_idx:
                        completed.append((seq, score))
                        continue
                    ys = torch.tensor([seq], dtype=torch.long, device=src.device)
                    tgt = self.pos(self.tgt_emb(ys) * math.sqrt(self.d_model))
                    tgt_mask = self._causal_mask(ys.size(1), src.device)
                    dec_out = self.transformer.decoder(
                        tgt,
                        memory,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=ys.eq(self.pad_idx),
                        memory_key_padding_mask=src_key_padding_mask,
                    )
                    log_probs = torch.log_softmax(self.proj(dec_out[:, -1]), dim=-1)
                    topk = torch.topk(log_probs, k=beam_width, dim=-1)
                    for k in range(beam_width):
                        tok = int(topk.indices[0, k].item())
                        tok_score = float(topk.values[0, k].item())
                        candidates.append((seq + [tok], score + tok_score))
                if not candidates:
                    break
                candidates.sort(key=lambda x: normalized_score(x[1], len(x[0])), reverse=True)
                beams = candidates[:beam_width]
            final = completed if completed else beams
            final.sort(key=lambda x: normalized_score(x[1], len(x[0])), reverse=True)
            out.append(final[0][0])
        return out
