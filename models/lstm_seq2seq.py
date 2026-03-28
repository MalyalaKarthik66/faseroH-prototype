from typing import List

import torch
from torch import nn

from models.decoders.symbolic_decoder import SymbolicDecoder
from models.encoders.text_encoder import TextEncoder


class LSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_idx=pad_idx,
        )
        enc_dim = hidden_dim * (2 if bidirectional else 1)
        self.decoder = SymbolicDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            enc_output_dim=enc_dim,
            pad_idx=pad_idx,
            dropout=dropout,
        )

    def forward(self, src_ids: torch.Tensor, src_len: torch.Tensor, tgt_ids: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        batch_size, tgt_len = tgt_ids.shape
        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size, device=src_ids.device)
        enc_outputs, hidden = self.encoder(src_ids, src_len)
        src_mask = src_ids[:, : enc_outputs.size(1)] != self.pad_idx

        token_t = tgt_ids[:, 0]
        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder.forward_step(token_t, hidden, enc_outputs, src_mask)
            outputs[:, t] = logits
            use_teacher = torch.rand(1, device=src_ids.device).item() < teacher_forcing_ratio
            token_t = tgt_ids[:, t] if use_teacher else logits.argmax(dim=-1)
        return outputs

    @torch.no_grad()
    def greedy_decode(self, src_ids: torch.Tensor, src_len: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 120) -> List[List[int]]:
        enc_outputs, hidden = self.encoder(src_ids, src_len)
        src_mask = src_ids[:, : enc_outputs.size(1)] != self.pad_idx
        batch_size = src_ids.size(0)
        token_t = torch.full((batch_size,), sos_idx, dtype=torch.long, device=src_ids.device)
        seqs = [[sos_idx] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src_ids.device)
        for _ in range(max_len - 1):
            logits, hidden, _ = self.decoder.forward_step(token_t, hidden, enc_outputs, src_mask)
            token_t = logits.argmax(dim=-1)
            for i in range(batch_size):
                if not finished[i]:
                    seqs[i].append(int(token_t[i].item()))
                    if token_t[i].item() == eos_idx:
                        finished[i] = True
            if finished.all():
                break
        return seqs

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
        # Beam search is applied per sample for clarity and easier extension.
        def normalized_score(total_log_prob: float, seq_len: int, alpha: float = length_penalty) -> float:
            return total_log_prob / (max(seq_len, 1) ** alpha)

        sequences: List[List[int]] = []
        for i in range(src_ids.size(0)):
            src = src_ids[i : i + 1]
            sl = src_len[i : i + 1]
            enc_outputs, hidden = self.encoder(src, sl)
            src_mask = src[:, : enc_outputs.size(1)] != self.pad_idx
            beams = [([sos_idx], 0.0, hidden)]
            completed = []
            for _ in range(max_len - 1):
                candidates = []
                for seq, score, h in beams:
                    if seq[-1] == eos_idx:
                        completed.append((seq, score))
                        continue
                    token_t = torch.tensor([seq[-1]], device=src_ids.device)
                    logits, next_h, _ = self.decoder.forward_step(token_t, h, enc_outputs, src_mask)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk = torch.topk(log_probs, k=beam_width, dim=-1)
                    for k in range(beam_width):
                        tok = int(topk.indices[0, k].item())
                        tok_score = float(topk.values[0, k].item())
                        candidates.append((seq + [tok], score + tok_score, next_h))
                if not candidates:
                    break
                candidates.sort(key=lambda x: normalized_score(x[1], len(x[0])), reverse=True)
                beams = candidates[:beam_width]
            final = completed if completed else [(s, sc) for s, sc, _ in beams]
            final.sort(key=lambda x: normalized_score(x[1], len(x[0])), reverse=True)
            sequences.append(final[0][0])
        return sequences
