import torch

from models.lstm_seq2seq import LSTMSeq2Seq
from models.transformer_seq2seq import TransformerSeq2Seq


def _tiny_batch():
    src_ids = torch.tensor([[1, 5, 6, 2, 0], [1, 7, 2, 0, 0]], dtype=torch.long)
    src_len = torch.tensor([4, 3], dtype=torch.long)
    tgt_ids = torch.tensor([[1, 8, 9, 2], [1, 10, 2, 0]], dtype=torch.long)
    return src_ids, src_len, tgt_ids


def test_lstm_forward_output_shape():
    src_ids, src_len, tgt_ids = _tiny_batch()
    model = LSTMSeq2Seq(
        vocab_size=32,
        pad_idx=0,
        embedding_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    )
    logits = model(src_ids, src_len, tgt_ids, teacher_forcing_ratio=1.0)
    assert logits.shape == (2, 4, 32)


def test_transformer_forward_and_decode():
    src_ids, src_len, tgt_ids = _tiny_batch()
    model = TransformerSeq2Seq(
        vocab_size=32,
        pad_idx=0,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    logits = model(src_ids, src_len, tgt_ids, teacher_forcing_ratio=1.0)
    assert logits.shape == (2, 4, 32)

    decoded = model.greedy_decode(src_ids, src_len, sos_idx=1, eos_idx=2, max_len=6)
    assert len(decoded) == src_ids.size(0)
    assert all(seq[0] == 1 for seq in decoded)
