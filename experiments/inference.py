import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.lstm_seq2seq import LSTMSeq2Seq
from models.transformer_seq2seq import TransformerSeq2Seq
from tokenizer.tokenizer import SymbolicTokenizer
from tokenizer.vocabulary import Vocabulary
from training.trainer import auto_device


def load_model(config_path: str, vocab: Vocabulary, device: torch.device):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    params = cfg["model_params"]
    if cfg["model"].lower() == "lstm":
        model = LSTMSeq2Seq(vocab_size=len(vocab), pad_idx=vocab.pad_idx, **params)
    else:
        model = TransformerSeq2Seq(vocab_size=len(vocab), pad_idx=vocab.pad_idx, **params)
    ckpt = torch.load("results/checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt)
    return model.to(device).eval(), cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Single expression inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--expression", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = auto_device(str(cfg.get("device", "auto")))
    vocab = Vocabulary.load("results/vocab.json")
    tokenizer = SymbolicTokenizer(vocab)
    model, cfg = load_model(args.config, vocab, device)

    src = torch.tensor([tokenizer.encode(args.expression)], dtype=torch.long, device=device)
    src_len = torch.tensor([src.size(1)], dtype=torch.long, device=device)
    beam_width = int(cfg.get("training", {}).get("beam_width", 1))
    length_penalty = float(cfg.get("training", {}).get("length_penalty", 0.6))
    if beam_width > 1 and hasattr(model, "beam_decode"):
        out = model.beam_decode(src, src_len, vocab.sos_idx, vocab.eos_idx, max_len=120, beam_width=beam_width, length_penalty=length_penalty)[0]
    else:
        out = model.greedy_decode(src, src_len, vocab.sos_idx, vocab.eos_idx, max_len=120)[0]
    prediction = tokenizer.decode(out)
    print(json.dumps({"expression": args.expression, "prediction": prediction}, indent=2))


if __name__ == "__main__":
    main()
