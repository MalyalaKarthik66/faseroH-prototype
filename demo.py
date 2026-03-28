import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.lstm_seq2seq import LSTMSeq2Seq
from models.transformer_seq2seq import TransformerSeq2Seq
from tokenizer.tokenizer import SymbolicTokenizer
from tokenizer.vocabulary import Vocabulary
from training.trainer import auto_device


def load_model(config_path: str, checkpoint_path: str, vocab: Vocabulary, device: torch.device):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    params = cfg["model_params"]

    if cfg["model"].lower() == "lstm":
        model = LSTMSeq2Seq(vocab_size=len(vocab), pad_idx=vocab.pad_idx, **params)
    else:
        model = TransformerSeq2Seq(vocab_size=len(vocab), pad_idx=vocab.pad_idx, **params)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    return model.to(device).eval(), cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="FASEROH symbolic demo")
    parser.add_argument("expression", type=str, help='Input expression, e.g. "sin(x)+x**2"')
    parser.add_argument("--config", type=str, default="configs/transformer.yaml")
    parser.add_argument("--vocab", type=str, default="results/vocab.json")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/best_model.pt")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = auto_device(str(cfg.get("device", "auto")))
    vocab = Vocabulary.load(args.vocab)
    tokenizer = SymbolicTokenizer(vocab)
    model, cfg = load_model(args.config, args.checkpoint, vocab, device)

    src_ids = torch.tensor([tokenizer.encode(args.expression)], dtype=torch.long, device=device)
    src_len = torch.tensor([src_ids.size(1)], dtype=torch.long, device=device)

    beam_width = int(cfg.get("training", {}).get("beam_width", 1))
    if beam_width > 1 and hasattr(model, "beam_decode"):
        out_seq = model.beam_decode(
            src_ids,
            src_len,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=120,
            beam_width=beam_width,
        )[0]
    else:
        out_seq = model.greedy_decode(
            src_ids,
            src_len,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=120,
        )[0]

    predicted = tokenizer.decode(out_seq)

    print(f"Input function: {args.expression}")
    print(f"Predicted Taylor expansion: {predicted}")
    print(json.dumps({"input_function": args.expression, "predicted_taylor": predicted}, indent=2))


if __name__ == "__main__":
    main()
