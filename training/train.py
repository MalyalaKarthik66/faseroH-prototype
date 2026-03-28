import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader import build_dataloaders
from models.lstm_seq2seq import LSTMSeq2Seq
from models.transformer_seq2seq import TransformerSeq2Seq
from training.trainer import Trainer, TrainerConfig, auto_device, set_seed


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg, vocab_size: int, pad_idx: int):
    model_name = cfg["model"].lower()
    params = cfg["model_params"]
    if model_name == "lstm":
        return LSTMSeq2Seq(vocab_size=vocab_size, pad_idx=pad_idx, **params)
    if model_name == "transformer":
        return TransformerSeq2Seq(vocab_size=vocab_size, pad_idx=pad_idx, **params)
    raise ValueError(f"Unsupported model: {cfg['model']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train seq2seq symbolic model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    loaders, tokenizer, vocab = build_dataloaders(
        csv_path=ds_cfg["path"],
        batch_size=int(tr_cfg["batch_size"]),
        max_vocab_size=1000,
        max_input_len=int(ds_cfg.get("max_input_len", 80)),
        max_target_len=int(ds_cfg.get("max_target_len", 120)),
        num_buckets=int(ds_cfg.get("num_buckets", 8)),
    )
    vocab.save(str(results_dir / "vocab.json"))

    device = auto_device(str(cfg.get("device", "auto")))
    print(f"[DEVICE] Using device: {device}")
    model = build_model(cfg, vocab_size=len(vocab), pad_idx=vocab.pad_idx)
    trainer = Trainer(
        model=model,
        pad_idx=vocab.pad_idx,
        tokenizer=tokenizer,
        device=device,
        cfg=TrainerConfig(
            epochs=int(tr_cfg["epochs"]),
            learning_rate=float(tr_cfg["learning_rate"]),
            weight_decay=float(tr_cfg["weight_decay"]),
            grad_clip=float(tr_cfg["grad_clip"]),
            early_stopping_patience=int(tr_cfg["early_stopping_patience"]),
            teacher_forcing_start=float(tr_cfg["teacher_forcing_start"]),
            teacher_forcing_end=float(tr_cfg["teacher_forcing_end"]),
            log_every=int(tr_cfg["log_every"]),
            beam_width=int(tr_cfg.get("beam_width", 1)),
            length_penalty=float(tr_cfg.get("length_penalty", 0.6)),
            label_smoothing=float(tr_cfg.get("label_smoothing", 0.0)),
            use_amp=bool(tr_cfg.get("use_amp", True)),
        ),
        results_dir=str(results_dir),
    )
    fit_info = trainer.fit(loaders["train"], loaders["val"])

    (results_dir / "metrics.json").write_text(
        json.dumps({"fit": fit_info}, indent=2),
        encoding="utf-8",
    )
    print("[TRAIN] Fit metrics written to disk.")
    print(json.dumps({"fit": fit_info}, indent=2))

    if args.dry_run:
        print("Dry run complete. Waiting for user confirmation to proceed with full training.")


if __name__ == "__main__":
    main()
