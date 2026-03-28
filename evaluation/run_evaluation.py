import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader import build_dataloaders
from evaluation.coefficient_metrics import coefficient_mse
from evaluation.semantic_metrics import semantic_equivalence_rate
from evaluation.syntactic_metrics import compute_syntactic_metrics
from models.lstm_seq2seq import LSTMSeq2Seq
from models.transformer_seq2seq import TransformerSeq2Seq
from training.trainer import Trainer, TrainerConfig, auto_device, set_seed


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_model(cfg: dict, vocab_size: int, pad_idx: int):
    model_name = str(cfg["model"]).lower()
    params = cfg["model_params"]
    if model_name == "lstm":
        return LSTMSeq2Seq(vocab_size=vocab_size, pad_idx=pad_idx, **params)
    if model_name == "transformer":
        return TransformerSeq2Seq(vocab_size=vocab_size, pad_idx=pad_idx, **params)
    raise ValueError(f"Unsupported model: {cfg['model']}")


def main() -> None:
    config_path = ROOT / "configs" / "transformer.yaml"
    results_dir = ROOT / "results" / "transformer_20k_30epochs"
    checkpoint_path = results_dir / "checkpoints" / "best_model.pt"
    output_path = results_dir / "metrics_full.json"

    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]
    loaders, tokenizer, vocab = build_dataloaders(
        csv_path=ds_cfg["path"],
        batch_size=int(tr_cfg["batch_size"]),
        max_vocab_size=1000,
        max_input_len=int(ds_cfg.get("max_input_len", 80)),
        max_target_len=int(ds_cfg.get("max_target_len", 120)),
        num_buckets=int(ds_cfg.get("num_buckets", 8)),
    )

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

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    pred_bundle = trainer.predict_loader(loaders["test"])
    test_metrics = compute_syntactic_metrics(pred_bundle["predictions"], pred_bundle["targets"])
    test_metrics["semantic_equivalence"] = semantic_equivalence_rate(
        pred_bundle["predictions"],
        pred_bundle["targets"],
    )
    test_metrics["coefficient_mse"] = coefficient_mse(
        pred_bundle["predictions"],
        pred_bundle["targets"],
        order=4,
    )

    fit_metrics = {}
    fit_metrics_path = results_dir / "metrics.json"
    if fit_metrics_path.exists():
        fit_metrics = json.loads(fit_metrics_path.read_text(encoding="utf-8")).get("fit", {})

    payload = {"fit": fit_metrics, "test": test_metrics}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()