import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from evaluation.semantic_metrics import semantic_equivalence_rate
from evaluation.syntactic_metrics import compute_syntactic_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device(preference: str = "auto") -> torch.device:
    pref = preference.lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainerConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    early_stopping_patience: int
    teacher_forcing_start: float
    teacher_forcing_end: float
    log_every: int
    beam_width: int
    length_penalty: float = 0.6
    label_smoothing: float = 0.0
    use_amp: bool = True


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        pad_idx: int,
        tokenizer,
        device: torch.device,
        cfg: TrainerConfig,
        results_dir: str = "results",
    ):
        self.model = model.to(device)
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = cfg
        self.results_dir = Path(results_dir)
        self.ckpt_dir = self.results_dir / "checkpoints"
        self.log_dir = self.results_dir / "training_logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=cfg.label_smoothing)
        self.use_amp = cfg.use_amp and device.type == "cuda"
        self.scaler: Optional[GradScaler] = GradScaler("cuda") if self.use_amp else None

    def _teacher_forcing_ratio(self, epoch_idx: int) -> float:
        if self.cfg.epochs <= 1:
            return self.cfg.teacher_forcing_end
        alpha = epoch_idx / max(1, self.cfg.epochs - 1)
        return self.cfg.teacher_forcing_start * (1 - alpha) + self.cfg.teacher_forcing_end * alpha

    def _compute_loss(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits[:, 1:].reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

    def train_epoch(self, loader, epoch_idx: int) -> float:
        self.model.train()
        tf_ratio = self._teacher_forcing_ratio(epoch_idx)
        total_loss = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"Train {epoch_idx+1}"), start=1):
            src_ids = batch["src_ids"].to(self.device)
            tgt_ids = batch["tgt_ids"].to(self.device)
            src_len = batch["src_len"].to(self.device)
            self.optimizer.zero_grad()
            if self.use_amp:
                assert self.scaler is not None
                with autocast("cuda", dtype=torch.float16):
                    logits = self.model(src_ids, src_len, tgt_ids, teacher_forcing_ratio=tf_ratio)
                    loss = self._compute_loss(logits, tgt_ids)
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(src_ids, src_len, tgt_ids, teacher_forcing_ratio=tf_ratio)
                loss = self._compute_loss(logits, tgt_ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            total_loss += float(loss.item())
            if step % self.cfg.log_every == 0:
                self.writer.add_scalar("train/step_loss", float(loss.item()), epoch_idx * len(loader) + step)
        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def evaluate_loss(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            src_ids = batch["src_ids"].to(self.device)
            tgt_ids = batch["tgt_ids"].to(self.device)
            src_len = batch["src_len"].to(self.device)
            logits = self.model(src_ids, src_len, tgt_ids, teacher_forcing_ratio=0.0)
            loss = self._compute_loss(logits, tgt_ids)
            total_loss += float(loss.item())
        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def predict_loader(self, loader, max_len: int = 120) -> Dict[str, List[str]]:
        self.model.eval()
        preds: List[str] = []
        targets: List[str] = []
        inputs: List[str] = []
        for batch in tqdm(loader, desc="Decode", leave=False):
            src_ids = batch["src_ids"].to(self.device)
            src_len = batch["src_len"].to(self.device)
            if self.cfg.beam_width > 1 and hasattr(self.model, "beam_decode"):
                decoded = self.model.beam_decode(
                    src_ids,
                    src_len,
                    sos_idx=self.tokenizer.vocab.sos_idx,
                    eos_idx=self.tokenizer.vocab.eos_idx,
                    max_len=max_len,
                    beam_width=self.cfg.beam_width,
                    length_penalty=self.cfg.length_penalty,
                )
            else:
                decoded = self.model.greedy_decode(
                    src_ids,
                    src_len,
                    sos_idx=self.tokenizer.vocab.sos_idx,
                    eos_idx=self.tokenizer.vocab.eos_idx,
                    max_len=max_len,
                )
            for seq in decoded:
                preds.append(self.tokenizer.decode(seq))
            targets.extend(batch["target"])
            inputs.extend(batch["expression"])
        return {"predictions": preds, "targets": targets, "inputs": inputs}

    def fit(self, train_loader, val_loader) -> Dict[str, float]:
        best_val = float("inf")
        best_epoch = -1
        history = []
        patience = 0
        for epoch in range(self.cfg.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.evaluate_loss(val_loader)
            self.scheduler.step(val_loss)
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience = 0
                torch.save(self.model.state_dict(), self.ckpt_dir / "best_model.pt")
            else:
                patience += 1
                if patience >= self.cfg.early_stopping_patience:
                    break
        Path(self.log_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return {"best_val_loss": best_val, "best_epoch": best_epoch}

    @torch.no_grad()
    def evaluate(self, test_loader) -> Dict[str, float]:
        pred_bundle = self.predict_loader(test_loader)
        metrics = compute_syntactic_metrics(pred_bundle["predictions"], pred_bundle["targets"])
        metrics["semantic_equivalence"] = semantic_equivalence_rate(pred_bundle["predictions"], pred_bundle["targets"])
        return metrics
