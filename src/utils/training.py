"""
Training Utilities for Nested Learning
======================================

Provides:
- Training loop with support for continual learning
- Evaluation metrics
- Checkpoint management
- Logging utilities
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Optimizer
    optimizer: str = "adamw"  # 'adamw', 'm3', 'dgd'
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"  # 'cosine', 'linear', 'constant'
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1

    # Continual learning
    continual_learning: bool = False
    update_cms_during_eval: bool = False

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./outputs"

    # Mixed precision
    fp16: bool = False
    bf16: bool = False

    # Distributed training
    local_rank: int = -1

    # Seed
    seed: int = 42


class Trainer:
    """
    Trainer for Nested Learning models.

    Supports:
    - Standard training with gradient accumulation
    - Continual learning with CMS updates
    - Mixed precision training
    - Checkpoint saving/loading
    - Wandb/Tensorboard logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.collate_fn = collate_fn

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = None
        if train_dataset is not None:
            self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = None
        if config.fp16 or config.bf16:
            self.scaler = torch.cuda.amp.GradScaler()

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        elif self.config.optimizer == "m3":
            from ..core.optimizers import M3Optimizer

            return M3Optimizer(
                self.model.parameters(),
                lr=self.config.learning_rate,
                beta1=self.config.beta1,
                beta2=self.config.beta2,
            )
        elif self.config.optimizer == "dgd":
            from ..core.optimizers import DeltaGradientDescent

            return DeltaGradientDescent(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        num_training_steps = (
            len(self.train_dataset)
            // self.config.batch_size
            * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )

        warmup_steps = self.config.warmup_steps
        if warmup_steps == 0 and self.config.warmup_ratio > 0:
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        elif self.config.scheduler == "linear":
            return get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer)

    def train(self) -> dict[str, float]:
        """Run training loop."""
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        self.model.train()
        total_loss = 0.0

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for _step, batch in enumerate(progress_bar):
                loss = self._training_step(batch)
                epoch_loss += loss
                total_loss += loss

                # Update progress bar
                progress_bar.set_postfix(
                    {"loss": f"{loss:.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"}
                )

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: loss={loss:.4f}, "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

                # Evaluation
                if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                    metrics = self.evaluate()
                    logger.info(f"Eval metrics: {metrics}")
                    self.model.train()

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

            logger.info(
                f"Epoch {epoch + 1} finished. Avg loss: {epoch_loss / len(train_loader):.4f}"
            )

        return {"train_loss": total_loss / (len(train_loader) * self.config.num_epochs)}

    def _training_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        # Mixed precision context
        autocast_ctx = torch.cuda.amp.autocast(
            enabled=self.config.fp16 or self.config.bf16,
            dtype=torch.float16 if self.config.fp16 else torch.bfloat16,
        )

        with autocast_ctx:
            outputs = self.model(**batch)
            loss = outputs["loss"] / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            if self.scheduler:
                self.scheduler.step()

        self.global_step += 1

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation."""
        if self.eval_dataset is None:
            return {}

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            outputs = self.model(**batch, update_cms=self.config.update_cms_during_eval)
            total_loss += outputs["loss"].item()

            if "logits" in outputs:
                preds = outputs["logits"].argmax(dim=-1)
                all_preds.append(preds.cpu())
                if "labels" in batch:
                    all_labels.append(batch["labels"].cpu())

        metrics = {"eval_loss": total_loss / len(eval_loader)}

        # Compute additional metrics
        if self.compute_metrics and all_labels:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            metrics.update(self.compute_metrics(all_preds, all_labels))

        return metrics

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(path))
        else:
            torch.save(self.model.state_dict(), path / "model.pt")

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }
        if self.scheduler:
            training_state["scheduler"] = self.scheduler.state_dict()
        if self.scaler:
            training_state["scaler"] = self.scaler.state_dict()

        torch.save(training_state, path / "training_state.pt")

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        path = Path(path)

        # Load model
        if hasattr(self.model, "load_state_dict"):
            state_dict = torch.load(path / "model.pt", map_location=self.device)
            self.model.load_state_dict(state_dict)

        # Load training state
        training_state = torch.load(path / "training_state.pt", map_location=self.device)
        self.global_step = training_state["global_step"]
        self.epoch = training_state["epoch"]
        self.optimizer.load_state_dict(training_state["optimizer"])

        if self.scheduler and "scheduler" in training_state:
            self.scheduler.load_state_dict(training_state["scheduler"])
        if self.scaler and "scaler" in training_state:
            self.scaler.load_state_dict(training_state["scaler"])

        logger.info(f"Loaded checkpoint from {path}")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create cosine schedule with warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create linear schedule with warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Common metrics
def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute accuracy metric."""
    mask = labels != -100
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return {"accuracy": correct / total if total > 0 else 0.0}


def compute_perplexity(loss: float) -> dict[str, float]:
    """Compute perplexity from loss."""
    return {"perplexity": math.exp(loss)}
