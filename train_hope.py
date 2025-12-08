#!/usr/bin/env python3
"""
Hope Model Training Script
==========================

Train the Hope architecture on a synthetic language modeling task.
Designed to run on NVIDIA A100 GPU via cgpu.

Usage:
    python train_hope.py --config small   # Quick test
    python train_hope.py --config medium  # Balanced
    python train_hope.py --config large   # Full training
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from src.core.optimizers import DeltaGradientDescent, M3Optimizer
from src.models.hope import Hope, HopeConfig

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainConfig:
    """Training configuration."""

    # Model
    d_model: int = 256
    d_hidden: int = 1024
    d_key: int = 32
    d_value: int = 32
    num_heads: int = 8
    num_layers: int = 4
    vocab_size: int = 10000
    max_seq_len: int = 512

    # Titans
    titans_chunk_size: int = 16
    titans_hidden: int = 128

    # CMS
    cms_num_levels: int = 3
    cms_base_chunk_size: int = 16

    # Training
    batch_size: int = 16
    seq_len: int = 128
    num_steps: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0

    # Optimizer
    optimizer: str = "adamw"  # adamw, m3, dgd

    # Logging
    log_interval: int = 10
    eval_interval: int = 100

    # Device
    device: str = "cuda"
    use_amp: bool = True  # Automatic mixed precision


CONFIGS = {
    "small": TrainConfig(
        d_model=128,
        d_hidden=512,
        num_layers=2,
        num_heads=4,
        batch_size=32,
        seq_len=64,
        num_steps=200,
        vocab_size=5000,
    ),
    "medium": TrainConfig(
        d_model=256,
        d_hidden=1024,
        num_layers=4,
        num_heads=8,
        batch_size=16,
        seq_len=128,
        num_steps=500,
        vocab_size=10000,
    ),
    "large": TrainConfig(
        d_model=512,
        d_hidden=2048,
        num_layers=6,
        num_heads=8,
        batch_size=8,
        seq_len=256,
        num_steps=1000,
        vocab_size=32000,
    ),
}


# ============================================================================
# Dataset
# ============================================================================


class SyntheticLMDataset(Dataset):
    """
    Synthetic language modeling dataset.

    Generates sequences with learnable patterns:
    - Repeated subsequences
    - Arithmetic progressions
    - Copy patterns
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Mix of different patterns
        pattern_type = idx % 4

        if pattern_type == 0:
            # Random sequence
            tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        elif pattern_type == 1:
            # Repeated pattern (model should learn to predict)
            pattern_len = torch.randint(4, 16, (1,)).item()
            pattern = torch.randint(0, self.vocab_size, (pattern_len,))
            repeats = self.seq_len // pattern_len + 1
            tokens = pattern.repeat(repeats)[: self.seq_len]
        elif pattern_type == 2:
            # Arithmetic sequence (mod vocab_size)
            start = torch.randint(0, self.vocab_size, (1,)).item()
            step = torch.randint(1, 10, (1,)).item()
            tokens = torch.arange(start, start + self.seq_len * step, step) % self.vocab_size
        else:
            # Copy pattern: [prefix | delimiter | prefix]
            half = self.seq_len // 2 - 1
            prefix = torch.randint(0, self.vocab_size - 1, (half,))
            delimiter = torch.tensor([self.vocab_size - 1])  # Special delimiter
            tokens = torch.cat([prefix, delimiter, prefix, delimiter])[: self.seq_len]

        return {"input_ids": tokens, "labels": tokens}


# ============================================================================
# Training
# ============================================================================


class Trainer:
    """Hope model trainer with structured logging."""

    def __init__(self, model: Hope, config: TrainConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # AMP
        self.scaler = torch.amp.GradScaler() if config.use_amp else None

        # Metrics
        self.metrics_history = []
        self.step = 0

    def _create_optimizer(self):
        """Create optimizer based on config."""
        params = self.model.parameters()

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "m3":
            return M3Optimizer(params, lr=self.config.learning_rate)
        elif self.config.optimizer == "dgd":
            return DeltaGradientDescent(params, lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _get_lr(self) -> float:
        """Get learning rate with warmup."""
        if self.step < self.config.warmup_steps:
            return self.config.learning_rate * self.step / self.config.warmup_steps

        # Cosine decay
        progress = (self.step - self.config.warmup_steps) / max(
            1, self.config.num_steps - self.config.warmup_steps
        )
        return self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def _update_lr(self):
        """Update learning rate."""
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step."""
        self.model.train()
        self.step += 1

        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Update LR
        lr = self._update_lr()

        # Forward pass
        with torch.amp.autocast(device_type="cuda", enabled=self.config.use_amp):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"]

        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            logits = outputs["logits"]
            preds = logits[:, :-1].argmax(dim=-1)
            targets = labels[:, 1:]
            accuracy = (preds == targets).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "perplexity": math.exp(min(loss.item(), 10)),
            "lr": lr,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, labels=labels, update_cms=False)
            total_loss += outputs["loss"].item()

            preds = outputs["logits"][:, :-1].argmax(dim=-1)
            targets = labels[:, 1:]
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.numel()

        avg_loss = total_loss / len(dataloader)
        return {
            "eval_loss": avg_loss,
            "eval_accuracy": total_correct / total_tokens,
            "eval_perplexity": math.exp(min(avg_loss, 10)),
        }

    def train(self, train_loader: DataLoader, eval_loader: DataLoader) -> dict:
        """Full training loop."""
        print("\n" + "=" * 70)
        print("HOPE MODEL TRAINING")
        print("=" * 70)

        # Model info
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Parameters: {num_params:,}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.config.use_amp}")
        print(f"Optimizer: {self.config.optimizer}")
        print("=" * 70 + "\n")

        start_time = time.time()
        best_loss = float("inf")

        # Training loop
        train_iter = iter(train_loader)

        for step in range(1, self.config.num_steps + 1):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Train step
            metrics = self.train_step(batch)
            self.metrics_history.append({"step": step, **metrics})

            # Logging
            if step % self.config.log_interval == 0:
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"Step {step:5d}/{self.config.num_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"PPL: {metrics['perplexity']:.2f} | "
                    f"Acc: {metrics['accuracy']:.3f} | "
                    f"LR: {metrics['lr']:.2e} | "
                    f"GPU: {gpu_mem:.2f}GB"
                )

            # Evaluation
            if step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(eval_loader)
                print(
                    f"\n>>> EVAL @ Step {step}: "
                    f"Loss={eval_metrics['eval_loss']:.4f}, "
                    f"PPL={eval_metrics['eval_perplexity']:.2f}, "
                    f"Acc={eval_metrics['eval_accuracy']:.3f}\n"
                )

                if eval_metrics["eval_loss"] < best_loss:
                    best_loss = eval_metrics["eval_loss"]

        # Final evaluation
        final_metrics = self.evaluate(eval_loader)
        total_time = time.time() - start_time

        # Summary
        results = {
            "training": {
                "total_steps": self.config.num_steps,
                "total_time_seconds": total_time,
                "steps_per_second": self.config.num_steps / total_time,
                "final_train_loss": metrics["loss"],
                "final_train_perplexity": metrics["perplexity"],
                "final_train_accuracy": metrics["accuracy"],
            },
            "evaluation": {
                "final_eval_loss": final_metrics["eval_loss"],
                "final_eval_perplexity": final_metrics["eval_perplexity"],
                "final_eval_accuracy": final_metrics["eval_accuracy"],
                "best_eval_loss": best_loss,
            },
            "model": {
                "parameters": num_params,
                "d_model": self.config.d_model,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
            },
            "hardware": {
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "gpu_memory_used_gb": torch.cuda.max_memory_allocated() / 1e9,
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available()
                else 0,
            },
            "config": asdict(self.config),
        }

        return results


def print_results(results: dict):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)

    print("\n## Training Summary")
    print(f"  Total Steps:        {results['training']['total_steps']}")
    print(f"  Total Time:         {results['training']['total_time_seconds']:.1f}s")
    print(f"  Steps/Second:       {results['training']['steps_per_second']:.2f}")
    print(f"  Final Train Loss:   {results['training']['final_train_loss']:.4f}")
    print(f"  Final Train PPL:    {results['training']['final_train_perplexity']:.2f}")
    print(f"  Final Train Acc:    {results['training']['final_train_accuracy']:.3f}")

    print("\n## Evaluation Summary")
    print(f"  Final Eval Loss:    {results['evaluation']['final_eval_loss']:.4f}")
    print(f"  Final Eval PPL:     {results['evaluation']['final_eval_perplexity']:.2f}")
    print(f"  Final Eval Acc:     {results['evaluation']['final_eval_accuracy']:.3f}")
    print(f"  Best Eval Loss:     {results['evaluation']['best_eval_loss']:.4f}")

    print("\n## Model Info")
    print(f"  Parameters:         {results['model']['parameters']:,}")
    print(f"  d_model:            {results['model']['d_model']}")
    print(f"  num_layers:         {results['model']['num_layers']}")
    print(f"  num_heads:          {results['model']['num_heads']}")

    print("\n## Hardware")
    print(f"  Device:             {results['hardware']['device']}")
    print(f"  GPU:                {results['hardware']['gpu_name']}")
    print(f"  GPU Memory Used:    {results['hardware']['gpu_memory_used_gb']:.2f}GB")
    print(f"  GPU Memory Total:   {results['hardware']['gpu_memory_total_gb']:.2f}GB")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train Hope Model")
    parser.add_argument(
        "--config", type=str, default="medium", choices=["small", "medium", "large"]
    )
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "m3", "dgd"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output", type=str, default="training_results.json")
    args = parser.parse_args()

    # Load config
    config = CONFIGS[args.config]

    # Override with CLI args
    if args.optimizer:
        config.optimizer = args.optimizer
    if args.steps:
        config.num_steps = args.steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    # Create model
    hope_config = HopeConfig(
        d_model=config.d_model,
        d_hidden=config.d_hidden,
        d_key=config.d_key,
        d_value=config.d_value,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        titans_chunk_size=config.titans_chunk_size,
        titans_hidden=config.titans_hidden,
        cms_num_levels=config.cms_num_levels,
        cms_base_chunk_size=config.cms_base_chunk_size,
    )
    model = Hope(hope_config)

    # Create datasets
    train_dataset = SyntheticLMDataset(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_samples=config.num_steps * config.batch_size * 2,
    )
    eval_dataset = SyntheticLMDataset(
        vocab_size=config.vocab_size, seq_len=config.seq_len, num_samples=config.batch_size * 10
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size)

    # Train
    trainer = Trainer(model, config)
    results = trainer.train(train_loader, eval_loader)

    # Print results
    print_results(results)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
