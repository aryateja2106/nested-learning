"""
Enterprise Continual Learning Pipeline
=======================================

Business Use Case: Real-time Customer Intelligence System

This experiment demonstrates how the HOPE model can be used for enterprise
applications requiring continual learning without catastrophic forgetting.

Scenario:
- Process streaming customer feedback, support tickets, or market news
- Maintain long-term pattern memory (via CMS)
- Adapt in real-time to new patterns (via Self-Modifying Titans)
- No catastrophic forgetting (key business requirement)

Why this matters:
- Customer support teams need AI that remembers previous interactions
- Market analysts need models that adapt to changing conditions
- Enterprise systems require continuous learning without retraining

Designed for A100 GPU with CUDA acceleration.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..core.optimizers import DeltaGradientDescent, M3Optimizer
from ..models.hope import Hope, HopeConfig
from .cuda_kernels import (
    benchmark_cuda_operations,
    check_cuda_available,
    get_tensor_core_dtype,
)


@dataclass
class EnterpriseConfig:
    """Configuration for enterprise pipeline experiment."""

    # Model architecture
    d_model: int = 512
    d_hidden: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    vocab_size: int = 10000
    max_seq_len: int = 512

    # CMS configuration
    cms_num_levels: int = 4
    cms_base_chunk_size: int = 16

    # Training
    batch_size: int = 32
    seq_len: int = 256
    num_steps: int = 1000
    learning_rate: float = 3e-4
    optimizer: str = "dgd"  # dgd, m3, adamw
    use_amp: bool = True
    use_cuda_acceleration: bool = True

    # Business scenario
    num_customer_segments: int = 5  # Different customer types
    adaptation_rate: float = 0.1  # How quickly model adapts to new patterns

    # Logging
    log_interval: int = 50
    output_json: bool = False  # For lecoder-cgpu integration


class CustomerFeedbackDataset(Dataset):
    """
    Simulated customer feedback dataset for enterprise use case.
    
    Simulates:
    - Different customer segments with distinct patterns
    - Temporal shifts (customer preferences change over time)
    - Sentiment patterns (positive/negative feedback)
    - Product-specific terminology
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int = 10000,
        customer_segment: int = 0,
        temporal_shift: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.customer_segment = customer_segment
        self.temporal_shift = temporal_shift

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Base pattern depends on customer segment
        base_pattern = self._generate_segment_pattern()
        
        # Apply temporal shift (simulates changing preferences)
        shifted_idx = int(idx * (1 + self.temporal_shift))
        
        # Generate sequence with segment-specific patterns
        tokens = self._generate_feedback_sequence(base_pattern, shifted_idx)
        
        # Labels are next-token prediction
        labels = torch.cat([tokens[1:], tokens[:1]])  # Shifted by one
        
        return {"input_ids": tokens, "labels": labels}

    def _generate_segment_pattern(self) -> List[int]:
        """Generate base pattern for customer segment."""
        # Each segment has characteristic vocabulary and patterns
        segment_vocab_start = (self.customer_segment * self.vocab_size) // 5
        segment_vocab_end = ((self.customer_segment + 1) * self.vocab_size) // 5
        
        pattern_len = 8
        pattern = torch.randint(
            segment_vocab_start,
            segment_vocab_end,
            (pattern_len,),
        ).tolist()
        
        return pattern

    def _generate_feedback_sequence(self, base_pattern: List[int], idx: int) -> torch.Tensor:
        """Generate customer feedback sequence."""
        # Mix of pattern repetition and variation
        pattern_type = idx % 4
        
        if pattern_type == 0:
            # Direct pattern repetition (common phrases)
            repeats = self.seq_len // len(base_pattern) + 1
            tokens = torch.tensor(base_pattern * repeats)[: self.seq_len]
        elif pattern_type == 1:
            # Pattern with noise (natural variation)
            repeats = self.seq_len // len(base_pattern) + 1
            tokens = torch.tensor(base_pattern * repeats)[: self.seq_len]
            # Add some random variation
            noise_mask = torch.rand(self.seq_len) < 0.1
            tokens[noise_mask] = torch.randint(0, self.vocab_size, (noise_mask.sum(),))
        elif pattern_type == 2:
            # Sentiment pattern: [positive | negative | positive]
            half = self.seq_len // 2
            positive_tokens = torch.tensor(base_pattern[:4])
            negative_tokens = torch.tensor(base_pattern[4:])
            tokens = torch.cat([
                positive_tokens.repeat(half // 4 + 1)[:half],
                negative_tokens.repeat((self.seq_len - half) // 4 + 1)[:self.seq_len - half]
            ])
        else:
            # Product-specific terminology (specialized vocabulary)
            product_vocab_start = self.vocab_size - 100
            tokens = torch.randint(product_vocab_start, self.vocab_size, (self.seq_len,))
            # Insert pattern occasionally
            for i in range(0, self.seq_len, 16):
                tokens[i:i+len(base_pattern)] = torch.tensor(base_pattern[:min(len(base_pattern), self.seq_len - i)])
        
        return tokens


class EnterprisePipeline:
    """
    Enterprise continual learning pipeline using HOPE model.
    
    Demonstrates:
    1. Long-term memory via CMS (remembers customer patterns)
    2. Real-time adaptation via Titans (adapts to new feedback)
    3. No catastrophic forgetting (maintains knowledge across segments)
    """

    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check CUDA capabilities
        self.use_cuda_accel = config.use_cuda_acceleration and check_cuda_available()
        if self.use_cuda_accel:
            print(f"✓ CUDA acceleration enabled: {torch.cuda.get_device_name(0)}")
            print(f"  Tensor cores available: {check_cuda_available()}")
            print(f"  Optimal dtype: {get_tensor_core_dtype()}")
        else:
            print("⚠ CUDA acceleration not available, using CPU")
        
        # Create model
        hope_config = HopeConfig(
            d_model=config.d_model,
            d_hidden=config.d_hidden,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            cms_num_levels=config.cms_num_levels,
            cms_base_chunk_size=config.cms_base_chunk_size,
        )
        self.model = Hope(hope_config).to(self.device)
        
        # Create optimizer
        if config.optimizer == "dgd":
            self.optimizer = DeltaGradientDescent(self.model.parameters(), lr=config.learning_rate)
        elif config.optimizer == "m3":
            self.optimizer = M3Optimizer(self.model.parameters(), lr=config.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # AMP scaler
        self.scaler = torch.amp.GradScaler() if config.use_amp else None
        
        # Metrics
        self.metrics_history = []
        self.step = 0
        self.start_time = None

    def train_on_segment(
        self,
        segment_id: int,
        num_steps: int,
        temporal_shift: float = 0.0,
    ) -> Dict:
        """Train on a specific customer segment."""
        dataset = CustomerFeedbackDataset(
            vocab_size=self.config.vocab_size,
            seq_len=self.config.seq_len,
            num_samples=num_steps * self.config.batch_size * 2,
            customer_segment=segment_id,
            temporal_shift=temporal_shift,
        )
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        segment_metrics = []
        train_iter = iter(dataloader)
        
        for step in range(1, num_steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader)
                batch = next(train_iter)
            
            metrics = self._train_step(batch)
            metrics["segment"] = segment_id
            metrics["step"] = self.step
            segment_metrics.append(metrics)
            
            if step % self.config.log_interval == 0:
                self._log_metrics(metrics, segment_id)
        
        return {
            "segment_id": segment_id,
            "final_loss": segment_metrics[-1]["loss"],
            "final_accuracy": segment_metrics[-1]["accuracy"],
            "steps": num_steps,
        }

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.step += 1
        
        if self.start_time is None:
            self.start_time = time.time()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass with AMP
        with torch.amp.autocast(device_type="cuda", enabled=self.config.use_amp):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"]
        
        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            logits = outputs["logits"]
            preds = logits[:, :-1].argmax(dim=-1)
            targets = labels[:, 1:]
            accuracy = (preds == targets).float().mean().item()
            
            # Throughput calculation
            tokens_processed = input_ids.numel()
            elapsed = time.time() - self.start_time
            throughput = (tokens_processed * self.step) / elapsed if elapsed > 0 else 0
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "perplexity": math.exp(min(loss.item(), 10)),
            "throughput_tokens_per_sec": throughput,
        }

    def _log_metrics(self, metrics: Dict, segment_id: int):
        """Log training metrics."""
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        log_msg = (
            f"Step {self.step:5d} | Segment {segment_id} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.3f} | "
            f"PPL: {metrics['perplexity']:.2f} | "
            f"Throughput: {metrics['throughput_tokens_per_sec']:.0f} tok/s"
        )
        
        if gpu_mem > 0:
            log_msg += f" | GPU: {gpu_mem:.2f}GB"
        
        print(log_msg)

    def evaluate_continual_learning(
        self,
        test_segments: List[int],
        num_samples_per_segment: int = 100,
    ) -> Dict:
        """
        Evaluate continual learning capability.
        
        Tests if model remembers patterns from previous segments
        (no catastrophic forgetting).
        """
        self.model.eval()
        results = {}
        
        for segment_id in test_segments:
            dataset = CustomerFeedbackDataset(
                vocab_size=self.config.vocab_size,
                seq_len=self.config.seq_len,
                num_samples=num_samples_per_segment,
                customer_segment=segment_id,
            )
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size)
            
            total_loss = 0.0
            total_correct = 0
            total_tokens = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    outputs = self.model(input_ids, labels=labels, update_cms=False)
                    total_loss += outputs["loss"].item()
                    
                    preds = outputs["logits"][:, :-1].argmax(dim=-1)
                    targets = labels[:, 1:]
                    total_correct += (preds == targets).sum().item()
                    total_tokens += targets.numel()
            
            results[f"segment_{segment_id}"] = {
                "loss": total_loss / len(dataloader),
                "accuracy": total_correct / total_tokens,
            }
        
        return results

    def run_experiment(self) -> Dict:
        """Run full enterprise experiment."""
        print("\n" + "=" * 80)
        print("ENTERPRISE CONTINUAL LEARNING PIPELINE")
        print("=" * 80)
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"CUDA Acceleration: {self.use_cuda_accel}")
        print(f"Mixed Precision: {self.config.use_amp}")
        print(f"Optimizer: {self.config.optimizer}")
        print("=" * 80 + "\n")
        
        # Benchmark CUDA operations if available
        if self.use_cuda_accel:
            print("Benchmarking CUDA operations...")
            benchmark_results = benchmark_cuda_operations(
                batch_size=self.config.batch_size,
                seq_len=self.config.seq_len,
                d_model=self.config.d_model,
            )
            print(f"  GPU: {benchmark_results.get('gpu_name', 'N/A')}")
            print(f"  Operations/sec: {benchmark_results.get('operations_per_second', 0):.0f}")
            print(f"  Tensor cores: {benchmark_results.get('tensor_cores_available', False)}")
            print()
        
        # Train on multiple customer segments sequentially
        # This tests continual learning (no forgetting)
        steps_per_segment = self.config.num_steps // self.config.num_customer_segments
        
        segment_results = []
        for segment_id in range(self.config.num_customer_segments):
            print(f"\n>>> Training on Customer Segment {segment_id} <<<")
            result = self.train_on_segment(
                segment_id=segment_id,
                num_steps=steps_per_segment,
                temporal_shift=segment_id * 0.1,  # Simulate temporal drift
            )
            segment_results.append(result)
        
        # Evaluate continual learning
        print("\n>>> Evaluating Continual Learning Capability <<<")
        eval_results = self.evaluate_continual_learning(
            test_segments=list(range(self.config.num_customer_segments)),
        )
        
        # Final summary
        total_time = time.time() - self.start_time if self.start_time else 0
        final_metrics = self.metrics_history[-1] if self.metrics_history else {}
        
        summary = {
            "experiment": "enterprise_continual_learning",
            "config": asdict(self.config),
            "hardware": {
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "cuda_acceleration": self.use_cuda_accel,
                "tensor_cores": check_cuda_available() if torch.cuda.is_available() else False,
            },
            "training": {
                "total_steps": self.step,
                "total_time_seconds": total_time,
                "steps_per_second": self.step / total_time if total_time > 0 else 0,
                "final_loss": final_metrics.get("loss", 0),
                "final_accuracy": final_metrics.get("accuracy", 0),
                "final_throughput": final_metrics.get("throughput_tokens_per_sec", 0),
            },
            "segment_results": segment_results,
            "continual_learning_evaluation": eval_results,
        }
        
        if self.config.output_json:
            print("\n" + json.dumps(summary, indent=2, default=str))
        else:
            self._print_summary(summary)
        
        return summary

    def _print_summary(self, summary: Dict):
        """Print human-readable summary."""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        
        print("\n## Hardware")
        hw = summary["hardware"]
        print(f"  Device: {hw['device']}")
        print(f"  GPU: {hw['gpu_name']}")
        print(f"  CUDA Acceleration: {hw['cuda_acceleration']}")
        print(f"  Tensor Cores: {hw['tensor_cores']}")
        
        print("\n## Training Performance")
        train = summary["training"]
        print(f"  Total Steps: {train['total_steps']}")
        print(f"  Total Time: {train['total_time_seconds']:.1f}s")
        print(f"  Steps/Second: {train['steps_per_second']:.2f}")
        print(f"  Final Loss: {train['final_loss']:.4f}")
        print(f"  Final Accuracy: {train['final_accuracy']:.3f}")
        print(f"  Throughput: {train['final_throughput']:.0f} tokens/sec")
        
        print("\n## Continual Learning Evaluation")
        cl_eval = summary["continual_learning_evaluation"]
        for segment_key, metrics in cl_eval.items():
            print(f"  {segment_key}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.3f}")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Enterprise Continual Learning Pipeline")
    parser.add_argument("--config", type=str, default="a100", choices=["a100", "t4", "cpu"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["dgd", "m3", "adamw"])
    parser.add_argument("--output-json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output-file", type=str, default=None, help="Save results to file")
    
    args = parser.parse_args()
    
    # Create config based on preset
    if args.config == "a100":
        config = EnterpriseConfig(
            d_model=512,
            d_hidden=2048,
            num_layers=6,
            batch_size=32,
            num_steps=1000,
            use_cuda_acceleration=True,
            use_amp=True,
        )
    elif args.config == "t4":
        config = EnterpriseConfig(
            d_model=256,
            d_hidden=1024,
            num_layers=4,
            batch_size=16,
            num_steps=500,
            use_cuda_acceleration=True,
            use_amp=True,
        )
    else:  # cpu
        config = EnterpriseConfig(
            d_model=128,
            d_hidden=512,
            num_layers=2,
            batch_size=8,
            num_steps=200,
            use_cuda_acceleration=False,
            use_amp=False,
        )
    
    # Override with CLI args
    if args.steps:
        config.num_steps = args.steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.optimizer:
        config.optimizer = args.optimizer
    if args.output_json:
        config.output_json = True
    
    # Run experiment
    pipeline = EnterprisePipeline(config)
    results = pipeline.run_experiment()
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

