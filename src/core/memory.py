"""
Continuum Memory System (CMS)
=============================

Implementation of the Continuum Memory System from "Nested Learning" paper.
CMS generalizes long-term/short-term memory by viewing memory as a distributed
inter-connected system with a spectrum of frequency updates.

Based on Equations 70-74 in the paper:
- Higher-frequency neurons: fast adaptation but store memories for short time
- Lower-frequency neurons: more persistent knowledge storage

Key insight: "When updating an arbitrary block MLP^(f_s), the potentially
forgotten knowledge is still stored in other components MLP^(f_s') where s' < s."
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CMSConfig:
    """Configuration for Continuum Memory System."""

    d_model: int = 512
    d_hidden: int = 2048
    num_levels: int = 4
    chunk_sizes: Optional[list[int]] = None  # If None, use exponential decay
    base_chunk_size: int = 16
    learning_rates: Optional[list[float]] = None
    aggregation: Literal["sequential", "independent", "nested"] = "independent"
    alpha: float = 0.1  # Weight for combining independent heads
    dropout: float = 0.1


class MLPBlock(nn.Module):
    """
    MLP block with residual connection.

    Architecture (Equation 89):
        M(x) = x + W1 * σ(W2 * x)
    """

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard practice."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System (CMS).

    From Section 7.1 of the paper:
    "CMS is formalized as a chain of MLP blocks MLP^(f1), ..., MLP^(fk),
    each associated with a chunk size C^(l) such that given input x,
    the output is calculated as:

        y_t = MLP^(fk)(MLP^(f(k-1))(...MLP^(f1)(x_t)))

    where parameters θ^(fl) are updated every C^(l) steps."

    The key innovation is that different MLP blocks are updated at different
    frequencies, allowing for:
    - High-frequency blocks: Fast adaptation, short-term memory
    - Low-frequency blocks: Slow adaptation, long-term memory

    This design helps with continual learning because when knowledge is
    forgotten from a high-frequency block, it may still be stored in
    lower-frequency blocks.

    Args:
        config: CMSConfig with all hyperparameters
    """

    def __init__(self, config: CMSConfig):
        super().__init__()
        self.config = config
        self.num_levels = config.num_levels

        # Compute chunk sizes (exponentially increasing)
        if config.chunk_sizes is None:
            self.chunk_sizes = [config.base_chunk_size * (2**i) for i in range(config.num_levels)]
        else:
            self.chunk_sizes = config.chunk_sizes

        # Compute learning rates (decreasing with level)
        if config.learning_rates is None:
            self.learning_rates = [0.01 / (2**i) for i in range(config.num_levels)]
        else:
            self.learning_rates = config.learning_rates

        # Create MLP blocks for each frequency level
        self.mlp_blocks = nn.ModuleList(
            [
                MLPBlock(config.d_model, config.d_hidden, config.dropout)
                for _ in range(config.num_levels)
            ]
        )

        # For independent aggregation variant
        if config.aggregation == "independent":
            self.aggregation_weights = nn.Parameter(
                torch.ones(config.num_levels) / config.num_levels
            )

        # Layer norms
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.num_levels)]
        )

        # Step counter for tracking when to update each level
        self.register_buffer("step_count", torch.tensor(0))

        # Store gradients for each level (for online updates)
        self._accumulated_grads = [None] * config.num_levels
        self._grad_counts = [0] * config.num_levels

    def forward(self, x: torch.Tensor, update_memories: bool = True) -> torch.Tensor:
        """
        Forward pass through CMS.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            update_memories: Whether to update memory blocks (set False for inference)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        if self.config.aggregation == "sequential":
            return self._forward_sequential(x, update_memories)
        elif self.config.aggregation == "independent":
            return self._forward_independent(x, update_memories)
        elif self.config.aggregation == "nested":
            return self._forward_nested(x, update_memories)
        else:
            raise ValueError(f"Unknown aggregation: {self.config.aggregation}")

    def _forward_sequential(self, x: torch.Tensor, update_memories: bool) -> torch.Tensor:
        """
        Sequential variant (Equation 70):
            y_t = MLP^(fk)(MLP^(f(k-1))(...MLP^(f1)(x_t)))

        Output of level s is input to level s+1.
        """
        output = x

        for level_idx in range(self.num_levels):
            output = self.layer_norms[level_idx](output)
            output = self.mlp_blocks[level_idx](output)

            if update_memories:
                self._maybe_update_level(level_idx)

        if update_memories:
            self.step_count += 1

        return output

    def _forward_independent(self, x: torch.Tensor, update_memories: bool) -> torch.Tensor:
        """
        Independent (head-wise) variant (Equation 74):
            y_t = Agg(MLP^(fk)(x_t), MLP^(f(k-1))(x_t), ..., MLP^(f1)(x_t))

        All blocks receive the same input, outputs are aggregated.
        """
        outputs = []

        for level_idx in range(self.num_levels):
            level_input = self.layer_norms[level_idx](x)
            level_output = self.mlp_blocks[level_idx](level_input)
            outputs.append(level_output)

            if update_memories:
                self._maybe_update_level(level_idx)

        # Aggregate using learned weights (softmax for stability)
        weights = F.softmax(self.aggregation_weights, dim=0)
        output = sum(w * out for w, out in zip(weights, outputs))

        if update_memories:
            self.step_count += 1

        return output

    def _forward_nested(self, x: torch.Tensor, update_memories: bool) -> torch.Tensor:
        """
        Nested variant (Equation 72):
        Higher level's initial state is meta-learned from lower level.
        This enables higher-order in-context learning.
        """
        # For nested variant, we use sequential but with meta-learned init
        # The meta-learning happens during training (outer loop optimization)
        return self._forward_sequential(x, update_memories)

    def _maybe_update_level(self, level_idx: int):
        """
        Update level if it's time according to its chunk size.

        From Equation 71:
            θ^(fl)_{i+1} = θ^(fl)_i - Σ η^(l)_t f(θ^(fl)_t; x_t)
            if i ≡ 0 (mod C^(l))
        """
        chunk_size = self.chunk_sizes[level_idx]
        current_step = self.step_count.item()

        # Check if it's time to update this level
        if current_step > 0 and current_step % chunk_size == 0:
            # Apply accumulated gradients
            if self._accumulated_grads[level_idx] is not None:
                lr = self.learning_rates[level_idx]
                with torch.no_grad():
                    for param, grad in zip(
                        self.mlp_blocks[level_idx].parameters(), self._accumulated_grads[level_idx]
                    ):
                        if grad is not None:
                            # Average the accumulated gradients
                            avg_grad = grad / max(1, self._grad_counts[level_idx])
                            param.add_(avg_grad, alpha=-lr)

                # Reset accumulated gradients
                self._accumulated_grads[level_idx] = None
                self._grad_counts[level_idx] = 0

    def accumulate_gradients(self, level_idx: int):
        """Accumulate gradients for a specific level."""
        if self._accumulated_grads[level_idx] is None:
            self._accumulated_grads[level_idx] = [
                p.grad.clone() if p.grad is not None else None
                for p in self.mlp_blocks[level_idx].parameters()
            ]
        else:
            for i, p in enumerate(self.mlp_blocks[level_idx].parameters()):
                if p.grad is not None:
                    if self._accumulated_grads[level_idx][i] is not None:
                        self._accumulated_grads[level_idx][i].add_(p.grad)
                    else:
                        self._accumulated_grads[level_idx][i] = p.grad.clone()

        self._grad_counts[level_idx] += 1

    def reset_step_counter(self):
        """Reset the step counter (e.g., at the start of a new sequence)."""
        self.step_count.zero_()
        self._accumulated_grads = [None] * self.num_levels
        self._grad_counts = [0] * self.num_levels

    def get_update_schedule(self, max_steps: int) -> list[list[int]]:
        """
        Get the update schedule for visualization/debugging.

        Returns:
            List of lists, where schedule[level][i] is the i-th update step for that level.
        """
        schedule = []
        for level_idx in range(self.num_levels):
            chunk_size = self.chunk_sizes[level_idx]
            level_updates = list(range(chunk_size, max_steps + 1, chunk_size))
            schedule.append(level_updates)
        return schedule


class CMSWithOnlineUpdate(ContinuumMemorySystem):
    """
    CMS variant with true online updates during forward pass.

    This implements the online consolidation process mentioned in the paper,
    where memory blocks are updated as data flows through them.
    """

    def __init__(self, config: CMSConfig):
        super().__init__(config)

        # Optimizer for online updates
        self.online_lr = config.learning_rates[0] if config.learning_rates else 0.01

    def forward_with_online_update(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with online memory updates.

        When targets are provided, computes loss and updates memory blocks
        according to their update frequencies.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            targets: Optional target tensor for computing update signal

        Returns:
            Tuple of (output, loss)
        """
        batch_size, seq_len, d_model = x.shape
        outputs = []
        total_loss = 0.0

        for t in range(seq_len):
            x_t = x[:, t : t + 1, :]  # (batch, 1, d_model)

            # Forward through CMS
            y_t = self._forward_sequential(x_t, update_memories=False)
            outputs.append(y_t)

            # Compute update signal if targets provided
            if targets is not None:
                target_t = targets[:, t : t + 1, :]
                loss_t = F.mse_loss(y_t, target_t)
                total_loss += loss_t

                # Compute gradients
                loss_t.backward(retain_graph=True)

                # Accumulate and potentially apply updates
                for level_idx in range(self.num_levels):
                    self.accumulate_gradients(level_idx)
                    self._maybe_update_level(level_idx)

            self.step_count += 1

        output = torch.cat(outputs, dim=1)
        return output, total_loss / seq_len if targets is not None else None


def create_cms(
    d_model: int = 512,
    d_hidden: int = 2048,
    num_levels: int = 4,
    base_chunk_size: int = 16,
    aggregation: str = "independent",
) -> ContinuumMemorySystem:
    """
    Factory function to create a CMS with sensible defaults.

    Args:
        d_model: Model dimension
        d_hidden: Hidden dimension in MLP blocks
        num_levels: Number of frequency levels
        base_chunk_size: Chunk size for highest frequency level
        aggregation: 'sequential', 'independent', or 'nested'

    Returns:
        Configured CMS instance
    """
    config = CMSConfig(
        d_model=d_model,
        d_hidden=d_hidden,
        num_levels=num_levels,
        base_chunk_size=base_chunk_size,
        aggregation=aggregation,
    )
    return ContinuumMemorySystem(config)


# Example usage and testing
if __name__ == "__main__":
    # Create CMS with 4 levels
    config = CMSConfig(
        d_model=256, d_hidden=1024, num_levels=4, base_chunk_size=8, aggregation="independent"
    )
    cms = ContinuumMemorySystem(config)

    # Test forward pass
    batch_size, seq_len = 2, 64
    x = torch.randn(batch_size, seq_len, config.d_model)

    output = cms(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Chunk sizes: {cms.chunk_sizes}")
    print(f"Learning rates: {cms.learning_rates}")

    # Show update schedule
    schedule = cms.get_update_schedule(max_steps=64)
    for level_idx, updates in enumerate(schedule):
        print(
            f"Level {level_idx} (chunk_size={cms.chunk_sizes[level_idx]}): "
            f"updates at steps {updates[:5]}..."
        )
