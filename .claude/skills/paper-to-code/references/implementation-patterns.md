# Implementation Patterns

Proven code patterns for ML paper implementations.

## Project Organization

### Module Exports

Every `__init__.py` should export the public API:

```python
# src/__init__.py
from src.core.optimizers import DeltaGradientDescent, M3Optimizer
from src.core.memory import ContinuumMemorySystem
from src.models.hope import Hope, create_hope

__all__ = [
    "DeltaGradientDescent",
    "M3Optimizer", 
    "ContinuumMemorySystem",
    "Hope",
    "create_hope",
]
```

### Config Management

Use dataclasses for type-safe configs:

```python
from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class ModelConfig:
    """Model configuration with paper defaults."""
    
    # Architecture
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    d_hidden: int = 2048
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0
    
    # Memory (if applicable)
    num_memory_levels: int = 4
    base_chunk_size: int = 16
    
    # Training
    max_seq_len: int = 2048
    vocab_size: int = 32000
    
    def __post_init__(self):
        """Validate config."""
        assert self.d_model % self.num_heads == 0
        assert self.num_memory_levels > 0


@dataclass  
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Batching
    batch_size: int = 32
    gradient_accumulation: int = 1
    
    # Precision
    mixed_precision: bool = True
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000


# Preset configs
CONFIGS = {
    "small": ModelConfig(d_model=256, num_layers=4, num_heads=4),
    "medium": ModelConfig(d_model=512, num_layers=8, num_heads=8),
    "large": ModelConfig(d_model=768, num_layers=12, num_heads=12),
}
```

## Device Handling

### Robust Device Selection

```python
import torch
from typing import Optional

def get_device(device: Optional[str] = None) -> torch.device:
    """Get best available device or use specified device."""
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Get device information for logging."""
    device = get_device()
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info["cuda_device"] = torch.cuda.get_device_name()
        info["cuda_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
    
    return info
```

### Device-Agnostic Code Pattern

```python
class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Don't call .to(device) here - let caller handle it
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tensors created inside forward should match input device
        batch_size = x.size(0)
        # Use x.device to create tensors on same device
        mask = torch.ones(batch_size, device=x.device)
        return self.process(x, mask)
```

## Optimizer Patterns

### Custom Optimizer Template

```python
from torch.optim import Optimizer
from typing import Dict, Any, Optional, Callable

class CustomOptimizer(Optimizer):
    """
    Custom optimizer implementing [Paper Name] Equation X.
    
    Args:
        params: Model parameters
        lr: Learning rate
        betas: Momentum coefficients (default: (0.9, 0.999))
        eps: Numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                
                state['step'] += 1
                m, v = state['m'], state['v']
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update parameters
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(m, denom, value=-step_size)
        
        return loss
```

## Memory Systems

### Associative Memory Pattern

```python
class AssociativeMemory(nn.Module):
    """
    Associative memory module implementing M(x) from paper.
    
    Learns key-value associations and retrieves by similarity.
    """
    
    def __init__(
        self,
        d_key: int,
        d_value: int,
        num_slots: int = 256,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.num_slots = num_slots
        
        # Memory banks
        self.keys = nn.Parameter(torch.randn(num_slots, d_key) * 0.02)
        self.values = nn.Parameter(torch.randn(num_slots, d_value) * 0.02)
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory.
        
        Args:
            query: (batch, seq_len, d_key)
        
        Returns:
            (batch, seq_len, d_value)
        """
        # Compute attention over memory slots
        # (batch, seq_len, d_key) @ (d_key, num_slots) -> (batch, seq_len, num_slots)
        attn = torch.matmul(query, self.keys.t()) / (self.d_key ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Retrieve values
        # (batch, seq_len, num_slots) @ (num_slots, d_value) -> (batch, seq_len, d_value)
        return torch.matmul(attn, self.values)
    
    def write(self, key: torch.Tensor, value: torch.Tensor, learning_rate: float = 0.1):
        """Update memory with new key-value pair (online learning)."""
        # Find most similar slot
        with torch.no_grad():
            sim = torch.matmul(key, self.keys.t())
            idx = sim.argmax(dim=-1)
            
            # Update that slot (Delta rule)
            self.keys.data[idx] += learning_rate * (key - self.keys[idx])
            self.values.data[idx] += learning_rate * (value - self.values[idx])
```

### Multi-Frequency Update Pattern

```python
class MultiFrequencyModule(nn.Module):
    """
    Module with components updating at different frequencies.
    
    Implements Eq. 71: θ^(f_ℓ) updates only when step ≡ 0 (mod C^(ℓ))
    """
    
    def __init__(
        self,
        d_model: int,
        num_levels: int = 4,
        base_chunk_size: int = 16,
    ):
        super().__init__()
        self.num_levels = num_levels
        
        # Exponentially increasing chunk sizes
        self.chunk_sizes = [base_chunk_size * (2 ** i) for i in range(num_levels)]
        
        # One module per frequency level
        self.levels = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_levels)
        ])
        
        # Track steps for each level
        self.register_buffer('step_count', torch.tensor(0))
    
    def should_update(self, level: int) -> bool:
        """Check if level should update at current step."""
        return self.step_count.item() % self.chunk_sizes[level] == 0
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass with frequency-based updates."""
        outputs = []
        
        for level, (module, chunk_size) in enumerate(zip(self.levels, self.chunk_sizes)):
            if training and self.should_update(level):
                # This level updates this step
                out = module(x)
            else:
                # Use cached/frozen output
                with torch.no_grad():
                    out = module(x)
            outputs.append(out)
        
        if training:
            self.step_count += 1
        
        # Aggregate outputs (paper-specific aggregation)
        return sum(outputs) / len(outputs)
```

## Testing Patterns

### Comprehensive Component Tests

```python
import pytest
import torch
import torch.nn as nn

class TestComponent:
    """Test suite for Component."""
    
    @pytest.fixture
    def model(self):
        """Create fresh model for each test."""
        return Component(d_model=64, num_layers=2)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 16, 64)
    
    def test_output_shape(self, model, sample_input):
        """Output shape matches input for residual component."""
        output = model(sample_input)
        assert output.shape == sample_input.shape
    
    def test_different_batch_sizes(self, model):
        """Works with various batch sizes."""
        for batch_size in [1, 2, 8, 32]:
            x = torch.randn(batch_size, 16, 64)
            out = model(x)
            assert out.shape[0] == batch_size
    
    def test_different_seq_lengths(self, model):
        """Works with various sequence lengths."""
        for seq_len in [1, 16, 128, 512]:
            x = torch.randn(2, seq_len, 64)
            out = model(x)
            assert out.shape[1] == seq_len
    
    def test_gradient_flow(self, model, sample_input):
        """Gradients flow through all parameters."""
        sample_input.requires_grad_(True)
        output = model(sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check input gradient
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        
        # Check parameter gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_deterministic(self, model, sample_input):
        """Same input produces same output (eval mode)."""
        model.eval()
        with torch.no_grad():
            out1 = model(sample_input)
            out2 = model(sample_input)
        assert torch.allclose(out1, out2)
    
    def test_no_nan_output(self, model, sample_input):
        """Output contains no NaN values."""
        output = model(sample_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self, model, sample_input):
        """Works on CUDA device."""
        model = model.cuda()
        x = sample_input.cuda()
        out = model(x)
        assert out.device.type == "cuda"
        assert out.shape == x.shape


class TestTrainingIntegration:
    """Integration tests for training."""
    
    def test_single_training_step(self):
        """Complete training step without errors."""
        model = Model(d_model=64, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(2, 32, 64)
        target = torch.randn(2, 32, 64)
        
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert loss.item() < 1e6  # Sanity check
    
    def test_loss_decreases(self):
        """Loss decreases over multiple steps."""
        model = Model(d_model=64, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should generally decrease
        assert losses[-1] < losses[0]
```

## Anti-Patterns to Avoid

### Don't Hardcode Devices

```python
# BAD
self.buffer = torch.zeros(100).cuda()

# GOOD
self.register_buffer('buffer', torch.zeros(100))
# Device handled by model.to(device)
```

### Don't Create Tensors in __init__ Without register_buffer

```python
# BAD - won't move with model.to(device)
self.mask = torch.ones(100, 100)

# GOOD
self.register_buffer('mask', torch.ones(100, 100))
```

### Don't Use Global State

```python
# BAD
GLOBAL_STEP = 0

def forward(self, x):
    global GLOBAL_STEP
    GLOBAL_STEP += 1

# GOOD
self.register_buffer('step', torch.tensor(0))

def forward(self, x):
    self.step += 1
```

### Don't Forget torch.no_grad() for Inference

```python
# BAD - wastes memory on gradient computation
output = model(input)

# GOOD
model.eval()
with torch.no_grad():
    output = model(input)
```
