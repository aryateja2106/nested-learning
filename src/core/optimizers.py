"""
Nested Learning Optimizers
==========================

Implementation of optimizers as associative memories, including:
- Delta Gradient Descent (DGD)
- Delta Momentum
- Multi-scale Momentum Muon (M3)

Based on: "Nested Learning: The Illusion of Deep Learning Architecture"
Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


def newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration for orthogonalization.

    Maps gradients/momentum to an orthogonal space via:
        O_{i+1} = O_i - ζ * ∇L(O_i; g)
    where L(O; g) = ||O^T O - I||^2_F

    This results in the polynomial:
        O_{i+1} = O_i * (3I - O_i^T O_i) / 2

    Args:
        M: Input matrix to orthogonalize
        steps: Number of Newton-Schulz iterations

    Returns:
        Orthogonalized matrix (same shape as input)
    """
    if M.dim() < 2:
        return M

    # Use Newton-Schulz iteration to preserve shape (QR changes shape for non-square)
    M = M / (M.norm() + 1e-8)
    for _ in range(steps):
        eye = torch.eye(M.shape[1], device=M.device, dtype=M.dtype)
        M = 0.5 * M @ (3 * eye - M.T @ M)

    # Handle NaN/Inf by returning normalized input
    if torch.isnan(M).any() or torch.isinf(M).any():
        return M / (M.norm() + 1e-8)

    return M


class DeltaGradientDescent(Optimizer):
    """
    Delta Gradient Descent (DGD) Optimizer.

    Unlike standard gradient descent which treats each data sample independently,
    DGD incorporates the current weight state into the update, capturing dependencies
    without i.i.d. assumptions.

    Update rule (Equation 57):
        W_{t+1} = W_t (I - η'_t x_t x_t^T) - η'_t ∇_y L(W_t; x_t) ⊗ x_t

    where:
        - η'_t = η_t / (1 + η_t) for normalized inputs
        - The first term provides adaptive decay based on current input
        - The second term is the gradient update

    From the paper:
        "This new algorithm updates the weights not only with respect to the
        current elements, but it also incorporates the previous state of weights,
        resulting in an adaptive decay term based on the current data sample."

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 1e-3)
        weight_decay: L2 regularization (default: 0)
        normalized_inputs: If True, assumes ||x_t||_2 is constant (default: True)
    """

    def __init__(
        self, params, lr: float = 1e-3, weight_decay: float = 0.0, normalized_inputs: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay, normalized_inputs=normalized_inputs)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None, input_tensor: Optional[torch.Tensor] = None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
            input_tensor: The input x_t for adaptive decay computation
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            normalized_inputs = group["normalized_inputs"]

            # Compute η' = η / (1 + η) for normalized case
            if normalized_inputs:
                eta_prime = lr / (1 + lr)
            else:
                eta_prime = lr

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Standard gradient update component
                p.data.add_(grad, alpha=-eta_prime)

                # Adaptive decay term: W_t (I - η'_t x_t x_t^T)
                # This is only applied when input_tensor is provided
                if input_tensor is not None and p.dim() >= 2:
                    # Compute outer product x_t x_t^T
                    if input_tensor.dim() == 1:
                        outer = torch.outer(input_tensor, input_tensor)
                    else:
                        # Batch case: average over batch
                        outer = input_tensor.T @ input_tensor / input_tensor.shape[0]

                    # Apply adaptive decay: W = W (I - η' x x^T)
                    identity = torch.eye(outer.shape[0], device=p.device, dtype=p.dtype)
                    decay_factor = identity - eta_prime * outer

                    # Reshape if needed
                    if p.shape[1] == decay_factor.shape[0]:
                        p.data = p.data @ decay_factor
                    elif p.shape[0] == decay_factor.shape[0]:
                        p.data = decay_factor @ p.data

        return loss


class DeltaMomentum(Optimizer):
    """
    Delta Momentum Optimizer.

    Extends momentum with L2-regression objective for the internal memory,
    resulting in gradient-dependent weight decay that helps the momentum
    decay or stop when needed.

    Update rule (Equations 48-49):
        W_{i+1} = W_i + m_{i+1}
        m_{i+1} = m_i (α_{i+1} - ∇L(W_i; x_i)^T ∇L(W_i; x_i)) - η_t P_i ∇L(W_i; x_i)

    The key difference from standard momentum:
        - Delta rule allows better memory management
        - Learns to forget some past gradients during optimization

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum factor (default: 0.9)
        weight_decay: L2 regularization (default: 0)
        use_preconditioning: Whether to use gradient preconditioning (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        use_preconditioning: bool = False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            use_preconditioning=use_preconditioning,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Get or initialize momentum buffer
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.zeros_like(p.data)

                m = param_state["momentum_buffer"]

                # Compute gradient-dependent decay: g^T g
                grad_norm_sq = (grad * grad).sum()

                # Delta momentum update:
                # m = m (α - g^T g) - η g
                # The (α - g^T g) term provides adaptive decay
                adaptive_decay = momentum - lr * grad_norm_sq
                adaptive_decay = adaptive_decay.clamp(min=0.0, max=1.0)

                m.mul_(adaptive_decay).add_(grad, alpha=-lr)

                # Update parameters
                p.data.add_(m)

        return loss


class M3Optimizer(Optimizer):
    """
    Multi-scale Momentum Muon (M3) Optimizer.

    Combines Adam-style adaptive learning with Muon's Newton-Schulz
    orthogonalization and a multi-scale momentum system (CMS for optimizers).

    From Algorithm 1 in the paper:
        1. Fast momentum (M^(1)) updated every step
        2. Slow momentum (M^(2)) updated every f steps
        3. Both passed through Newton-Schulz orthogonalization
        4. Aggregated with weighted sum

    This design provides:
        - Fast adaptation from high-frequency momentum
        - Long-term memory from low-frequency momentum
        - Better optimization landscape understanding

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.02)
        beta1: Fast momentum decay (default: 0.95)
        beta2: Second moment decay for normalization (default: 0.999)
        beta3: Slow momentum decay (default: 0.9)
        alpha: Weight for slow momentum (default: 0.1)
        eps: Numerical stability (default: 1e-8)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        chunk_size: Update frequency for slow momentum (default: 100)
        weight_decay: L2 regularization (default: 0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        beta2: float = 0.999,
        beta3: float = 0.9,
        alpha: float = 0.1,
        eps: float = 1e-8,
        ns_steps: int = 5,
        chunk_size: int = 100,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            ns_steps=ns_steps,
            chunk_size=chunk_size,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.step_count = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta3 = group["beta3"]
            alpha = group["alpha"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            chunk_size = group["chunk_size"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Skip if grad has NaN/Inf
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    continue

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Clip gradient for stability
                grad_norm = grad.norm()
                if grad_norm > 1.0:
                    grad = grad / grad_norm

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["m1"] = torch.zeros_like(p.data)  # Fast momentum
                    state["m2"] = torch.zeros_like(p.data)  # Slow momentum
                    state["v"] = torch.zeros_like(p.data)  # Second moment
                    state["grad_buffer"] = []  # Gradient buffer for slow update

                m1, m2, v = state["m1"], state["m2"], state["v"]

                # Update fast momentum (every step)
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update second moment (for normalization)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Buffer gradient for slow momentum
                state["grad_buffer"].append(grad.clone())

                # Update slow momentum (every chunk_size steps)
                if self.step_count % chunk_size == 0 and len(state["grad_buffer"]) > 0:
                    # Aggregate buffered gradients
                    grad_sum = torch.stack(state["grad_buffer"]).mean(
                        dim=0
                    )  # Use mean for stability
                    m2.mul_(beta3).add_(grad_sum, alpha=1 - beta3)
                    state["grad_buffer"] = []  # Clear buffer

                # Apply Newton-Schulz orthogonalization to both momentums
                if p.dim() >= 2:
                    o1 = newton_schulz(m1.view(p.shape[0], -1), ns_steps).view_as(m1)
                    o2 = newton_schulz(m2.view(p.shape[0], -1), ns_steps).view_as(m2)

                    # Handle NaN from Newton-Schulz
                    if torch.isnan(o1).any():
                        o1 = m1 / (m1.norm() + eps)
                    if torch.isnan(o2).any():
                        o2 = m2 / (m2.norm() + eps)
                else:
                    o1, o2 = m1, m2

                # Bias correction
                bias_correction1 = 1 - beta1**self.step_count
                bias_correction2 = 1 - beta2**self.step_count

                # Aggregate momentums
                combined = o1 / bias_correction1 + alpha * o2

                # Normalize by second moment (like Adam)
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Compute update
                update = combined / denom

                # Clip update for stability
                update_norm = update.norm()
                if update_norm > 1.0:
                    update = update / update_norm

                # Update parameters
                p.data.add_(update, alpha=-lr)

        return loss


class DeepMomentum(Optimizer):
    """
    Deep Momentum Gradient Descent (DMGD).

    Replaces linear momentum with an MLP to increase memory capacity
    for compressing past gradients.

    Update rule (Equation 50):
        W_{i+1} = W_i + m_{i+1}(u_i)
        m_{i+1} = α_{i+1} m_i - η_t ∇L^(2)(m_i; u_i, 1)

    where:
        - u_i = ∇L(W_i; x_i) (the gradient)
        - m(·) is an MLP that maps gradients to updates
        - L^(2) is the internal objective of momentum

    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum decay factor
        hidden_dim: Hidden dimension of momentum MLP
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.9, hidden_dim: int = 64):
        defaults = dict(lr=lr, momentum=momentum, hidden_dim=hidden_dim)
        super().__init__(params, defaults)

        # Create momentum networks for each parameter
        self._momentum_networks = {}

    def _get_momentum_network(self, p: torch.Tensor, hidden_dim: int) -> nn.Module:
        """Get or create momentum network for a parameter."""
        key = id(p)
        if key not in self._momentum_networks:
            in_dim = p.numel()
            self._momentum_networks[key] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim)
            ).to(p.device)
        return self._momentum_networks[key]

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            group["momentum"]
            hidden_dim = group["hidden_dim"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.flatten()

                # Get momentum network
                m_net = self._get_momentum_network(p, hidden_dim)

                # Compute momentum update
                with torch.enable_grad():
                    m_output = m_net(grad)

                # Update parameters
                p.data.add_(m_output.view_as(p.data), alpha=-lr)

        return loss


# Convenience function to create optimizer by name
def create_optimizer(name: str, params, **kwargs) -> Optimizer:
    """
    Create an optimizer by name.

    Args:
        name: Optimizer name ('dgd', 'm3', 'delta_momentum', 'deep_momentum', 'adam', 'sgd')
        params: Parameters to optimize
        **kwargs: Optimizer-specific arguments

    Returns:
        Optimizer instance
    """
    optimizers = {
        "dgd": DeltaGradientDescent,
        "m3": M3Optimizer,
        "delta_momentum": DeltaMomentum,
        "deep_momentum": DeepMomentum,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }

    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")

    return optimizers[name.lower()](params, **kwargs)
