"""
Self-Modifying Titans
=====================

Implementation of Self-Modifying Deep Associative Memory (Titans) from
"Nested Learning: The Illusion of Deep Learning Architecture" paper.

Key innovation (Section 8.1):
    "Self-modifying deep associative memory where models generate their own values"

This allows the model to not just adapt to context, but actually modify its
own learning process - a form of meta-learning at test time.

From Equations 83-88:
    - Each memory module generates its own values: v̂_□,t = M_□,t-1(v_t)
    - Memories are updated using DGD: M_□,t = M_□,t-1(αI - η k k^T) - η ∇L
    - All components (k, v, q, η, α, memory) are adaptive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class TitansConfig:
    """Configuration for Self-Modifying Titans."""
    d_model: int = 512
    d_key: int = 64
    d_value: int = 64
    d_hidden: int = 256  # Hidden dim for memory MLPs
    num_heads: int = 8
    chunk_size: int = 16
    memory_chunk_size: int = 64
    dropout: float = 0.1
    use_l2_regression: bool = True  # vs dot-product similarity
    normalize_keys: bool = True
    conv_kernel_size: int = 4


class MemoryMLP(nn.Module):
    """
    MLP-based memory module (Equation 89):
        M(x) = x + W1 * σ(W2 * x)
    
    Used for all adaptive components: M_k, M_v, M_q, M_η, M_α, M_memory
    """
    
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_hidden: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        
        self.W2 = nn.Linear(d_input, d_hidden, bias=False)
        self.W1 = nn.Linear(d_hidden, d_output, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize near identity mapping
        nn.init.xavier_uniform_(self.W2.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W1.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        h = self.W2(x)
        h = F.silu(h)  # SiLU/Swish activation
        h = self.dropout(h)
        h = self.W1(h)
        
        # Residual connection if dimensions match
        if self.d_input == self.d_output:
            return x + h
        else:
            return h


class SelfModifyingTitans(nn.Module):
    """
    Self-Modifying Titans Layer.
    
    This is a self-referential learning module that generates its own values,
    allowing it to modify its own learning process in-context.
    
    Architecture (Equations 86-88):
        k_t = M_k,t-1(x_t)          # Adaptive key projection
        v_t = M_v,t-1(x_t)          # Adaptive value projection  
        η_t = M_η,t-1(x_t)          # Adaptive learning rate
        α_t = M_α,t-1(x_t)          # Adaptive retention/decay
        v̂_□,t = M_□,t-1(v_t)       # Self-generated values
        y_t = M_memory,t-1(q_t)     # Output retrieval
        
    Update rule (DGD with weight decay):
        M_□,t = M_□,t-1(α_t I - η_t k_t k_t^T) - η_t ∇L(M_□,t-1; k_t, v̂_□,t)
    """
    
    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_key = config.d_key
        self.d_value = config.d_value
        self.num_heads = config.num_heads
        
        # Query projection (non-adaptive, Equation 84)
        self.W_q = nn.Linear(config.d_model, config.d_key * config.num_heads, bias=False)
        
        # Adaptive memory modules for key, value, learning rate, decay
        self.M_k = MemoryMLP(config.d_model, config.d_key * config.num_heads, config.d_hidden, config.dropout)
        self.M_v = MemoryMLP(config.d_model, config.d_value * config.num_heads, config.d_hidden, config.dropout)
        self.M_eta = MemoryMLP(config.d_model, config.num_heads, config.d_hidden, config.dropout)
        self.M_alpha = MemoryMLP(config.d_model, config.num_heads, config.d_hidden, config.dropout)
        
        # Self-modification: memories that generate their own values
        self.M_v_hat_k = MemoryMLP(config.d_value * config.num_heads, config.d_key * config.num_heads, config.d_hidden, config.dropout)
        self.M_v_hat_v = MemoryMLP(config.d_value * config.num_heads, config.d_value * config.num_heads, config.d_hidden, config.dropout)
        self.M_v_hat_memory = MemoryMLP(config.d_value * config.num_heads, config.d_value * config.num_heads, config.d_hidden, config.dropout)
        
        # Main memory for storage and retrieval (matrix-valued)
        # Shape: (num_heads, d_value, d_key)
        self.register_buffer(
            'M_memory', 
            torch.zeros(config.num_heads, config.d_value, config.d_key)
        )
        
        # Output projection
        self.out_proj = nn.Linear(config.d_value * config.num_heads, config.d_model)
        
        # Optional: local convolution for token mixing
        if config.conv_kernel_size > 0:
            self.conv = nn.Conv1d(
                config.d_model, config.d_model, 
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.d_model
            )
        else:
            self.conv = None
        
        # Layer normalization
        self.norm_k = nn.LayerNorm(config.d_key * config.num_heads)
        self.norm_q = nn.LayerNorm(config.d_key * config.num_heads)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def reset_memory(self, batch_size: int = 1):
        """Reset memory state for new sequence."""
        self.M_memory.zero_()
    
    def _compute_adaptive_params(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute adaptive parameters from input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            k, v, q, eta, alpha tensors
        """
        batch_size, seq_len, _ = x.shape
        
        # Adaptive key: k_t = M_k,t-1(x_t)
        k = self.M_k(x)  # (batch, seq_len, d_key * num_heads)
        
        # Adaptive value: v_t = M_v,t-1(x_t)
        v = self.M_v(x)  # (batch, seq_len, d_value * num_heads)
        
        # Non-adaptive query: q_t = x_t W_q
        q = self.W_q(x)  # (batch, seq_len, d_key * num_heads)
        
        # Adaptive learning rate: η_t = M_η,t-1(x_t)
        eta = self.M_eta(x)  # (batch, seq_len, num_heads)
        eta = F.softplus(eta) * 0.1  # Keep learning rate positive and small
        
        # Adaptive retention/decay: α_t = M_α,t-1(x_t)
        alpha = self.M_alpha(x)  # (batch, seq_len, num_heads)
        alpha = torch.sigmoid(alpha)  # Keep in (0, 1)
        
        return k, v, q, eta, alpha
    
    def _compute_self_generated_values(
        self, 
        v: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate values for each memory module (self-modification).
        
        Equation 84: v̂_□,t = M_□,t-1(v_t)
        """
        return {
            'k': self.M_v_hat_k(v),
            'v': self.M_v_hat_v(v),
            'memory': self.M_v_hat_memory(v)
        }
    
    def _update_memory_dgd(
        self,
        k: torch.Tensor,
        v_hat: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Update memory using Delta Gradient Descent (Equation 88).
        
        DGD update:
            M_t = M_t-1 (α I - η k k^T) - η ∇L(M_t-1; k, v̂)
        
        For L2 regression loss (Equation 93):
            ∇L = (M k - v̂) k^T
        
        So:
            M_t = M_t-1 (α I - η k k^T) - η (M k - v̂) k^T
        
        Args:
            k: Keys (batch, num_heads, d_key)
            v_hat: Self-generated values (batch, num_heads, d_value)
            eta: Learning rate (batch, num_heads, 1)
            alpha: Retention (batch, num_heads, 1)
            memory: Current memory state (batch, num_heads, d_value, d_key)
        
        Returns:
            Updated memory
        """
        batch_size = k.shape[0]
        
        # Normalize keys if configured
        if self.config.normalize_keys:
            k = F.normalize(k, dim=-1)
        
        # Compute k k^T: (batch, num_heads, d_key, d_key)
        kk_T = torch.einsum('bhk,bhl->bhkl', k, k)
        
        # Adaptive decay term: M (α I - η k k^T)
        # First compute α I - η k k^T
        d_key = k.shape[-1]
        identity = torch.eye(d_key, device=k.device, dtype=k.dtype)
        identity = identity.unsqueeze(0).unsqueeze(0)  # (1, 1, d_key, d_key)
        
        # (batch, num_heads, d_key, d_key)
        decay_matrix = alpha.unsqueeze(-1) * identity - eta.unsqueeze(-1) * kk_T
        
        # Apply decay: M (decay_matrix)
        # memory: (batch, num_heads, d_value, d_key)
        # decay_matrix: (batch, num_heads, d_key, d_key)
        decayed_memory = torch.einsum('bhvk,bhkl->bhvl', memory, decay_matrix)
        
        if self.config.use_l2_regression:
            # L2 regression gradient: (M k - v̂) k^T
            # M k: (batch, num_heads, d_value)
            retrieved = torch.einsum('bhvk,bhk->bhv', memory, k)
            # Error: (batch, num_heads, d_value)
            error = retrieved - v_hat
            # Gradient: error k^T -> (batch, num_heads, d_value, d_key)
            gradient = torch.einsum('bhv,bhk->bhvk', error, k)
        else:
            # Dot-product similarity gradient: -v̂ k^T
            gradient = -torch.einsum('bhv,bhk->bhvk', v_hat, k)
        
        # Final update
        updated_memory = decayed_memory - eta.unsqueeze(-1) * gradient
        
        return updated_memory
    
    def forward(
        self,
        x: torch.Tensor,
        return_memory_states: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through Self-Modifying Titans.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            return_memory_states: Whether to return intermediate memory states
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply local convolution if configured
        if self.conv is not None:
            x_conv = self.conv(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            x = x + x_conv
        
        # Compute adaptive parameters
        k, v, q, eta, alpha = self._compute_adaptive_params(x)
        
        # Reshape for multi-head processing
        k = k.view(batch_size, seq_len, self.num_heads, self.d_key)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_value)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_key)
        eta = eta.view(batch_size, seq_len, self.num_heads, 1)
        alpha = alpha.view(batch_size, seq_len, self.num_heads, 1)
        
        # Transpose for easier processing: (batch, num_heads, seq_len, dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q.transpose(1, 2)
        eta = eta.transpose(1, 2)
        alpha = alpha.transpose(1, 2)
        
        # Normalize queries and keys
        q = self.norm_q(q.transpose(1, 2).reshape(batch_size, seq_len, -1))
        q = q.view(batch_size, seq_len, self.num_heads, self.d_key).transpose(1, 2)
        
        k = self.norm_k(k.transpose(1, 2).reshape(batch_size, seq_len, -1))
        k = k.view(batch_size, seq_len, self.num_heads, self.d_key).transpose(1, 2)
        
        # Initialize memory for this batch
        memory = self.M_memory.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        
        outputs = []
        memory_states = [] if return_memory_states else None
        
        # Process chunk by chunk for efficiency
        chunk_size = self.config.chunk_size
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            
            # Get chunk data
            k_chunk = k[:, :, start:end, :]  # (batch, heads, chunk, d_key)
            v_chunk = v[:, :, start:end, :]  # (batch, heads, chunk, d_value)
            q_chunk = q[:, :, start:end, :]  # (batch, heads, chunk, d_key)
            eta_chunk = eta[:, :, start:end, :]  # (batch, heads, chunk, 1)
            alpha_chunk = alpha[:, :, start:end, :]  # (batch, heads, chunk, 1)
            
            chunk_outputs = []
            
            for t in range(end - start):
                # Get single timestep
                k_t = k_chunk[:, :, t, :]  # (batch, heads, d_key)
                v_t = v_chunk[:, :, t, :]  # (batch, heads, d_value)
                q_t = q_chunk[:, :, t, :]  # (batch, heads, d_key)
                eta_t = eta_chunk[:, :, t, :]  # (batch, heads, 1)
                alpha_t = alpha_chunk[:, :, t, :]  # (batch, heads, 1)
                
                # Retrieve from memory: y_t = M_memory(q_t)
                # memory: (batch, heads, d_value, d_key)
                # q_t: (batch, heads, d_key)
                y_t = torch.einsum('bhvk,bhk->bhv', memory, q_t)  # (batch, heads, d_value)
                chunk_outputs.append(y_t)
                
                # Compute self-generated values
                v_flat = v_t.view(batch_size, -1)  # (batch, heads * d_value)
                v_hat_memory = self.M_v_hat_memory(v_flat)
                v_hat_memory = v_hat_memory.view(batch_size, self.num_heads, self.d_value)
                
                # Update memory using DGD
                memory = self._update_memory_dgd(k_t, v_hat_memory, eta_t, alpha_t, memory)
                
                if return_memory_states:
                    memory_states.append(memory.clone())
            
            # Stack chunk outputs
            chunk_output = torch.stack(chunk_outputs, dim=2)  # (batch, heads, chunk, d_value)
            outputs.append(chunk_output)
        
        # Concatenate all outputs
        output = torch.cat(outputs, dim=2)  # (batch, heads, seq_len, d_value)
        
        # Reshape and project
        output = output.transpose(1, 2)  # (batch, seq_len, heads, d_value)
        output = output.reshape(batch_size, seq_len, -1)  # (batch, seq_len, heads * d_value)
        output = self.out_proj(output)  # (batch, seq_len, d_model)
        output = self.dropout(output)
        
        if return_memory_states:
            return output, memory_states
        return output


class TitansBlock(nn.Module):
    """
    Complete Titans block with normalization and residual connection.
    """
    
    def __init__(self, config: TitansConfig):
        super().__init__()
        self.titans = SelfModifyingTitans(config)
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with pre-norm and residual."""
        return x + self.titans(self.norm(x))


def create_titans(
    d_model: int = 512,
    d_key: int = 64,
    d_value: int = 64,
    num_heads: int = 8,
    chunk_size: int = 16
) -> SelfModifyingTitans:
    """Factory function for creating Titans with sensible defaults."""
    config = TitansConfig(
        d_model=d_model,
        d_key=d_key,
        d_value=d_value,
        num_heads=num_heads,
        chunk_size=chunk_size
    )
    return SelfModifyingTitans(config)


# Example usage
if __name__ == '__main__':
    config = TitansConfig(
        d_model=256,
        d_key=32,
        d_value=32,
        num_heads=4,
        chunk_size=8
    )
    
    titans = SelfModifyingTitans(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    output = titans(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with memory states
    output, memory_states = titans(x, return_memory_states=True)
    print(f"Number of memory states: {len(memory_states)}")
    print(f"Memory state shape: {memory_states[0].shape}")
