"""
Hope Architecture
==================

Implementation of Hope - A Self-Referential Learning Module with Continuum Memory.

From Section 8.3 of the paper:
    "We present Hope architecture: A neural learning module that incorporates 
    self-modifying Titans followed by Continuum Memory System."

Hope combines two complementary systems:
1. Self-Modifying Titans: Small capacity but expressive learning rule
2. CMS: Large capacity with simple learning rule for persistent knowledge

The combination enhances model expressiveness from different aspects.

Forward pass (Equations 94-97):
    k_t = M_k,t-1(x_t)
    v_t = M_v,t-1(x_t)
    η_t = M_η,t-1(x_t)
    α_t = M_α,t-1(x_t)
    o_t = M_memory,t-1(q_t)
    v̂_□,t = M_□,t-1(v_t)
    M_□,t = M_□,t-1(α I - η kk^T) - η ∇L(M_□,t-1; k_t, v̂_□,t)
    y_t = MLP^(fk)(MLP^(f(k-1))(...MLP^(f1)(o_t)))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import math

from .titans import SelfModifyingTitans, TitansConfig, TitansBlock
from ..core.memory import ContinuumMemorySystem, CMSConfig, create_cms


@dataclass
class HopeConfig:
    """Configuration for Hope architecture."""
    # Model dimensions
    d_model: int = 512
    d_hidden: int = 2048
    
    # Titans configuration
    d_key: int = 64
    d_value: int = 64
    num_heads: int = 8
    titans_chunk_size: int = 16
    titans_hidden: int = 256
    
    # CMS configuration
    cms_num_levels: int = 4
    cms_base_chunk_size: int = 16
    cms_aggregation: str = 'independent'
    
    # Architecture
    num_layers: int = 12
    dropout: float = 0.1
    
    # Attention variant
    use_attention_variant: bool = False  # Hope-Attention uses softmax instead of Titans
    
    # Convolution
    conv_kernel_size: int = 4
    
    # Vocabulary (for language modeling)
    vocab_size: int = 32000
    max_seq_len: int = 8192
    
    # Training
    tie_embeddings: bool = True


class HopeLayer(nn.Module):
    """
    Single Hope layer combining Self-Modifying Titans + CMS.
    
    Architecture:
        x → LayerNorm → Titans/Attention → + → LayerNorm → CMS → + → output
                           ↑                                  ↑
                           └──────── residual ────────────────┘
    """
    
    def __init__(self, config: HopeConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-norm layers
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Sequence mixing: Titans or Attention
        if config.use_attention_variant:
            self.seq_mixer = MultiHeadAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        else:
            titans_config = TitansConfig(
                d_model=config.d_model,
                d_key=config.d_key,
                d_value=config.d_value,
                d_hidden=config.titans_hidden,
                num_heads=config.num_heads,
                chunk_size=config.titans_chunk_size,
                dropout=config.dropout,
                conv_kernel_size=config.conv_kernel_size
            )
            self.seq_mixer = SelfModifyingTitans(titans_config)
        
        # CMS for persistent memory
        cms_config = CMSConfig(
            d_model=config.d_model,
            d_hidden=config.d_hidden,
            num_levels=config.cms_num_levels,
            base_chunk_size=config.cms_base_chunk_size,
            aggregation=config.cms_aggregation,
            dropout=config.dropout
        )
        self.cms = ContinuumMemorySystem(cms_config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_cms: bool = True
    ) -> torch.Tensor:
        """
        Forward pass (Equations 94-97).
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            attention_mask: Optional causal mask
            update_cms: Whether to update CMS memories
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-Modifying Titans (or Attention)
        # o_t = M_memory,t-1(q_t) with all the adaptive updates
        residual = x
        x = self.norm1(x)
        
        if hasattr(self.seq_mixer, 'forward') and 'attention_mask' in self.seq_mixer.forward.__code__.co_varnames:
            x = self.seq_mixer(x, attention_mask=attention_mask)
        else:
            x = self.seq_mixer(x)
        
        x = self.dropout(x)
        x = residual + x
        
        # Continuum Memory System
        # y_t = MLP^(fk)(MLP^(f(k-1))(...MLP^(f1)(o_t)))
        residual = x
        x = self.norm2(x)
        x = self.cms(x, update_memories=update_cms)
        x = self.dropout(x)
        x = residual + x
        
        return x
    
    def reset_memory(self, batch_size: int = 1):
        """Reset all memory states."""
        if hasattr(self.seq_mixer, 'reset_memory'):
            self.seq_mixer.reset_memory(batch_size)
        self.cms.reset_step_counter()


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention for Hope-Attention variant.
    Used when config.use_attention_variant = True.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
        scores = scores.masked_fill(attention_mask, float('-inf'))
        
        # Softmax and apply
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output


class Hope(nn.Module):
    """
    Hope: A Self-Referential Learning Module with Continuum Memory.
    
    This is the main architecture combining:
    1. Embedding layers (token + position)
    2. Stack of HopeLayers (Titans + CMS)
    3. Output head
    
    Key properties:
    - Self-modifying: Model can adapt its learning in-context
    - Continual learning: CMS enables persistent memory across contexts
    - Multi-scale: Different update frequencies for different knowledge types
    """
    
    def __init__(self, config: HopeConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Rotary Position Embedding (RoPE)
        self.rotary_emb = RotaryEmbedding(
            dim=config.d_key,
            max_seq_len=config.max_seq_len
        )
        
        # Input normalization
        self.embed_norm = nn.LayerNorm(config.d_model)
        
        # Hope layers
        self.layers = nn.ModuleList([
            HopeLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(config.d_model)
        
        # Output head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        update_cms: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            return_hidden_states: Whether to return all layer outputs
            update_cms: Whether to update CMS memories
        
        Returns:
            Dictionary with 'logits', optionally 'loss' and 'hidden_states'
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_norm(hidden_states)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
                diagonal=1
            )
        
        all_hidden_states = [hidden_states] if return_hidden_states else None
        
        # Process through Hope layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, update_cms)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        output = {'logits': logits}
        if loss is not None:
            output['loss'] = loss
        if return_hidden_states:
            output['hidden_states'] = all_hidden_states
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting token IDs (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding
        
        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                output = self(input_ids, update_cms=False)
                logits = output['logits'][:, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids
    
    def reset_memories(self, batch_size: int = 1):
        """Reset all memory states in all layers."""
        for layer in self.layers:
            layer.reset_memory(batch_size)
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'Hope':
        """Load pretrained model."""
        import json
        
        # Load config
        with open(f"{path}/config.json", 'r') as f:
            config_dict = json.load(f)
        config = HopeConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(f"{path}/model.pt", map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model
    
    def save_pretrained(self, path: str):
        """Save model."""
        import os
        import json
        
        os.makedirs(path, exist_ok=True)
        
        # Save config
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_')
        }
        with open(f"{path}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), f"{path}/model.pt")


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin
        t = torch.arange(max_seq_len)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def create_hope(
    d_model: int = 512,
    num_layers: int = 12,
    num_heads: int = 8,
    vocab_size: int = 32000,
    use_attention: bool = False
) -> Hope:
    """
    Factory function to create Hope model with sensible defaults.
    
    Args:
        d_model: Model dimension
        num_layers: Number of Hope layers
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
        use_attention: If True, use softmax attention instead of Titans
    
    Returns:
        Configured Hope model
    """
    config = HopeConfig(
        d_model=d_model,
        d_hidden=d_model * 4,
        d_key=d_model // num_heads,
        d_value=d_model // num_heads,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        use_attention_variant=use_attention
    )
    return Hope(config)


# Example usage
if __name__ == '__main__':
    # Create a small Hope model
    config = HopeConfig(
        d_model=256,
        d_hidden=1024,
        d_key=32,
        d_value=32,
        num_heads=8,
        num_layers=4,
        vocab_size=1000,
        cms_num_levels=2
    )
    
    model = Hope(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    
    # Test with labels
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids, labels=labels)
    print(f"Loss: {output['loss'].item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
