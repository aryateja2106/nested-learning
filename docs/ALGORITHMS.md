# Nested Learning Algorithms

Detailed mathematical formulations from "Nested Learning: The Illusion of Deep Learning Architecture" by Behrouz et al. (NeurIPS 2025).

## Table of Contents

1. [Core Definitions](#core-definitions)
2. [Delta Gradient Descent (DGD)](#delta-gradient-descent-dgd)
3. [Multi-scale Momentum Muon (M3)](#multi-scale-momentum-muon-m3)
4. [Continuum Memory System (CMS)](#continuum-memory-system-cms)
5. [Self-Modifying Titans](#self-modifying-titans)
6. [Hope Architecture](#hope-architecture)

---

## Core Definitions

### Definition 1: Associative Memory

Given a set of keys K ⊆ ℝ^{d_k} and values V ⊆ ℝ^{d_v}, associative memory is an operator M(·) that maps keys K to values V:

```
M* = argmin_M L̃(M(K); V)
```

where L̃(·;·) measures the quality of the mapping.

### Definition 2: Update Frequency

For any component A, the update frequency f_A is defined as the number of times A is updated per unit of time.

### Definition 3: Nested System

A nested system is a system with K ordered levels where each level k (1 ≤ k ≤ K) has:
- A set of optimization problems with parameters θ^(k)
- Its own context flow
- Update frequency f_k

The optimization at level k is:

```
θ^(k)_{i,t+1} = argmin_Φ ⟨Φ x_{t+1}, -∇L^(k)_i(θ^(k)_{it}; x_{t+1})⟩ + (1/2η)||Φ - θ^(k)_{it}||²
```

---

## Delta Gradient Descent (DGD)

### Motivation

Standard gradient descent treats each data sample independently. DGD incorporates the current weight state into the update, capturing dependencies without i.i.d. assumptions.

### Formulation (Equation 56-57)

Given L2 regression objective:

```
W_{t+1} = argmin_W (1/2)||W x_t - u_t||²_2 + (1/2η_t)||W - W_t||²_2
```

where u_t = -∇_{y_t} L(W_t; x_t) is the local surprise signal.

For normalized inputs (||x_t||₂ = λ), using Sherman-Morrison lemma:

```
W_{t+1} = W_t(I - η'_t x_t x_t^T) - η'_t ∇_{y_t}L(W_t; x_t) ⊗ x_t
```

where:
- η'_t = η_t / (1 + η_t)
- First term: **Adaptive decay** based on current input
- Second term: Standard gradient update

### Key Properties

1. **Input-dependent decay**: Weight decay varies with input x_t
2. **State incorporation**: Update depends on current W_t state
3. **Non-i.i.d. handling**: Captures dependencies between samples

### Pseudocode

```python
def dgd_step(W, x, grad_y, lr, normalized=True):
    if normalized:
        eta_prime = lr / (1 + lr)
    else:
        eta_prime = lr
    
    # Adaptive decay: W(I - η' x x^T)
    outer_product = torch.outer(x, x)
    decay_matrix = torch.eye(x.shape[0]) - eta_prime * outer_product
    W_decayed = W @ decay_matrix
    
    # Gradient update
    W_new = W_decayed - eta_prime * torch.outer(grad_y, x)
    
    return W_new
```

---

## Multi-scale Momentum Muon (M3)

### Motivation

Combines CMS design with optimizer momentum to enable:
- Fast adaptation from high-frequency momentum
- Long-term memory from low-frequency momentum
- Better optimization landscape understanding

### Formulation (Algorithm 1, Equations 75)

Two-level momentum system:

```
M^(1)_t = M^(1)_{t-1} + β₁ g_t                    # Fast momentum (every step)

M^(2)_t = M^(2)_{t-1} + β₃ Σ_{i=t-Ĉ}^t g_i       # Slow momentum (every Ĉ steps)
```

Orthogonalization via Newton-Schulz:

```
O^(1)_t = NewtonSchulz_T(M^(1)_t)
O^(2)_t = NewtonSchulz_T(M^(2)_t)
```

Parameter update:

```
Θ_t = Θ_{t-1} - η (O^(1)_t + α O^(2)_t) / (√V_t + ε)
```

### Newton-Schulz Iteration

Maps gradients to orthogonal space:

```
O_{i+1} = O_i - ζ ∇L(O_i; g)
```

where L(O; g) = ||O^T O - I||²_F

Resulting update:

```
O_{i+1} = O_i(3I - O_i^T O_i) / 2
```

### Pseudocode

```python
def m3_step(params, grads, state, config):
    # Fast momentum update
    state.m1 = state.m1 + config.beta1 * grads
    
    # Second moment (for normalization)
    state.v = state.v + config.beta2 * grads**2
    
    # Slow momentum update (every chunk_size steps)
    if state.step % config.chunk_size == 0:
        grad_sum = sum(state.grad_buffer)
        state.m2 = state.m2 + config.beta3 * grad_sum
        state.grad_buffer = []
    else:
        state.grad_buffer.append(grads)
    
    # Newton-Schulz orthogonalization
    o1 = newton_schulz(state.m1, steps=config.ns_steps)
    o2 = newton_schulz(state.m2, steps=config.ns_steps)
    
    # Aggregate and update
    combined = o1 + config.alpha * o2
    params = params - config.lr * combined / (sqrt(state.v) + config.eps)
    
    return params, state
```

---

## Continuum Memory System (CMS)

### Motivation

Generalizes long-term/short-term memory by viewing memory as a distributed system with spectrum of update frequencies.

### Formulation (Equations 70-71)

Chain of MLP blocks with different frequencies:

```
y_t = MLP^(f_k)(MLP^(f_{k-1})(... MLP^(f_1)(x_t)))
```

Update rule for level ℓ:

```
θ^(f_ℓ)_{i+1} = θ^(f_ℓ)_i - Σ_{t=i-C^(ℓ)}^i η^(ℓ)_t f(θ^(f_ℓ)_t; x_t)   if i ≡ 0 (mod C^(ℓ))
              = θ^(f_ℓ)_i                                               otherwise
```

where:
- C^(ℓ) is the chunk size for level ℓ
- Higher frequency → smaller chunk size → faster adaptation
- Lower frequency → larger chunk size → more persistent memory

### Variants

**Sequential CMS** (Equation 70):
```
y_t = MLP^(f_k)(MLP^(f_{k-1})(... MLP^(f_1)(x_t)))
```

**Independent CMS** (Equation 74):
```
y_t = Agg(MLP^(f_k)(x_t), MLP^(f_{k-1})(x_t), ..., MLP^(f_1)(x_t))
```

**Nested CMS** (Equation 72):
Initial state of level s+1 is meta-learned from level s.

### Why CMS Helps with Continual Learning

When updating MLP^(f_s), forgotten knowledge is still stored in MLP^(f_{s'}) where s' < s. Knowledge transfer through backpropagation can "circle back" the knowledge.

---

## Self-Modifying Titans

### Motivation

Allow the model to generate its own learning signals, enabling self-modification during inference.

### Formulation (Equations 83-88)

Adaptive projections:
```
k_t = M_{k,t-1}(x_t)        # Adaptive key
v_t = M_{v,t-1}(x_t)        # Adaptive value
η_t = M_{η,t-1}(x_t)        # Adaptive learning rate
α_t = M_{α,t-1}(x_t)        # Adaptive retention
q_t = x_t W_q               # Non-adaptive query
```

Self-generated values (self-modification):
```
v̂_{□,t} = M_{□,t-1}(v_t)   for □ ∈ {k, v, q, η, α, memory}
```

Memory update with DGD:
```
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t ∇L_{M_{□,t-1}}(M_{□,t-1}; k_t, v̂_{□,t})
```

Output retrieval:
```
y_t = M_{memory,t-1}(q_t)
```

### Update Rules for Different Objectives

**Dot-product similarity** L(M; k, v) = -⟨Mk, v⟩:
```
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t v̂_{□,t} k_t^T
```

**L2 regression** L(M; k, v) = ||Mk - v||²:
```
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t (M_{□,t-1} k_t - v̂_{□,t}) k_t^T
```

### Memory Architecture (Equation 89)

Each memory is a 2-layer MLP with residual:
```
M_□(x) = x + W_{□,1} σ(W_{□,2} x)
```

---

## Hope Architecture

### Overview

Hope = Self-Modifying Titans + Continuum Memory System

Combines:
- **Titans**: Small capacity, expressive learning rule
- **CMS**: Large capacity, simple learning rule

### Forward Pass (Equations 94-97)

```
# Self-Modifying Titans
k_t = M_{k,t-1}(x_t)
v_t = M_{v,t-1}(x_t)
η_t = M_{η,t-1}(x_t)
α_t = M_{α,t-1}(x_t)
v̂_{□,t} = M_{□,t-1}(v_t)
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t ∇L_{M_{□,t-1}}(M_{□,t-1}; k_t, v̂_{□,t})
o_t = M_{memory,t-1}(q_t)

# Continuum Memory System
y_t = MLP^(f_k)(MLP^(f_{k-1})(... MLP^(f_1)(o_t)))
```

### Hope-Attention Variant

Replace Self-Modifying Titans with standard softmax attention:
```
o_t = Softmax(QK^T / √d) V
y_t = MLP^(f_k)(MLP^(f_{k-1})(... MLP^(f_1)(o_t)))
```

### Architecture Diagram

```
Input x_t
    │
    ▼
┌───────────────────────────────────────┐
│     Self-Modifying Titans             │
│  ┌─────────────────────────────────┐  │
│  │ Adaptive: M_k, M_v, M_η, M_α    │  │
│  │ Self-modification: v̂ generation │  │
│  │ DGD update: M_{□,t}             │  │
│  │ Output: o_t = M_memory(q_t)     │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│     Continuum Memory System           │
│  ┌───────┐ ┌───────┐      ┌───────┐  │
│  │MLP^f1 │→│MLP^f2 │→ ... │MLP^fk │  │
│  │(fast) │ │       │      │(slow) │  │
│  └───────┘ └───────┘      └───────┘  │
└───────────────────────────────────────┘
    │
    ▼
Output y_t
```

---

## Implementation Notes

### Chunk-wise Training

For efficiency, process sequences in chunks:
1. Generate all k_t, v_t, η_t, α_t, v̂_t for chunk in parallel
2. Process chunk sequentially for memory updates
3. Take gradients w.r.t. last state of previous chunk

### Hyperparameters

| Component | Parameter | Typical Value |
|-----------|-----------|---------------|
| DGD | η (learning rate) | 0.01 - 0.1 |
| M3 | β₁ (fast momentum) | 0.95 |
| M3 | β₃ (slow momentum) | 0.9 |
| M3 | α (slow weight) | 0.1 |
| M3 | Ĉ (chunk size) | 100 |
| CMS | num_levels | 4 |
| CMS | base_chunk_size | 16 |
| Titans | chunk_size | 16 |
| Hope | num_layers | 12 |

---

## References

1. Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). Nested Learning: The Illusion of Deep Learning Architecture. NeurIPS.
2. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. ICLR.
3. Jordan, K., et al. (2024). Muon: A Scalable Mutually-Orthogonal Update. arXiv.
4. Sun, Y., et al. (2024). Learning to (Learn at Test Time). arXiv.
