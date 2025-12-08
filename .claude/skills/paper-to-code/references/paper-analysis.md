# Paper Analysis Deep Dive

Systematic approach to extracting implementable algorithms from research papers.

## Paper Structure Recognition

### Typical ML Paper Structure
```
1. Abstract          → Key claims, results summary
2. Introduction      → Problem statement, contributions
3. Related Work      → Context, but rarely implementation details
4. Method/Approach   → CORE: Algorithms, equations, architecture
5. Experiments       → Hyperparameters, baselines, datasets
6. Results           → Benchmarks to reproduce
7. Conclusion        → Summary
8. Appendix          → Often contains critical implementation details
```

### Where to Find What

| Need | Look In |
|------|---------|
| Algorithm pseudocode | Method section, Algorithm boxes |
| Equations | Method section, numbered equations |
| Architecture diagrams | Method section figures |
| Hyperparameters | Experiments section, Appendix tables |
| Training details | Experiments section, Appendix |
| Ablations | Results section |
| Proofs (for verification) | Appendix |

## Extraction Patterns

### Algorithm Boxes

Papers often present algorithms in structured boxes:

```
Algorithm 1: Name
Input: x, parameters θ
Output: y

1: Initialize M ← 0
2: for t = 1 to T do
3:   Compute gradient g_t
4:   Update M ← βM + (1-β)g_t
5:   θ ← θ - η * M
6: end for
7: return θ
```

**Convert to code**:
```python
def algorithm_name(x, theta, T, beta, eta):
    M = 0
    for t in range(1, T + 1):
        g_t = compute_gradient(theta, x)
        M = beta * M + (1 - beta) * g_t
        theta = theta - eta * M
    return theta
```

### Equation Extraction

Common equation patterns and their PyTorch equivalents:

| Paper Notation | PyTorch |
|---------------|---------|
| `x^T y` | `torch.dot(x, y)` or `x @ y` |
| `X Y` (matrix mult) | `X @ Y` or `torch.matmul(X, Y)` |
| `x ⊗ y` (outer product) | `torch.outer(x, y)` |
| `x ⊙ y` (elementwise) | `x * y` |
| `‖x‖` | `torch.norm(x)` |
| `σ(x)` | `torch.sigmoid(x)` or `F.silu(x)` |
| `softmax(x)` | `F.softmax(x, dim=-1)` |
| `∇_θ L` | `loss.backward(); param.grad` |

### Architecture Figures

When extracting from architecture diagrams:

1. **Identify data flow**: Follow arrows for forward pass
2. **Note dimensions**: Look for d_model, d_hidden annotations
3. **Identify skip connections**: Residual paths
4. **Find normalization**: LayerNorm positions (pre-norm vs post-norm)
5. **Note activation functions**: GELU, SiLU, ReLU

## Common Paper Patterns

### Transformer Variants

Most transformer papers modify:
- Attention mechanism (linear, local, sparse)
- Position encoding (learned, rotary, ALiBi)
- Normalization (pre-norm, post-norm, RMSNorm)
- FFN structure (GLU variants, MoE)

**Standard transformer block**:
```python
def forward(self, x):
    # Pre-norm pattern (most modern papers)
    x = x + self.attn(self.norm1(x))
    x = x + self.ffn(self.norm2(x))
    return x
```

### Memory Systems

Papers with memory often have:
- Key-value stores
- Update rules (write operations)
- Retrieval rules (read operations)
- Forgetting/decay mechanisms

**Pattern**:
```python
class Memory:
    def write(self, key, value):
        # Update memory with new key-value
        pass
    
    def read(self, query):
        # Retrieve from memory
        pass
    
    def forget(self, decay_rate):
        # Apply decay to old memories
        pass
```

### Optimizer Papers

Optimizer papers typically present:
- Momentum formulation
- Adaptive learning rate (if any)
- Update rule
- Hyperparameter defaults

**Standard optimizer structure**:
```python
class CustomOptimizer(Optimizer):
    def __init__(self, params, lr, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Get state
                state = self.state[p]
                # Initialize state if needed
                # Apply update rule
```

## Handling Ambiguity

### Missing Details

Papers often omit:
- Initialization schemes (assume Xavier/He)
- Exact layer dimensions (derive from context)
- Training batch sizes (check experiments)
- Learning rate schedules (often cosine decay)

**Default assumptions**:
```python
# Weight initialization
nn.init.xavier_uniform_(self.weight)
nn.init.zeros_(self.bias)

# LayerNorm
self.norm = nn.LayerNorm(d_model, eps=1e-6)

# Dropout (if not specified)
self.dropout = nn.Dropout(0.1)
```

### Conflicting Information

When paper sections conflict:
1. Trust Algorithm boxes over prose
2. Trust Appendix over main text
3. Trust Tables over text descriptions
4. Check supplementary material

### Notation Inconsistencies

Papers sometimes use different notation in different sections:
- Make a notation glossary while reading
- Map paper notation to code variable names
- Document mappings in code comments

## Verification Strategies

### Unit Test Each Equation

```python
def test_equation_57():
    """
    Verify Equation 57: W' = W(I - η x x^T) - η ∇L ⊗ x
    """
    W = torch.randn(4, 4)
    x = torch.randn(4)
    eta = 0.1
    grad = torch.randn(4)
    
    # Manual computation
    I = torch.eye(4)
    decay = I - eta * torch.outer(x, x)
    expected = W @ decay - eta * torch.outer(grad, x)
    
    # Function under test
    result = delta_update(W, x, eta, grad)
    
    assert torch.allclose(result, expected, atol=1e-6)
```

### Reproduce Paper Figures

If paper shows training curves or attention patterns:
1. Implement visualization
2. Run small-scale experiment
3. Compare qualitative behavior

### Ablation Sanity Checks

Run paper's ablations to verify:
- Removing component X hurts performance
- Hyperparameter sensitivity matches
- Scaling behavior is similar

## Red Flags in Papers

Watch for:
- **No code release promised**: Implementation may be tricky
- **Unusually good results**: May require specific hyperparameters
- **Vague training details**: Missing batch size, learning rate, etc.
- **Single dataset**: May not generalize
- **No ablations**: Hard to verify components work

## Tools for Analysis

### PDF Search Commands

```bash
# After converting to markdown
# Find all numbered equations
grep -n -E "^\s*\([0-9]+\)" paper.md

# Find algorithm blocks
grep -n -A 20 "Algorithm [0-9]" paper.md

# Find hyperparameter tables
grep -n -B 2 -A 10 "Hyperparameter\|learning rate\|batch size" paper.md

# Find architecture details
grep -n -i "layer\|block\|head\|dimension" paper.md
```

### Cross-Reference with Citations

Check papers that cite your target paper:
- May have clearer explanations
- May report bugs/errata
- May have open implementations

### Author's Previous Work

Authors often reuse patterns:
- Check their GitHub for related code
- Earlier papers may have clearer explanations
- Workshop versions may have different details
