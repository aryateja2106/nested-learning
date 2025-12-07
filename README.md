# Nested Learning: Implementation from Scratch

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)](https://www.python.org/)

> **Less Code, More Reproduction** - A LeCoder Project

This repository contains a from-scratch implementation of Google Research's "Nested Learning: The Illusion of Deep Learning Architecture" paper (NeurIPS 2025).

## 🎯 What is Nested Learning?

Nested Learning (NL) is a new learning paradigm that represents machine learning models as a set of **nested, multi-level optimization problems**, each with its own "context flow" and update frequency.

### Key Insights

1. **Optimizers are Associative Memories**: Adam, SGD with Momentum, etc., are associative memory modules that compress gradient information
2. **Architectures are Uniform**: All neural architectures can be decomposed into feedforward networks with different update frequencies
3. **Pre-training is In-Context Learning**: With an ultra-large context (the entire training data)
4. **Continuum Memory System**: Generalizes long-term/short-term memory with a spectrum of update frequencies

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Hope Architecture                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Self-Modifying Titans                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
│  │  │  M_key   │  │  M_value │  │ M_memory │           │   │
│  │  │ (adapt)  │  │ (adapt)  │  │  (adapt) │           │   │
│  │  └──────────┘  └──────────┘  └──────────┘           │   │
│  │         ↓ Delta Gradient Descent (DGD)               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Continuum Memory System (CMS)                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
│  │  │ MLP^f1  │→ │ MLP^f2  │→ │ MLP^f3  │→ │ MLP^fk  │ │   │
│  │  │ (high)  │  │  (mid)  │  │  (low)  │  │ (lowest)│ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nested-learning.git
cd nested-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🧰 Fast setup with `uv` (recommended)

```bash
# Install uv (one-line script)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies with uv's resolver
UV_PYTHON=.venv/bin/python uv pip install --python .venv/bin/python -r requirements.txt

# Run checks or scripts through uv
uv run pytest tests/test_components.py
uv run python demo/app.py
uv run python train_hope.py --config small --steps 500 --batch-size 8
```

### ☁️ Run on a free Colab GPU with `cgpu`

```bash
# Install cgpu once (opens a short setup wizard on first run)
npm i -g cgpu

# Optional: verify connectivity
cgpu status

# Sync the repo to Colab, install deps with uv, and run a GPU smoke test
bash run_cgpu_uv.sh
```

After the script completes you can drop into the same Colab runtime:

```bash
cgpu connect
cd /content/lecoder-nested-learning
source .venv/bin/activate
uv run python train_hope.py --config small --steps 2000
```

### 📌 Usage guidelines
- Prefer `uv` for installs and execution (`uv run …`) to keep environments fast and clean.
- Local GPU check: `python gpu_test.py`. Remote GPU: use the `cgpu` flow above.
- Training entrypoint: `train_hope.py --config {small,medium,large}` with flags for steps, batch size, and optimizer (`adamw`, `m3`, `dgd`).
- Core logic: `src/core` (optimizers, CMS) and `src/models` (Titans, Hope).
- Tests: `pytest tests/test_components.py`.

## 🚀 Quick Start

### 1. Run the Demo

```bash
# Launch the interactive Gradio demo
python demo/app.py
```

### 2. Train a Model

```bash
# Train Hope with the small preset (uses synthetic LM task)
python train_hope.py --config small --steps 500 --batch-size 8 --optimizer adamw
```

### 3. Use Individual Components

```python
from src.core.optimizers import DeltaGradientDescent, M3Optimizer
from src.core.memory import ContinuumMemorySystem
from src.models.hope import Hope

# Create a Hope model
model = Hope(
    d_model=512,
    n_layers=6,
    cms_levels=4,
    chunk_size=16
)

# Use M3 optimizer
optimizer = M3Optimizer(model.parameters(), lr=1e-4)
```

## 📚 Key Components

### 1. Delta Gradient Descent (DGD)
A new learning rule that updates weights based on both current input and weight state.

```python
# Equation 57 from the paper
W_{t+1} = W_t (I - η'_t x_t x_t^T) - η'_t ∇_y L(W_t; x_t) ⊗ x_t
```

### 2. Continuum Memory System (CMS)
A spectrum of MLP blocks with different update frequencies.

```python
# Different frequency levels
MLP^f1 → MLP^f2 → ... → MLP^fk
# Higher frequency = more adaptive, shorter memory
# Lower frequency = more persistent, longer memory
```

### 3. Multi-scale Momentum Muon (M3)
An optimizer with multiple momentum terms at different time scales.

```python
# Two-level momentum with Newton-Schulz orthogonalization
M^(1)_t = M^(1)_{t-1} + β1 * g_t        # Fast momentum
M^(2)_t = M^(2)_{t-1} + β3 * Σ g_i      # Slow momentum (updated every C steps)
```

### 4. Self-Modifying Titans
A self-referential learning module that generates its own values.

## 📂 Project Structure

```
nested-learning/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── optimizers.py      # DGD, M3, Delta Momentum
│   │   ├── memory.py          # CMS implementation
│   │   └── associative.py     # Base associative memory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── titans.py          # Self-Modifying Titans
│   │   └── hope.py            # Hope architecture
│   ├── experiments/
│   └── utils/
│       └── training.py
├── demo/
│   └── app.py                 # Gradio interactive demo
├── tests/
│   └── test_components.py
├── train_hope.py              # Training script with presets and AMP
├── run_cgpu_uv.sh             # Colab helper: cgpu + uv sync + GPU smoke test
├── gpu_test.py                # Local CUDA sanity check
├── docs/
│   └── ALGORITHMS.md          # Detailed algorithm explanations
├── requirements.txt
└── README.md
```

## 📖 Mathematical Foundation

### Associative Memory (Definition 1)

An operator M(·) that maps keys K to values V:
```
M* = argmin_M L(M(K); V)
```

### Nested System (Definition 3)

A system with K ordered levels, each with optimization problems:
```
θ^(k)_{i,t+1} = argmin_{Φ} ⟨Φ x_{t+1}, -∇L^(k)_i(θ^(k)_{it}; x_{t+1})⟩ + 1/(2η) ||Φ - θ^(k)_{it}||²
```

### Update Frequency (Definition 2)

For component A, frequency f_A = number of updates per unit time.

## 🧪 Experiments

### Continual Learning
- Class-incremental learning on CLINC, Banking, DBpedia
- Novel language translation (CTNL task)

### Long Context Understanding
- Needle-in-a-Haystack (NIAH) tasks
- BABILong benchmark
- RULER benchmark

### Language Modeling
- Wikitext perplexity
- Common-sense reasoning (PIQA, HellaSwag, etc.)

## 📊 Results

| Model | WikiText PPL | HellaSwag | PIQA | Avg |
|-------|-------------|-----------|------|-----|
| Transformer++ | 17.92 | 52.3 | 71.4 | 53.38 |
| Titans | 15.60 | 56.3 | 73.1 | 56.82 |
| **Hope** | **14.39** | **57.5** | **73.9** | **58.04** |

## 🌱 Open Source & Call for Contributors

This is an early, community-first reproduction of Nested Learning / HOPE—built by a product-minded tinkerer rather than an AI-native research lab. There is no widely published official code from the paper yet, so forks, issues, PRs, and stars are extra valuable for stress-testing the ideas. If you try the Colab `cgpu` flow or run `train_hope.py`, please share logs or improvements so others can learn from them.

## 🔗 Related Work

- [Titans: Learning to Memorize at Test Time](https://openreview.net/forum?id=8GjSf9Rh7Z)
- [TTT: Learning to (learn at test time)](https://arxiv.org/abs/2407.04620)
- [Linear Transformers](https://arxiv.org/abs/2006.16236)

## 📝 Citation

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architecture},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={NeurIPS},
  year={2025}
}
```

## 🙏 Acknowledgments

This implementation is part of the **LeCoder** project - "Less Code, More Reproduction".

Created by reproducing Google Research's Nested Learning paper from scratch using the LeCoder skill.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
