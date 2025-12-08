# Nested Learning: Implementation from Scratch

![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue) ![PDF](https://img.shields.io/badge/PDF-NL.pdf-0b7285) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.9%2B-yellow) ![Docker](https://img.shields.io/badge/Docker-Ready-blue)

> **Less Code, More Reproduction** — a LeCoder project  
> Built to learn Google Research's Nested Learning paper end-to-end and invite others—researchers, developers, product folks, and the simply curious—to explore, fork, and improve together.

**Paper & Blog**:  
- 📄 PDF: https://abehrouz.github.io/files/NL.pdf  
- 📝 Blog: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

---

## 🎯 What is Nested Learning?

**In plain English**: Nested Learning is a new way of thinking about deep learning models. Instead of viewing neural networks as fixed architectures, it treats them as **nested optimization problems** where different parts update at different speeds—like having fast, short-term memory and slow, long-term memory working together.

**For researchers**: Nested Learning (NL) views models as nested, multi-level optimization problems, each with its own "context flow" and update frequency. Key insights:
- **Optimizers as associative memories**: Adam, SGD with momentum compress gradients into memory.
- **Uniform architecture**: Feedforward networks with different update clocks.
- **Pre-training as in-context learning** over long contexts.
- **Continuum Memory System (CMS)**: Spectrum of fast/slow memories for long-/short-term storage.

---

## 🚀 Quick Start

### Option 1: Docker (Easiest)

```bash
# Clone the repository
git clone https://github.com/aryateja2106/nested-learning.git
cd nested-learning

# Run the interactive demo
docker compose up
# Opens at http://localhost:7860
```

### Option 2: UV (Recommended for Development)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv venv .venv && source .venv/bin/activate
UV_PYTHON=.venv/bin/python uv pip install --python .venv/bin/python -r requirements.txt

# Run tests
uv run pytest tests/test_components.py

# Launch demo
uv run python demo/app.py

# Train a small model
uv run python train_hope.py --config small --steps 500 --batch-size 8
```

### Option 3: Standard pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo/app.py
```

### 📓 Try the Notebook

Open `notebooks/quickstart.ipynb` in Jupyter or upload to [Google Colab](https://colab.research.google.com/). It runs a quick sanity check in under 2 minutes (works on CPU, faster on GPU).

---

## 🏗️ Architecture Overview (HOPE)

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

Hope combines **Self-Modifying Titans** (adaptive memory that learns how to learn) with **Continuum Memory System** (multi-frequency memory blocks).

---

## 📚 Key Components

- **Delta Gradient Descent (DGD)**: Updates weights with adaptive decay tied to current input.  
  \(W_{t+1} = W_t (I - η'_t x_t x_t^T) - η'_t ∇_y L(W_t; x_t) ⊗ x_t\)
- **Continuum Memory System (CMS)**: Spectrum of MLP blocks with different update frequencies (fast ↔ slow).
- **Multi-scale Momentum Muon (M3)**: Fast + slow momentum with Newton-Schulz orthogonalization.
- **Self-Modifying Titans**: Generates and updates its own memory values—meta-learning at inference time.

---

## 🔬 For Researchers & Developers

### Training Presets

```bash
# Quick test (runs in minutes)
uv run python train_hope.py --config small  --steps 500  --optimizer adamw

# Balanced run
uv run python train_hope.py --config medium --steps 2000 --optimizer m3

# Full training (GPU recommended)
uv run python train_hope.py --config large  --steps 5000 --optimizer dgd
```

### Using Components

```python
from src.core.optimizers import DeltaGradientDescent, M3Optimizer
from src.core.memory import ContinuumMemorySystem
from src.models.hope import Hope, HopeConfig

# Create model
config = HopeConfig(
    d_model=512,
    num_layers=6,
    cms_num_levels=4,
    cms_base_chunk_size=16
)
model = Hope(config)

# Use novel optimizers
optimizer = M3Optimizer(model.parameters(), lr=1e-4)
```

### ☁️ Colab GPU via `cgpu`

For seamless Colab GPU access from your terminal:

```bash
npm i -g cgpu
cgpu status
bash run_cgpu_uv.sh   # sync repo, install via uv, run GPU smoke test
cgpu connect          # shell into the same Colab runtime
```

Tested on A100; small configs run on Colab L4/T4 or CPU.

---

## 📂 Project Structure

```
src/core/          # optimizers, CMS
src/models/        # Titans, Hope
train_hope.py      # training entrypoint with presets (AMP on)
demo/app.py        # Gradio interactive demo
tests/             # unit tests
notebooks/         # quickstart notebook
docs/ALGORITHMS.md # algorithm notes
requirements.txt
```

---

## 🎁 Bonus: The Skill That Built This

This implementation was created using the **LeCoder Paper-to-Code Skill**—a methodology for AI agents to systematically implement research papers from scratch.

**Want to use it?** The skill is available in `.claude/skills/paper-to-code/`. Download it as a ZIP and upload to Claude or other AI agents to implement your own papers.

**What's included:**
- Complete workflow: PDF → Markdown → Algorithm → Code → Test
- Deep-dive guides on paper analysis and implementation patterns
- Best practices for packaging and testing

---

## 🤝 Contributing

- Fork, open issues/PRs, or share logs/results—all backgrounds welcome.
- Keep PRs small and include `pytest` output when touching code paths.
- Curious how this was built? The skill that created this is included—check `.claude/skills/paper-to-code/`.

---

## 🙏 Acknowledgments

- **Research**: "Nested Learning: The Illusion of Deep Learning Architecture" (Behrouz, Razaviyayn, Zhong, Mirrokni).  
- **Blog**: Google Research introduction (link above).  
- **Tools**: [`cgpu`](https://github.com/RohanAdwankar/cgpu) for seamless Colab-from-terminal access.  
- **Inspiration**: Open-source efforts that make cutting-edge research runnable and teachable.

---

## 🔒 Security Note

Past commits contained credentials that are now removed; **rotate/regenerate any exposed keys**. `.gitignore` excludes common secret patterns—please keep secrets out of the repo.

---

## 📜 Citation

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architecture},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={NeurIPS},
  year={2025}
}
```

---

**⭐ If this helped you, please star the repo!** It helps others discover this implementation and encourages more open-source research reproductions.
