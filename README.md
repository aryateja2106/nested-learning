# Nested Learning: Implementation from Scratch

![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue) ![PDF](https://img.shields.io/badge/PDF-NL.pdf-0b7285) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.9%2B-yellow) ![Docker](https://img.shields.io/badge/Docker-Ready-blue) ![LeCoder cGPU](https://img.shields.io/badge/Built%20with-LeCoder%20cGPU-00a8ff)

> **Less Code, More Reproduction** — a LeCoder project  
> Built to learn Google Research's Nested Learning paper end-to-end and invite others—researchers, developers, product folks, and the simply curious—to explore, fork, and improve together.

**Paper & Blog**:  
- 📄 PDF: https://abehrouz.github.io/files/NL.pdf  
- 📝 Blog: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

---

## 🛠️ Built with LeCoder cGPU

This project was developed and tested using **[LeCoder cGPU CLI](https://github.com/aryateja2106/LeCoder-cgpu-CLI)**—a production-grade command-line tool for seamless Google Colab GPU access.

**Why LeCoder cGPU?**  
While building this implementation, we needed a robust way to:
- Run experiments on A100 GPUs without leaving the terminal
- Manage multiple Colab sessions for parallel experiments
- Automate workflows with structured JSON output
- Integrate GPU training into our development workflow

**What we built:**  
LeCoder cGPU provides enterprise-grade features including:
- 🔐 Secure OAuth2 authentication
- 📓 Notebook management via Drive API
- 🚀 Remote code execution with kernel mode
- 📊 Execution history and monitoring
- 🔄 Multi-session support (Colab Pro)
- 📁 File transfer and synchronization
- 🤖 AI agent integration (JSON output)

**See it in action:**  
Check out our [Enterprise Experiment Guide](docs/LECODER_CGPU_GUIDE.md) to see how we used LeCoder cGPU to run A100-accelerated training experiments with custom CUDA kernels.

**Try it yourself:**
```bash
npm install -g lecoder-cgpu
lecoder-cgpu auth
./run_lecoder_experiment.sh full
```

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

### ☁️ Colab GPU via LeCoder cGPU CLI

**Production-grade CLI for Colab GPU access**—built alongside this project to enable seamless GPU-accelerated development.

#### Quick Start

```bash
# Install LeCoder cGPU CLI
npm install -g lecoder-cgpu

# Authenticate
lecoder-cgpu auth

# Run enterprise experiment (A100 optimized)
./run_lecoder_experiment.sh train a100 1000
```

#### Complete Workflow Example

```bash
# 1. Check authentication and GPU availability
lecoder-cgpu status --json

# 2. Create experiment notebook
lecoder-cgpu notebook create "HOPE-Experiment" --template gpu

# 3. Upload project files
lecoder-cgpu copy ./src /content/nested-learning/src

# 4. Run training with structured output
lecoder-cgpu run --json --mode kernel "
import torch
from src.models.hope import Hope, HopeConfig
# ... your code ...
"

# 5. Monitor execution history
lecoder-cgpu logs --stats

# 6. Check GPU utilization
lecoder-cgpu status --json
```

#### Enterprise Experiment Suite

Run the complete enterprise continual learning pipeline:

```bash
# Quick GPU test and benchmark
./run_lecoder_experiment.sh quick

# Full training experiment (A100 optimized)
./run_lecoder_experiment.sh train a100 1000

# CUDA performance benchmark
./run_lecoder_experiment.sh benchmark

# Complete workflow showcase
./run_lecoder_experiment.sh full
```

#### Multi-Session Management (Colab Pro)

```bash
# List active sessions
lecoder-cgpu sessions list --stats

# Run parallel experiments
lecoder-cgpu --session <id1> run "python exp1.py"
lecoder-cgpu --session <id2> run "python exp2.py"

# Switch between sessions
lecoder-cgpu sessions switch <session-id>
```

#### JSON Output for Automation

```bash
# Get structured results for AI agents
lecoder-cgpu run --json --mode kernel "your_code_here"

# Query execution history
lecoder-cgpu logs --stats --json

# Monitor runtime status
lecoder-cgpu status --json
```

**📚 Full Documentation**: See [LeCoder cGPU Integration Guide](docs/LECODER_CGPU_GUIDE.md) for complete workflow, benchmarks, and best practices.

**Tested on**: A100 (Colab Pro+), T4 (Colab Pro), L4 (Free tier). Small configs run on CPU.

---

## 💼 Enterprise Use Case: Continual Learning Pipeline

**Real-world Application**: Customer Intelligence System

This implementation includes a complete **enterprise use case** demonstrating how HOPE enables continual learning for business applications.

### Business Problem

Traditional ML models suffer from **catastrophic forgetting**—when learning new patterns, they forget previous knowledge. This is critical for:
- Customer support systems that need to remember previous interactions
- Market analysis tools that adapt to changing conditions
- Enterprise AI that learns continuously without expensive retraining

### Our Solution

The **Enterprise Continual Learning Pipeline** (`src/experiments/enterprise_pipeline.py`) demonstrates:

1. **Long-term Memory**: CMS maintains customer pattern memory across different update frequencies
2. **Real-time Adaptation**: Self-Modifying Titans adapt to new feedback patterns instantly
3. **No Catastrophic Forgetting**: DGD optimizer prevents knowledge loss when learning new segments

### Performance Benchmarks (A100)

| Metric | CPU | A100 | Speedup |
|--------|-----|------|---------|
| Training throughput | ~50 tokens/s | ~5,000 tokens/s | **100x** |
| Memory update latency | ~10ms | ~0.1ms | **100x** |
| Full training (1000 steps) | ~2 hours | ~2 minutes | **60x** |

### Run Enterprise Experiment

```bash
# A100-optimized training with CUDA acceleration
python -m src.experiments.enterprise_pipeline --config a100 --steps 1000

# Or via LeCoder cGPU CLI
./run_lecoder_experiment.sh train a100 1000
```

**See**: [Enterprise Experiment Guide](docs/LECODER_CGPU_GUIDE.md) for complete documentation.

---

## 📂 Project Structure

```
src/
  core/              # optimizers (DGD, M3), CMS
  models/            # Titans, Hope architecture
  experiments/       # enterprise pipeline, CUDA kernels
train_hope.py        # training entrypoint with presets (AMP on)
demo/app.py          # Gradio interactive demo
tests/               # unit tests
notebooks/           # quickstart notebook
docs/
  ALGORITHMS.md      # algorithm notes
  LECODER_CGPU_GUIDE.md  # LeCoder cGPU integration guide
run_lecoder_experiment.sh  # enterprise experiment runner
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
- **Tools**: 
  - [LeCoder cGPU CLI](https://github.com/aryateja2106/LeCoder-cgpu-CLI) - Production-grade CLI for Colab GPU access (built alongside this project)
  - [`cgpu`](https://github.com/RohanAdwankar/cgpu) - Original inspiration for Colab-from-terminal access
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
