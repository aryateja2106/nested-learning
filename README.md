# Nested Learning: Implementation from Scratch

![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue) ![Paper%20PDF](https://img.shields.io/badge/PDF-NL.pdf-0b7285) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)

> Less Code, More Reproduction — a LeCoder project (skill file coming soon to help any coding agent run research code).  
> Built humbly to learn the paper end-to-end and invite others—researchers, product folks, and the simply curious—to explore, fork, and improve together.

**Paper & Blog**:  
- PDF: https://abehrouz.github.io/files/NL.pdf  
- Blog: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

---

## 🎯 What is Nested Learning?
Nested Learning (NL) views models as nested, multi-level optimization problems, each with its own “context flow” and update frequency.

Key insights:
- **Optimizers as associative memories**: Adam, SGD with momentum compress gradients.  
- **Uniform architecture**: Feedforward networks with different update clocks.  
- **Pre-training as in-context learning** over long contexts.  
- **Continuum Memory System (CMS)** spans fast/slow memories for long-/short-term storage.

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

---

## 📦 Installation
```bash
git clone https://github.com/aryateja2106/nested-learning.git
cd nested-learning
```

### 🧰 Fast setup with `uv` (recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv && source .venv/bin/activate
UV_PYTHON=.venv/bin/python uv pip install --python .venv/bin/python -r requirements.txt

# Smoke checks
uv run pytest tests/test_components.py
uv run python demo/app.py
uv run python train_hope.py --config small --steps 500 --batch-size 8
```
Prefer pip? `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

### ☁️ Optional Colab GPU via `cgpu`
[`cgpu`](https://github.com/RohanAdwankar/cgpu) opens a Colab GPU session from your terminal.
```bash
npm i -g cgpu
cgpu status
bash run_cgpu_uv.sh   # sync repo, install via uv, run GPU smoke test on Colab
cgpu connect          # shell into the same Colab runtime
```
Tested on A100; small configs and the notebook run on Colab L4/T4 or CPU for quick checks.

### 📓 Notebook
`notebooks/quickstart.ipynb` — minimal forward/backward sanity check (CPU or GPU). Upload to Colab or run locally after installing deps.

---

## 🚀 Quick Start
1) **Demo**  
```bash
python demo/app.py
```
2) **Train (presets)**  
```bash
uv run python train_hope.py --config small  --steps 500  --optimizer adamw   # quick
uv run python train_hope.py --config medium --steps 2000 --optimizer m3     # mid
uv run python train_hope.py --config large  --steps 5000 --optimizer dgd    # bigger
```
3) **Components**  
```python
from src.core.optimizers import DeltaGradientDescent, M3Optimizer
from src.core.memory import ContinuumMemorySystem
from src.models.hope import Hope

model = Hope(d_model=512, n_layers=6, cms_levels=4, chunk_size=16)
opt = M3Optimizer(model.parameters(), lr=1e-4)
```

---

## 📚 Key Components (skim)
- **Delta Gradient Descent (DGD)**: updates weights with an adaptive decay term tied to current input.  
  \(W_{t+1} = W_t (I - η'_t x_t x_t^T) - η'_t ∇_y L(W_t; x_t) ⊗ x_t\)
- **Continuum Memory System (CMS)**: spectrum of MLP blocks with different update frequencies (fast ↔ slow).
- **Multi-scale Momentum Muon (M3)**: fast + slow momentum with Newton-Schulz orthogonalization.
- **Self-Modifying Titans**: generates and updates its own memory values.

---

## 📂 Project Structure
```
src/core/          # optimizers, CMS
src/models/        # Titans, Hope
train_hope.py      # training entrypoint with presets (AMP on)
demo/app.py        # Gradio demo
tests/             # unit tests
notebooks/         # quickstart notebook
run_cgpu_uv.sh     # Colab helper using cgpu + uv
docs/ALGORITHMS.md # algorithm notes
requirements.txt
```

---

## 🤝 Welcome to contribute
- Fork, open issues/PRs, or share logs/results; all backgrounds are welcome.
- Keep PRs small and include `pytest` output when touching code paths.
- Curious how this was built with coding agents? Reach out—happy to share. A LeCoder skill file will follow.

---

## 🙏 Acknowledgments
- Research: “Nested Learning: The Illusion of Deep Learning Architecture” (Behrouz, Razaviyayn, Zhong, Mirrokni).  
- Blog: Google Research introduction (link above).  
- Tools: [`cgpu`](https://github.com/RohanAdwankar/cgpu) for seamless Colab-from-terminal access.  
- Inspiration: open-source efforts that make cutting-edge research runnable and teachable.

## 🔒 Security note
- Past commits contained credentials that are now removed; **rotate/regenerate any exposed keys**. `.gitignore` excludes common secret patterns—please keep secrets out of the repo.

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
