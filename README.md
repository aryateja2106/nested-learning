# Nested Learning: Implementation from Scratch

![Paper](https://img.shields.io/badge/Paper-Nested%20Learning-blue) ![Blog](https://img.shields.io/badge/Blog-Google%20Research-0b7285) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)

> Less Code, More Reproduction — a LeCoder project (skill file coming soon to help any coding agent run research code).  
> From-scratch reproduction of “Nested Learning: The Illusion of Deep Learning Architecture” (NeurIPS 2025) — paper: https://abehrouz.github.io/files/NL.pdf • blog: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

I built this to understand the paper’s algorithms end-to-end and welcome others (researchers, product folks, non-technical leaders, curious learners) to try it, fork it, and send feedback/PRs.

## Quick start
```bash
# Clone
git clone https://github.com/aryateja2106/nested-learning.git
cd nested-learning

# Fast setup with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv && source .venv/bin/activate
UV_PYTHON=.venv/bin/python uv pip install --python .venv/bin/python -r requirements.txt

# Smoke checks
uv run pytest tests/test_components.py
uv run python demo/app.py
uv run python train_hope.py --config small --steps 500 --batch-size 8
```
Prefer pip only? Use `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## GPU (optional)
- Tested on NVIDIA A100 via [`cgpu`](https://github.com/RohanAdwankar/cgpu); small configs and the notebook run on Colab L4/T4 or CPU for quick sanity.
- Colab from terminal:
  ```bash
  npm i -g cgpu
  cgpu status
  bash run_cgpu_uv.sh   # sync repo to Colab, install via uv, run GPU smoke test
  cgpu connect          # open a shell in the same Colab runtime
  ```

## Notebook
- `notebooks/quickstart.ipynb` — minimal forward/backward sanity check (CPU or GPU). Upload to Colab or run locally after installing deps.

## Architecture (HOPE)
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

## What is Nested Learning? (skim)
- Nested, multi-level optimization with per-level update frequencies (“context flow”).
- Optimizers as associative memories (Adam, SGD with momentum compress gradients).
- Architectures as uniform feedforward nets with differing update clocks.
- Pre-training as in-context learning over long contexts.
- CMS spans fast/slow memories to generalize long-/short-term storage.

## Train in one line
```bash
uv run python train_hope.py --config small  --steps 500  --optimizer adamw   # quick
uv run python train_hope.py --config medium --steps 2000 --optimizer m3     # mid
uv run python train_hope.py --config large  --steps 5000 --optimizer dgd    # bigger
```
Presets adjust model size/batch/seq length; AMP on by default. Reduce sizes for CPU.

## Project layout
```
src/core/          # optimizers, CMS
src/models/        # Titans, Hope
train_hope.py      # training entrypoint with presets
demo/app.py        # Gradio demo
tests/             # unit tests
notebooks/         # quickstart notebook
run_cgpu_uv.sh     # Colab helper using cgpu + uv
docs/ALGORITHMS.md # algorithm notes
requirements.txt
```

## For contributors & readers
- Please open issues/PRs, fork, or share logs; all backgrounds are welcome.
- Keep PRs small; include `pytest` output when changing code paths.
- If you’re curious how this was built with coding agents, reach out—happy to share the process. A LeCoder skill file will arrive soon.

## Acknowledgments
- Research: “Nested Learning: The Illusion of Deep Learning Architecture” (Behrouz, Razaviyayn, Zhong, Mirrokni).
- Blog: Google Research introduction linked above.
- Tools: [`cgpu`](https://github.com/RohanAdwankar/cgpu) for seamless Colab access from the terminal.
- Inspiration: the open-source community reproducing cutting-edge papers so others can learn faster.

## Security note
- Past commits contained credentials that have been removed from the tree; rotate/regenerate any exposed keys and avoid reusing them. `.gitignore` now excludes common secret patterns.
