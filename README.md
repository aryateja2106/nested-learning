# Nested Learning (HOPE) – open, humble reproduction

Paper: [Google Research blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) • License: MIT • Python: 3.9+

This repo is a learning-driven, from-scratch reproduction of the Nested Learning / HOPE ideas. It is shared early so researchers, product folks, and curious builders can explore, critique, and improve it together. “LeCoder” stands for *Less Code, More Implementation*; a skill file for coding agents will be shared soon to help translate research papers into runnable code.

## What is Nested Learning?
Nested Learning views models as nested, multi-level optimization problems, each with its own “context flow” and update frequency.

Key insights:
- Optimizers as associative memories (Adam, SGD with momentum compress gradients).
- Architectures as uniform feedforward networks with different update frequencies.
- Pre-training as in-context learning over a very long context.
- Continuum Memory System (CMS) spans fast/slow memories to generalize long-/short-term storage.

## Architecture overview (HOPE)
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

## How this repo is built
- PyTorch implementation of HOPE (Titans + CMS) under `src/models` and optimizers under `src/core`.
- Training script `train_hope.py` with small/medium/large presets; AMP on by default.
- Minimal notebook `notebooks/quickstart.ipynb` for a fast forward/backward sanity check.
- Optional Colab bridge via [`cgpu`](https://github.com/RohanAdwankar/cgpu); tested on A100, also runs on Colab L4/T4 for small configs.
- “LeCoder” skill (Less Code, More Implementation) coming soon as a `skill.md` for coding agents.

## What’s inside
- HOPE model (Titans + Continuum Memory System) in PyTorch
- Optimizers (M3, DGD, Delta Momentum) under `src/core/`
- Training script with presets: `train_hope.py`
- Minimal notebook: `notebooks/quickstart.ipynb`
- Optional Colab bridge via [`cgpu`](https://github.com/RohanAdwankar/cgpu) for GPU access from your terminal

## Quick start (uv)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create venv and install deps
uv venv .venv && source .venv/bin/activate
UV_PYTHON=.venv/bin/python uv pip install --python .venv/bin/python -r requirements.txt
# Sanity checks
uv run pytest tests/test_components.py
uv run python demo/app.py
uv run python train_hope.py --config small --steps 500 --batch-size 8
```
Prefer pip only? Use your usual `python -m venv` + `pip install -r requirements.txt`.

## Notebook quick run
- Open `notebooks/quickstart.ipynb` (works on CPU or GPU).
- If in Colab, upload the repo and run cells; if local, install deps first.

## GPU options
- Colab from your terminal (optional):
  ```bash
  npm i -g cgpu
  cgpu status         # check auth
  bash run_cgpu_uv.sh # sync repo, install with uv, run GPU smoke test
  cgpu connect        # drop into the same Colab runtime
  ```
  `cgpu` just opens a Colab GPU session through your terminal; useful if you want to train/debug remotely.

## Training
```bash
uv run python train_hope.py --config small  --steps 500  --optimizer adamw   # quick
uv run python train_hope.py --config medium --steps 2000 --optimizer m3     # bigger
uv run python train_hope.py --config large  --steps 5000 --optimizer dgd    # largest
```
Presets adjust model size, batch, and seq length. AMP is on by default; reduce sizes for CPU runs.

## Tests
```bash
uv run pytest tests/test_components.py
```

## Project layout
```
src/core/          # optimizers, CMS
src/models/        # Titans, Hope
train_hope.py      # training entrypoint with presets
demo/app.py        # Gradio demo
tests/             # unit tests
notebooks/         # quickstart notebook
run_cgpu_uv.sh     # Colab helper using cgpu + uv
```

## Hardware notes
- Validated on an NVIDIA A100 via `cgpu`; small presets and the notebook also run on Colab L4/T4 or CPU for smoke tests.

## Security & hygiene
- Removed files that contained client IDs/secrets. Those credentials are now public in git history—**rotate/regenerate them immediately** and avoid reusing them.
- `.gitignore` excludes common env/cache/logs and credential patterns; keep secrets out of the repo.

## Contributing
- All backgrounds welcome—research, product, business, or just curious. If you try it, please share logs/results so others can learn.
- Keep changes small and include `pytest` output when touching code paths.
- A LeCoder “skill” file will arrive soon to help coding agents turn research papers into runnable code.

## Attribution
- Based on the Nested Learning / HOPE paper and blog linked above (no official code release yet).
- Colab bridging via [`cgpu`](https://github.com/RohanAdwankar/cgpu).
