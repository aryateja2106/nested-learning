# Nested Learning (HOPE) – Community Reproduction

Early, open-source implementation of the Nested Learning / HOPE ideas. Built by a product-focused tinkerer; “LeCoder skills” coming soon to bundle this into a friendlier toolkit.

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
- Local GPU sanity: `python gpu_test.py`
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
gpu_test.py        # local CUDA sanity check
```

## Hardware notes
- Validated on an NVIDIA A100 via `cgpu`; small presets and the notebook also run on Colab L4/T4 or CPU for smoke tests.

## Security & hygiene
- Removed files that contained client IDs/secrets. Those credentials are now public in git history—**rotate/regenerate them immediately** and avoid reusing them.
- `.gitignore` excludes common env/cache/logs and credential patterns; keep secrets out of the repo.

## Contributing
- Issues/PRs welcome; keep changes small and include `pytest` output when touching code paths.
- If you try the Colab flow, share logs so others can compare results.

## Attribution
- Inspired by the Nested Learning / HOPE paper (no official code release yet).
- Colab bridging via [`cgpu`](https://github.com/RohanAdwankar/cgpu).
