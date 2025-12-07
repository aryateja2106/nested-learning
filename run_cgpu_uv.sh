#!/usr/bin/env bash
# End-to-end script to run this repo on a Colab GPU via cgpu + uv.
# Assumes `cgpu` is installed and authenticated locally.

set -euo pipefail

REPO_NAME="lecoder-nested-learning"
ARCHIVE="/tmp/${REPO_NAME}.tar.gz"
REMOTE_DIR="/content/${REPO_NAME}"

echo "[1/5] Packing repository..."
tar -czf "$ARCHIVE" .

echo "[2/5] Uploading to Colab via cgpu copy..."
cgpu copy "$ARCHIVE"

echo "[3/5] Setting up project directory on Colab..."
cgpu run "rm -rf ${REMOTE_DIR} && mkdir -p ${REMOTE_DIR} && tar -xzf /content/${REPO_NAME}.tar.gz -C ${REMOTE_DIR}"

echo "[4/5] Installing uv and project deps inside Colab venv..."
cgpu run "cd ${REMOTE_DIR} && pip install -U uv && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt"

echo "[5/5] Running smoke tests inside uv env (GPU if available)..."
cgpu run "cd ${REMOTE_DIR} && UV_PYTHON=.venv/bin/python uv run --python .venv/bin/python python - <<'PY'
import torch
from src.models.hope import Hope, HopeConfig

torch.manual_seed(0)
config = HopeConfig(
    d_model=8, d_hidden=32, d_key=4, d_value=4, num_heads=2,
    titans_chunk_size=2, titans_hidden=16, cms_num_levels=2,
    cms_base_chunk_size=2, num_layers=1, vocab_size=32, max_seq_len=16,
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = Hope(config).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

ids = torch.randint(0, config.vocab_size, (1, 8), device=device)
labels = torch.randint(0, config.vocab_size, (1, 8), device=device)
out = model(ids, labels=labels)
opt.zero_grad(); out['loss'].backward(); opt.step()

print('Loss:', float(out['loss']))
print('Logits shape:', out['logits'].shape)
print('First token logits (5 dims):', out['logits'][0, 0, :5].detach().cpu())
PY"

echo "Done. Colab workspace: ${REMOTE_DIR}"
