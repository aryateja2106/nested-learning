# Packaging Checklist

Complete checklist for shipping a production-ready paper implementation.

## Pre-Commit Checklist

### Day 1 Essentials (Before ANY Commits)

```markdown
- [ ] .gitignore created with comprehensive patterns
- [ ] No secrets/credentials in any file
- [ ] README.md has quickstart section
- [ ] License file present (MIT, Apache 2.0, etc.)
```

### .gitignore Template

```gitignore
# === Python ===
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# === Virtual Environments ===
.venv/
venv/
ENV/
env/

# === UV ===
.uv/

# === Testing ===
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# === Type Checking ===
.mypy_cache/
.pytype/

# === Linting ===
.ruff_cache/

# === IDEs ===
.vscode/
.idea/
*.swp
*.swo
*~

# === OS ===
.DS_Store
Thumbs.db

# === Project Specific ===
# Logs
logs/
*.log

# Outputs
outputs/
results/
checkpoints/
*.pt
*.pth
*.ckpt

# Weights & Biases
wandb/

# === SECRETS (CRITICAL) ===
.env
.env.*
*.env
secrets/
credentials/
*secret*
*credential*
*token*
*key*.json
!package*.json

# Config files that might contain keys
config.local.yaml
config.local.json
```

## File Structure Checklist

### Required Files

```markdown
- [ ] pyproject.toml (UV project config)
- [ ] uv.lock (commit this!)
- [ ] .python-version (Python version pin)
- [ ] .gitignore
- [ ] README.md
- [ ] requirements.txt (for pip users)
- [ ] LICENSE
```

### Source Code Structure

```markdown
- [ ] src/__init__.py exports public API
- [ ] src/core/__init__.py exports core components
- [ ] src/models/__init__.py exports models
- [ ] All modules have docstrings
- [ ] Type hints on public functions
```

### Tests

```markdown
- [ ] tests/test_components.py exists
- [ ] All tests pass: `uv run pytest -v`
- [ ] Tests are lightweight (no OOM risk)
- [ ] Tests cover core functionality
- [ ] No hardcoded paths or devices
```

### Documentation

```markdown
- [ ] README leads with quickstart
- [ ] Installation instructions (UV + pip)
- [ ] Usage examples with copy-paste commands
- [ ] Config preset explanations
- [ ] Hardware requirements noted
- [ ] Paper citation included
- [ ] Architecture diagram/overview
```

### Demo

```markdown
- [ ] demo/app.py runs without errors
- [ ] Gradio interface is intuitive
- [ ] Reasonable defaults selected
- [ ] Error handling for edge cases
```

### Notebooks

```markdown
- [ ] notebooks/quickstart.ipynb runs end-to-end
- [ ] Notebook is self-contained
- [ ] Clear markdown explanations
- [ ] Works on CPU (GPU optional)
```

## Code Quality Checklist

### Style & Formatting

```bash
# Run these before committing
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

### Docstrings

```python
# Every public class/function needs:
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    Brief description.
    
    Implements Equation X from [Paper Name].
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Example:
        >>> result = function_name(x, y)
    """
```

### Type Hints

```python
# Required on public API
from typing import Optional, Tuple, Dict, List, Union

def create_model(
    d_model: int = 512,
    num_layers: int = 6,
    config: Optional[ModelConfig] = None,
) -> Model:
    ...
```

## README Template

```markdown
# Project Name

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](PAPER_URL)

Brief description. Links to paper and blog.

## Quick Start

\`\`\`bash
# Clone
git clone https://github.com/user/repo.git
cd repo

# Setup (UV recommended)
uv venv && source .venv/bin/activate
uv sync

# Or pip
pip install -r requirements.txt

# Verify installation
uv run pytest tests/

# Run demo
uv run python demo/app.py

# Train
uv run python train.py --config small --steps 500
\`\`\`

## What is [Concept]?

2-3 sentence explanation with optional diagram.

## Architecture

\`\`\`
┌─────────────────────────────┐
│     Architecture Diagram     │
└─────────────────────────────┘
\`\`\`

## Usage

### Basic Usage

\`\`\`python
from src import create_model

model = create_model(d_model=512)
output = model(input_tensor)
\`\`\`

### Training

\`\`\`bash
# Small (quick test)
uv run python train.py --config small --steps 500

# Medium
uv run python train.py --config medium --steps 2000

# Large (GPU recommended)
uv run python train.py --config large --steps 10000
\`\`\`

### Colab/Cloud GPU

\`\`\`bash
# If using cgpu for Colab
bash run_cgpu_uv.sh
\`\`\`

## Components

| Component | Description | Reference |
|-----------|-------------|-----------|
| Name | Brief | Equation X |

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## Citation

\`\`\`bibtex
@inproceedings{author2025paper,
  title={Paper Title},
  author={Author Names},
  booktitle={Venue},
  year={2025}
}
\`\`\`

## License

Apache 2.0
```

## UV Setup Commands

### Initialize Project

```bash
# Create new project
uv init project-name
cd project-name

# Pin Python
uv python pin 3.11

# Add dependencies
uv add torch numpy scipy tqdm rich pyyaml

# Add dev dependencies
uv add --dev pytest pytest-cov ruff mypy

# Add optional groups
uv add --group demo gradio
uv add --group notebook jupyter ipywidgets
```

### Export for pip Users

```bash
# Basic requirements
uv export --format requirements-txt > requirements.txt

# With all extras
uv export --format requirements-txt --all-extras > requirements-all.txt
```

### CI/CD Setup

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup UV
        uses: astral-sh/setup-uv@v6
        with:
          version: "0.9.5"
          enable-cache: true
      
      - name: Install Python
        run: uv python install
      
      - name: Install dependencies
        run: uv sync --frozen
      
      - name: Run tests
        run: uv run pytest -v
      
      - name: Run linting
        run: uv run ruff check .
```

## Security Checklist

### Before Every Commit

```bash
# Search for potential secrets
grep -r "sk-\|api_key\|secret\|password\|token" . --include="*.py" --include="*.yaml" --include="*.json"

# Check for hardcoded URLs with credentials
grep -r "https://.*:.*@" .

# Check for private keys
grep -r "PRIVATE KEY" .
```

### Common Mistakes

1. **API keys in config files**: Use environment variables
2. **Credentials in notebooks**: Clear outputs before commit
3. **Hardcoded paths**: Use relative paths or config
4. **Debug prints with sensitive data**: Remove before commit

### Security Note Template

If credentials were accidentally committed:

```markdown
## Security Note

Past commits contained credentials that have been removed. 
If you forked before [date], please:
1. Rotate any exposed keys
2. Pull latest changes
3. Check your fork for leaked secrets
```

## Final Release Checklist

```markdown
## v1.0.0 Release Checklist

### Code Quality
- [ ] All tests pass
- [ ] No linting errors
- [ ] Type hints complete
- [ ] Docstrings complete

### Documentation
- [ ] README accurate and complete
- [ ] Quickstart works end-to-end
- [ ] All links work
- [ ] Paper citation correct

### Files
- [ ] No secrets in any file
- [ ] .gitignore comprehensive
- [ ] requirements.txt generated
- [ ] License file present

### Functionality
- [ ] Demo runs without errors
- [ ] Training script works (all configs)
- [ ] Notebook executes cleanly
- [ ] CPU fallback works

### Optional
- [ ] CI/CD pipeline set up
- [ ] Pre-commit hooks configured
- [ ] CHANGELOG.md created
- [ ] Release notes written
```

## Post-Release

### Monitoring

- Watch GitHub issues for bugs
- Check if users can run quickstart
- Monitor for security reports

### Maintenance

- Keep dependencies updated
- Respond to issues promptly
- Tag releases properly

### Community

- Add CONTRIBUTING.md if accepting PRs
- Respond to questions
- Credit contributors
