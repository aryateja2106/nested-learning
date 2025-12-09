# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/` with core logic under `src/core/`, models in `src/models/`, experiments in `src/experiments/`, and utilities in `src/utils/`.
- Interactive demo is in `demo/app.py`.
- Tests live in `tests/` (mainly `tests/test_components.py`).
- Documentation and algorithm notes are in `docs/` and `README.md`.

## Build, Test, and Development Commands
- Install dependencies (prefer a virtualenv):  
  ```bash
  python -m pip install -r requirements.txt
  ```
- Run the demo locally (CPU is fine; GPU optional):  
  ```bash
  python demo/app.py
  ```
- Execute the test suite:  
  ```bash
  pytest tests/test_components.py
  ```
- (Optional) Check types and formatting if installed:  
  ```bash
  mypy src
  black --check .
  isort --check-only .
  ```

## Coding Style & Naming Conventions
- Python code should be formatted with Black and import-sorted with isort; adhere to type hints and keep functions small and readable.
- Use descriptive, lowercase_with_underscores for modules/functions, CamelCase for classes, and UPPER_CASE for constants.
- Keep docstrings concise and explain non-obvious logic; prefer English comments sparingly.

## Testing Guidelines
- Primary framework: `pytest` (see `tests/test_components.py` for patterns).
- Name tests with `test_...` functions inside `Test...` classes when grouping related behaviors.
- Aim to cover new logic with focused unit tests; run `pytest` before pushing.

## Commit & Pull Request Guidelines
- Write clear, imperative commit messages (e.g., “Add CMS reset helper”, “Fix DGD decay math”).
- PRs should include: brief summary, scope of changes, testing evidence (`pytest` output or relevant logs), and any caveats (e.g., known NaNs with `M3Optimizer` at aggressive lrs).
- Link issues when applicable and keep PRs small and reviewable.

## Security & Configuration Tips
- Keep secrets out of the repo; do not commit tokens or Colab credentials.
- When using `cgpu`/Colab, prefer uploading minimal artifacts and avoid storing large checkpoints in the repo.
- Pin runtime with `requirements.txt`; Python 3.9+ is expected.
