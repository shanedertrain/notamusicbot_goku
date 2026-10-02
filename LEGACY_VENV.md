# Legacy RVC virtual environment note

The existing `.venv/` at the repository root is the **legacy Python 3.10
environment** used by the old RVC/fairseq stack. It is left untouched on
purpose:

- Phase D (RVC adapter) will decide whether legacy RVC runs in-process under
  Python 3.12 or behind an isolated subprocess/HTTP service using this venv.
- The new main application environment is `.venv-main/` (Python 3.12), created
  with `uv venv --python 3.12 .venv-main` and installed from
  `pyproject.toml` dependencies (torch, qwen-tts, soundfile, pydantic*, yaml,
  dev tools).
- Do not install legacy RVC dependencies (`fairseq`, old torch pins, etc.)
  into `.venv-main`.

Original `pyproject.toml` and `poetry.lock` were moved to `backups/` before
the modern `pyproject.toml` replaced them.
