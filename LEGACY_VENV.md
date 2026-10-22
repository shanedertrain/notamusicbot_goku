# Legacy RVC virtual environment note

The `.venv/` at the repository root is the **legacy RVC environment**, now
built with **Python 3.12** by `scripts/setup.sh`.

- **fairseq is NOT installed and is no longer required.** fairseq 0.12.2 does
  not work on Python >= 3.11 (it fails to build/import there), so the legacy
  stack was made fairseq-free instead of pinning an old interpreter.
- The HuBERT feature extractor is loaded by `rvc_cli/hubert_loader.py`, a
  pure-PyTorch reimplementation of fairseq's `HubertModel.extract_features`
  (same layer outputs, same `final_proj` behaviour). It exposes
  `load_hubert(path, device, is_half) -> torch.nn.Module`, which moves the
  model to `device`, casts to half/float and returns it in eval mode.
- The model file it loads is `hubert_base.state_dict.pt` at the repository
  root: a **plain tensor state dict** loaded with
  `torch.load(..., weights_only=True)`.

## Producing `hubert_base.state_dict.pt` (one time only)

The state dict is produced **ONCE** from the original `hubert_base.pt` using
the old Python 3.10 fairseq venv (the only place fairseq is still needed):

```bash
python -c "import torch; from fairseq import checkpoint_utils; m,_,_ = checkpoint_utils.load_model_ensemble_and_task(['hubert_base.pt'], suffix=''); torch.save(m[0].state_dict(), 'hubert_base.state_dict.pt')"
```

After that, `hubert_base.pt` and fairseq are not needed at runtime.

## Running the worker

The legacy RVC worker (`legacy/rvc_worker.py` / `rvc_cli`) runs from this
`.venv/`; the main application environment is `.venv-main/` (Python 3.12,
created with `uv venv --python 3.12 .venv-main` and installed from
`pyproject.toml`). Do not install legacy RVC dependencies (old torch pins,
`pyworld`, `faiss`, `torchcrepe`, etc.) into `.venv-main`.

Original `pyproject.toml` and `poetry.lock` were moved to `backups/` before
the modern `pyproject.toml` replaced them.
