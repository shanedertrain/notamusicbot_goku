"""Shared ASR helpers: the timeout error, local-cache lookup, normalization.

Local only: nothing here (or in the backends) downloads a model or calls a
speech API. Weights come from the Hugging Face cache and a missing file is a
clear error naming the one-time setup step.
"""

from __future__ import annotations

from notamusicbot.domain.listen import Transcript
from notamusicbot.listen.conversation import is_noise_transcript


class AsrTimeoutError(TimeoutError):
    """Transcription exceeded ``listen.asr.timeout_s``; the caller drops the turn."""


class AsrModelMissingError(RuntimeError):
    """Model weights are not in the local HF cache (never downloaded at runtime)."""


def missing_model_message(repo_id: str) -> str:
    return (
        f"ASR model {repo_id!r} is not in the local Hugging Face cache and the bot "
        f"never downloads at runtime. One-time setup: "
        f"`hf download {repo_id}` (or `.venv-main/bin/python -c "
        f"\"from huggingface_hub import snapshot_download; snapshot_download('{repo_id}')\"`)."
    )


def normalize_transcript(text: str, *, language: str, elapsed_s: float) -> Transcript:
    """Empty, one-word and known-hallucination text becomes an empty Transcript."""
    cleaned = text.strip()
    if is_noise_transcript(cleaned):
        cleaned = ""
    return Transcript(text=cleaned, language=language, elapsed_s=elapsed_s)
