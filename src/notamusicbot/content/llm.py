"""Local Qwen LLM content provider (fully offline, no API key).

Runs ``Qwen/Qwen3-4B-Instruct-2507`` through transformers on the local GPU.
The model is loaded lazily (never at import time, per handoff section 19) and
generation runs in ``asyncio.to_thread`` behind the SHARED GPU inference
coordinator (:mod:`notamusicbot.speech.coordinator`): at most one main-process
GPU job runs at a time, and the event loop is never blocked. Review Round-3
P1: the provider no longer owns a private semaphore — LLM generation and
speech-pipeline inference (TTS/RVC) are serialized against each other on the
same GPU through one shared coordinator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from notamusicbot.content import prompts
from notamusicbot.content.base import ContentRequest
from notamusicbot.speech.coordinator import InferenceCoordinator, shared_gpu_coordinator

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(slots=True)
class LlmLoadInfo:
    """Diagnostics recorded when the model is loaded."""

    model_id: str
    device: str
    load_s: float
    vram_used_mib: int
    vram_total_mib: int


def _vram_mib(device: str) -> tuple[int, int]:
    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(_device_index(device))
            return int((total_b - free_b) // (1024 * 1024)), int(total_b // (1024 * 1024))
    except Exception:  # pragma: no cover - diagnostics only
        pass
    return 0, 0


def _device_index(device: str):
    if device.startswith("cuda"):
        parts = device.split(":")
        return int(parts[1]) if len(parts) > 1 else None
    return None


class LocalQwenLlmProvider:
    """ContentProvider backed by a local Qwen3 instruct model."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        system_prompt: str = prompts.RADIO_SYSTEM_PROMPT,
        gpu_coordinator: InferenceCoordinator | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._attn_implementation = attn_implementation
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._gpu = gpu_coordinator or shared_gpu_coordinator()
        self._model = None
        self._tokenizer = None
        self.load_info: LlmLoadInfo | None = None

    @classmethod
    def from_settings(
        cls, settings, gpu_coordinator: InferenceCoordinator | None = None
    ) -> LocalQwenLlmProvider:
        return cls(
            model_id=settings.model_id,
            device=settings.device,
            dtype=settings.dtype,
            attn_implementation=settings.attn_implementation,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            system_prompt=settings.system_prompt or prompts.RADIO_SYSTEM_PROMPT,
            gpu_coordinator=gpu_coordinator,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> LlmLoadInfo:
        """Load model + tokenizer. Call off the event loop during startup."""
        if self._model is not None:
            assert self.load_info is not None
            return self.load_info
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        before_used, _before_total = _vram_mib(self._device)
        dtype = getattr(torch, self._dtype)
        start = time.perf_counter()
        logger.info("Loading local LLM %s on %s (%s)...", self._model_id, self._device, self._dtype)
        tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            dtype=dtype,
            attn_implementation=self._attn_implementation,
        )
        model.to(self._device)
        model.eval()
        load_s = time.perf_counter() - start
        used, total = _vram_mib(self._device)
        self._tokenizer = tokenizer
        self._model = model
        self.load_info = LlmLoadInfo(
            model_id=self._model_id,
            device=self._device,
            load_s=load_s,
            vram_used_mib=used,
            vram_total_mib=total,
        )
        logger.info(
            "LLM %s loaded in %.1fs; VRAM %d/%d MiB (delta %d MiB)",
            self._model_id,
            load_s,
            used,
            total,
            used - before_used,
        )
        return self.load_info

    async def generate(self, request: ContentRequest) -> str:
        """Generate commentary text; heavy work happens off the event loop.

        Review Round-4 P0: the blocking inference runs through
        ``coordinator.run_blocking`` instead of ``asyncio.to_thread`` inside
        ``slot()``. If this coroutine is cancelled (e.g. the player's stop
        timeout), the GPU lease stays held by the coordinator's detached
        holder until ``_generate_sync`` really finishes — a follow-up GPU job
        can never enter the coordinator while the cancelled job's thread is
        still generating.
        """
        prompt = prompts.build_prompt(request)
        return await self._gpu.run_blocking(self._generate_sync, prompt)

    def _generate_sync(self, prompt: str) -> str:
        import torch

        if self._model is None or self._tokenizer is None:
            self.load()
        assert self._model is not None and self._tokenizer is not None
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        start = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._temperature > 0,
                temperature=max(self._temperature, 1e-5),
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        completion = output[0][inputs["input_ids"].shape[1] :]
        reply = self._tokenizer.decode(completion, skip_special_tokens=True).strip()
        new_tokens = len(completion)
        logger.info(
            "LLM generation done in %.2fs (%d new tokens)", time.perf_counter() - start, new_tokens
        )
        if not reply:
            raise ValueError("Local LLM returned empty commentary")
        return reply


def describe_vram() -> str:
    """Human-readable VRAM snapshot for smoke tests / startup logs."""
    used, total = _vram_mib("cuda")
    return f"VRAM {used}/{total} MiB" if total else "VRAM unavailable (no CUDA)"
