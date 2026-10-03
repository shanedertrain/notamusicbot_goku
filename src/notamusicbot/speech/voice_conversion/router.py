"""Voice-converter router: selects the backend from the profile.

The pipeline stays generic: it always talks to a :class:`VoiceConverter`.
The router inspects ``VoiceConversionProfile.backend`` per request and
dispatches to the matching implementation:

- ``passthrough`` -> :class:`PassthroughVoiceConverter` (no-op, always available)
- ``rvc``         -> one persistent :class:`RvcVoiceConverter` owned by the router

Lifecycle: :meth:`startup` spawns and health-checks the RVC worker once
(passthrough needs nothing); :meth:`shutdown` tears it down. Call these from
the application lifecycle boundary, not per utterance.

Failure policy: if the RVC worker cannot be started (no GPU, missing worker
runtime, ...), ``startup`` logs the failure and marks RVC unavailable instead
of crashing the bot. What happens afterwards is decided by
``rvc_unavailable_policy``:

- ``"error"``       -> requests asking for the ``rvc`` backend raise
  :class:`RvcUnavailableError` (commentary is then skipped by its caller;
  media playback continues). This is the default: the bot must never silently
  output the un-converted source voice.
- ``"passthrough"`` -> requests degrade to the passthrough converter with a
  warning (legacy permissive behavior).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.speech.voice_conversion.base import VoiceConverter
from notamusicbot.speech.voice_conversion.passthrough import PassthroughVoiceConverter

logger = logging.getLogger(__name__)

PASSTHROUGH_BACKEND = "passthrough"
RVC_BACKEND = "rvc"
KNOWN_BACKENDS = (PASSTHROUGH_BACKEND, RVC_BACKEND)


class UnknownBackendError(ValueError):
    """The profile requests a voice-conversion backend the router cannot serve."""


class RvcUnavailableError(RuntimeError):
    """The RVC backend is requested but its worker is not available."""


class RouterClosedError(RuntimeError):
    """The router was shut down; conversion is refused, not lazily restarted.

    Review Round-6 §3/B: after :meth:`shutdown`, a stray late ``convert``
    (e.g. from a detached pipeline task that escaped the coordinator drain)
    must NOT lazily restart the RVC worker. A deliberate new :meth:`startup`
    cycle re-enables conversion.
    """


class VoiceConverterRouter:
    """Dispatch ``convert`` by profile backend; owns one persistent RVC adapter.

    ``rvc_factory`` is injectable so tests can install a spy/fake adapter at
    the composition root; production uses :class:`RvcVoiceConverter`.
    """

    def __init__(
        self,
        *,
        enabled_backends: list[str] | tuple[str, ...] = KNOWN_BACKENDS,
        rvc_factory: Callable[[], VoiceConverter] | None = None,
        rvc_unavailable_policy: str = "error",
    ) -> None:
        unknown = [b for b in enabled_backends if b not in KNOWN_BACKENDS]
        if unknown:
            msg = f"Unknown voice-conversion backend(s): {unknown} (known: {KNOWN_BACKENDS})"
            raise UnknownBackendError(msg)
        if rvc_unavailable_policy not in ("error", "passthrough"):
            msg = (
                f"Unknown rvc_unavailable_policy {rvc_unavailable_policy!r}; "
                "expected 'error' or 'passthrough'"
            )
            raise ValueError(msg)
        self._enabled_backends = frozenset(enabled_backends)
        self._rvc_unavailable_policy = rvc_unavailable_policy
        self._rvc_factory = rvc_factory or self._default_rvc_factory
        self._rvc: VoiceConverter | None = None
        self._rvc_available = False
        self._rvc_start_attempted = False
        # Review Round-6 §3/B: explicit closed/shutting-down state. Once
        # ``shutdown`` ran, ``convert`` raises RouterClosedError instead of
        # lazily restarting RVC; a deliberate ``startup`` cycle clears it.
        self._closed = False
        self._passthrough = PassthroughVoiceConverter()

    @staticmethod
    def _default_rvc_factory() -> VoiceConverter:
        from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

        return RvcVoiceConverter()

    # ------------------------------------------------------------------ state

    @property
    def rvc_available(self) -> bool:
        """True once the RVC worker has been started and health-checked."""
        return self._rvc_available

    @property
    def enabled_backends(self) -> frozenset[str]:
        return self._enabled_backends

    # -------------------------------------------------------------- lifecycle

    async def startup(self) -> None:
        """Start and health-check the persistent RVC worker (if enabled).

        A failure is logged and remembered (RVC marked unavailable) so a
        missing GPU or worker runtime degrades per policy instead of killing
        application startup.

        Review Round-6 §3/B: this is the deliberate new startup cycle — it
        clears the closed state from a previous ``shutdown`` so conversion is
        served again.
        """
        if RVC_BACKEND not in self._enabled_backends:
            return
        self._closed = False
        self._rvc_start_attempted = False
        await self._ensure_rvc_started()
        logger.info("RVC voice converter startup complete (available=%s)", self._rvc_available)

    async def _ensure_rvc_started(self) -> None:
        """Start the RVC worker once; a failure marks it unavailable (logged)."""
        if self._rvc_start_attempted:
            return
        self._rvc_start_attempted = True
        if self._rvc is None:
            self._rvc = self._rvc_factory()
        try:
            start = getattr(self._rvc, "start", None)
            if start is not None:
                await start()
        except Exception:
            self._rvc_available = False
            logger.exception("RVC voice converter failed to start; marking it unavailable")
            return
        self._rvc_available = True

    async def shutdown(self) -> None:
        """Shut down the RVC worker if one was created (passthrough: no-op).

        Review Round-6 §3/B: the closed state is entered BEFORE the worker is
        torn down, so a ``convert`` arriving mid-shutdown already sees the
        closed router. The closed state stays latched: late converts raise
        :class:`RouterClosedError` instead of lazily restarting RVC.
        """
        self._closed = True
        rvc = self._rvc
        if rvc is None:
            return
        shutdown = getattr(rvc, "shutdown", None)
        if shutdown is not None:
            await shutdown()
        self._rvc_available = False

    # ----------------------------------------------------------------- convert

    async def convert(
        self,
        source: AudioArtifact,
        profile: VoiceConversionProfile,
        output_path: Path,
    ) -> AudioArtifact:
        # Review Round-6 §3/B: after shutdown, refuse instead of lazily
        # restarting the RVC worker behind the caller's back.
        if self._closed:
            msg = (
                "voice-converter router is shut down; conversion is refused "
                "until a deliberate new startup cycle"
            )
            raise RouterClosedError(msg)
        backend = profile.backend
        if backend not in KNOWN_BACKENDS:
            msg = (
                f"Unknown voice-conversion backend {backend!r} in voice profile "
                f"(known backends: {', '.join(KNOWN_BACKENDS)})"
            )
            raise UnknownBackendError(msg)
        if backend not in self._enabled_backends:
            msg = (
                f"Voice-conversion backend {backend!r} is not enabled in the "
                "application configuration "
                f"(enabled: {', '.join(sorted(self._enabled_backends))})"
            )
            raise UnknownBackendError(msg)
        if backend == PASSTHROUGH_BACKEND:
            return await self._passthrough.convert(source, profile, output_path)
        return await self._convert_via_rvc(source, profile, output_path)

    async def _convert_via_rvc(
        self,
        source: AudioArtifact,
        profile: VoiceConversionProfile,
        output_path: Path,
    ) -> AudioArtifact:
        if not self._rvc_available:
            # First use before any startup(): create and start lazily so the
            # adapter's own convert() path can spawn the worker.
            await self._ensure_rvc_started()
        if not self._rvc_available:
            if self._rvc_unavailable_policy == "passthrough":
                logger.warning(
                    "RVC unavailable; falling back to passthrough voice conversion "
                    "(output keeps the source voice)"
                )
                return await self._passthrough.convert(source, profile, output_path)
            msg = (
                "RVC voice conversion is unavailable (worker failed to start or "
                "was not started); rvc_unavailable_policy='error' refuses to emit "
                "un-converted audio"
            )
            raise RvcUnavailableError(msg)
        assert self._rvc is not None
        return await self._rvc.convert(source, profile, output_path)
