# Test Results — notamusicbot rework

Semantics per REWORK_HANDOFF.md section 32: every result is PASS / FAIL /
SKIPPED / NOT_RUN. Nothing unavailable was converted into a pass.

| Check | Command | Result | Count |
|---|---|---|---|
| Unit tests | `.venv-main/bin/python -m pytest tests/unit` | **PASS** | 105 passed, 0 failed, 0 skipped (1 benign warning) |
| RVC integration | `.venv-main/bin/python -m pytest tests/integration -m "rvc and integration"` | **PASS** | 2 passed, 2 deselected |
| Lint | `.venv-main/bin/ruff check src tests` | **PASS** | All checks passed |
| Format | `.venv-main/bin/ruff format --check src tests` | **PASS** | 61 files already formatted |
| Import smoke | `.venv-main/bin/python -c "import notamusicbot, notamusicbot.app"` | **PASS** | — |
| Setup script syntax | `bash -n scripts/setup.sh` | **PASS** | no syntax errors |
| Qwen TTS smoke (Gate C, coordinator-verified) | GPU generation of `var/speech/smoke/hey_it_me_goku.wav` | **PASS** | 3.92 s audio, 24 kHz returned rate, bf16/sdpa/cuda, peak −4.3 dBFS |
| End-to-end local speech (Gate D, coordinator-verified) | Qwen → RVC (Goku) → postprocess | **PASS** | `var/speech/e2e_goku.wav` 4.14 s; tts 7.9 s, vc 7.98 s warm |
| Mocked Discord startup smoke (coordinator-verified) | `tools/smoke_discord_startup.py` (mocked gateway) | **PASS** | — |
| **Live Discord smoke** | real gateway connection | **PASS** | Bot logged in as `Goku#8715` on the real gateway, connected to 2 guilds, `!g` commands registered, clean shutdown (coordinator, 2026-10-02). Live voice-channel playback needs a human issuing `!gplay` in a voice channel — left for the user. |
| Fresh install (Gate G, coordinator-run) | fresh uv venv from `pyproject.toml` | **PASS** | Fresh py3.12 venv: `.[discord,media,dev]` installed after fixing an unsatisfiable `yt-dlp-ejs>=2025.10.14` pin (→ `>=0.8`); import OK; **105 unit tests passed** in the fresh venv. `scripts/setup.sh` additionally exercised end-to-end (idempotent run) on the development machine with all steps OK. |

## Notes

- The RVC integration tests spawn `legacy/rvc_worker.py` with the legacy
  Python 3.10 venv and the real GOKU model; both `test_worker_health_and_convert_real_model`
  and `test_worker_rejects_bad_requests` pass on this machine.
- Phase G temporarily broke the RVC integration by moving the root legacy
  modules `tts.py` / `configuration.py` into `legacy_flat/`; they were restored
  to the repo root (they are part of the rvc_cli import surface) and the tests
  were re-run green afterwards. The unit-test and lint results above are from
  after that fix.
- Benchmarks are reported separately in `BENCHMARK_RESULTS.md`.
