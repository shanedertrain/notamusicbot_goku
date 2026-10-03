"""Shared helper for Round-4 review evidence files (evidence/*.json).

Each real-GPU integration test writes a machine-readable record containing at
least: timestamp, commit, GPU name, driver, torch version, its measurements,
and pass/fail. The packaging script already ships ``evidence/*.json``.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "evidence"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def gpu_info() -> dict[str, str]:
    info = {"gpu": "unknown", "driver": "unknown"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        name, driver = (part.strip() for part in out.splitlines()[0].split(",", 1))
        info = {"gpu": name, "driver": driver}
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
    except Exception:
        info["torch"] = "unavailable"
        info["cuda_available"] = "False"
    return info


def write_evidence(filename: str, payload: dict[str, object]) -> Path:
    """Write ``evidence/<filename>`` with the common envelope fields added."""
    record: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        **gpu_info(),
        **payload,
    }
    EVIDENCE_DIR.mkdir(exist_ok=True)
    path = EVIDENCE_DIR / filename
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[evidence] wrote {path}")
    return path
