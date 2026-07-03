"""Per-run provenance manifest: pin what produced a result so it is reproducible.

Beyond the config and package versions, the manifest records best-effort "resolved" facts
about the actual run environment: the pilot's git commit, the resolved model Hub revision,
the GPU / CUDA / torch versions, and a hash of the pinned lockfile. Every resolved probe is
best-effort: when a dependency or resource is absent (mock runs, the light test venv,
offline) the field is null rather than an error.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def _git_info() -> dict | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return None


def _hardware_info() -> dict:
    info: dict = {"gpu": None, "cuda": None, "torch": None}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def _model_commit_sha(name: str, revision: str) -> str | None:
    """Resolve the Hub commit SHA for (name, revision). None for mock / offline / error."""
    if not name or name == "mock":
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(name, revision=revision or None).sha
    except Exception:
        return None


def _lockfile_sha(path: str = "requirements-lock.txt") -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def build_manifest(cfg) -> dict:
    cfg_json = json.dumps(cfg.to_dict(), sort_keys=True)
    model = cfg.to_dict().get("model", {})
    hw = _hardware_info()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "config_sha256": hashlib.sha256(cfg_json.encode()).hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            p: _pkg_version(p)
            for p in ("numpy", "scipy", "datasets", "transformers", "torch")
        },
        "resolved": {
            "git": _git_info(),
            "model_name": model.get("name"),
            "model_revision_config": model.get("revision") or None,
            "model_commit_sha": _model_commit_sha(
                model.get("name", ""), model.get("revision", "")
            ),
            "gpu": hw["gpu"],
            "cuda": hw["cuda"],
            "torch": hw["torch"],
            "lockfile_sha256": _lockfile_sha(),
        },
    }
