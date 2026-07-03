"""Per-run provenance manifest: pin what produced a result so it is reproducible."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def build_manifest(cfg) -> dict:
    cfg_json = json.dumps(cfg.to_dict(), sort_keys=True)
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
    }
