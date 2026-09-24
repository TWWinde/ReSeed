"""Project-local path helpers.

Paths resolve relative to this package's checkout so the evaluation and
training scripts never depend on absolute cluster locations. Set
``RESEED_ROOT`` to override the root (used by tools and tests that run from a
different working directory).
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the checkout root, optionally overridden via ``RESEED_ROOT``."""
    override = os.environ.get("RESEED_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # <root>/image/src/reseed/paths.py -> <root>/image
    return Path(__file__).resolve().parents[2]


def config_root() -> Path:
    return project_root() / "configs"
