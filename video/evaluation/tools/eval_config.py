# SPDX-License-Identifier: Apache-2.0
"""Load evaluation/config.yaml with ${VAR} / ${VAR:-default} expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

# Innermost first: the default may itself contain a ${...}, e.g.
# ${DFD_X_PYTHON:-${DFD_ROOT:-.}/venv_x/bin/python}. A default that admitted "}"
# matched the outer form up to the inner "}" and, once DFD_X_PYTHON was set,
# left "/venv_x/bin/python}" dangling after the value.
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^{}]*))?\}")


def _expand_str(value: str) -> str:
    while True:
        new = _VAR.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
        if new == value:
            return new
        value = new


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_str(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    import yaml

    path = Path(path).resolve()
    cfg = _expand(yaml.safe_load(path.read_text()))

    root = Path(cfg.get("repo_root") or ".")
    if not root.is_absolute():
        root = (path.parent.parent / root).resolve()
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(root)
    return cfg


def resolve(cfg: Dict[str, Any], value: str) -> Path:
    """Resolve a config path against the repo root."""
    p = Path(value)
    return p if p.is_absolute() else (Path(cfg["_repo_root"]) / p).resolve()


def parse_resolution(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def model_spec(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Resolve an arm name to its model spec, allowing per-run aliases.

    `dmd2_reseed_ckpt500` is not a configured model, but it starts with the
    configured `dmd2_reseed` + "_", so it inherits that spec. This is how one
    ReSeed checkpoint per run gets its own arm name (own video directory, own
    metadata records, own VBench raw dir) without a config entry per
    checkpoint and without two checkpoints ever sharing `dmd2_reseed/`.
    The longest configured prefix wins, so `dmd2_reseed_x` never falls back to
    `dmd2`. Raises KeyError for anything that matches nothing.
    """
    models = cfg["models"]
    if name in models:
        return {**models[name], "base_model": name}
    for base in sorted(models, key=len, reverse=True):
        if name.startswith(base + "_"):
            return {**models[base], "base_model": base}
    raise KeyError(name)
