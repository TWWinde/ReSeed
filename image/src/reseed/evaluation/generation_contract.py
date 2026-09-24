"""Immutable generation-request contracts for resumable image benchmarks.

The request is written before the first image.  A resumed process must present
the byte-equivalent scientific contract, which prevents valid PNG files from a
different checkpoint or sampler from being silently reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


REQUEST_NAME = "generation_request.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_generation_request(
    output: Path, request: Mapping[str, Any], *, resume: bool
) -> tuple[Path, str]:
    """Create or validate an immutable request before generation starts."""

    output.mkdir(parents=True, exist_ok=True)
    request_path = output / REQUEST_NAME
    generation_artifacts = [
        output / "images",
        output / "generation_manifest.jsonl",
        output / "generation_protocol.json",
        output / "scores.csv",
        output / "metrics.json",
    ]
    if request_path.is_file():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing != dict(request):
            raise ValueError(
                f"Refusing resume: generation request differs from {request_path}"
            )
        if not resume and any(path.exists() for path in generation_artifacts):
            raise FileExistsError(
                f"Refusing to reuse populated output without --resume: {output}"
            )
    else:
        if any(path.exists() for path in generation_artifacts):
            raise ValueError(
                "Refusing provenance-free output containing generation artifacts: "
                f"{output}. Use a fresh directory."
            )
        atomic_json(request_path, request)
    return request_path, sha256_file(request_path)

