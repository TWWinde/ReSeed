"""Canonical and legacy checkpoint discovery/import.

Inference only needs a student directory.  Resuming an optimizer/RNG state is
stricter and remains implementation-specific; this module deliberately exposes
legacy checkpoints as *weight-only* canonical views.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


STUDENT_NAMES = ("student", "unet")
FAKE_SCORE_NAMES = ("fake_score", "unet_fake")


@dataclass(frozen=True)
class CheckpointLayout:
    root: Path
    student: Path
    fake_score: Path | None
    student_layout_name: str
    fake_score_layout_name: str | None

    @property
    def resumable(self) -> bool:
        return self.fake_score is not None


def _model_dir(root: Path, names: tuple[str, ...]) -> tuple[Path | None, str | None]:
    for name in names:
        candidate = root / name
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return candidate, name
    if (root / "config.json").is_file():
        return root, "root"
    return None, None


def inspect_checkpoint(path: str | Path) -> CheckpointLayout:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    student, student_name = _model_dir(root, STUDENT_NAMES)
    if student is None or student_name is None:
        expected = ", ".join(STUDENT_NAMES)
        raise FileNotFoundError(f"No student model under {root}; expected one of: {expected}")
    fake_score, fake_name = _model_dir(root, FAKE_SCORE_NAMES)
    return CheckpointLayout(root, student, fake_score, student_name, fake_name)


def resolve_student_dir(path: str | Path) -> Path:
    return inspect_checkpoint(path).student


def resolve_fake_score_dir(path: str | Path, *, required: bool = False) -> Path | None:
    fake_score = inspect_checkpoint(path).fake_score
    if required and fake_score is None:
        raise FileNotFoundError(f"No fake-score model found under {Path(path).expanduser()}")
    return fake_score


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path, mode: Literal["link", "copy"]) -> None:
    if mode == "link":
        os.symlink(source, destination, target_is_directory=True)
    else:
        shutil.copytree(source, destination)


def import_legacy_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    mode: Literal["link", "copy"] = "link",
    method: str | None = None,
) -> CheckpointLayout:
    """Create a canonical view from a legacy checkpoint without editing it."""
    if mode not in {"link", "copy"}:
        raise ValueError("mode must be 'link' or 'copy'")
    layout = inspect_checkpoint(source)
    output = Path(destination).expanduser().resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"Destination must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _link_or_copy(layout.student, output / "student", mode)
    if layout.fake_score is not None:
        _link_or_copy(layout.fake_score, output / "fake_score", mode)

    student_weight = layout.student / "diffusion_pytorch_model.safetensors"
    manifest = {
        "schema_version": 1,
        "kind": "legacy_weight_only_import",
        "method": method,
        "source_checkpoint": str(layout.root),
        "source_layout": {
            "student": layout.student_layout_name,
            "fake_score": layout.fake_score_layout_name,
        },
        "import_mode": mode,
        "student_weight_sha256": sha256_file(student_weight),
        "resume_supported": False,
        "resume_reason": "Only model directories are imported; optimizer, scaler, and RNG state are not translated.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return inspect_checkpoint(output)


def layout_as_json(layout: CheckpointLayout) -> str:
    payload = asdict(layout)
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
    payload["resumable"] = layout.resumable
    return json.dumps(payload, indent=2, sort_keys=True)
