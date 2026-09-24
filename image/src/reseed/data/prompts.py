"""Prompt-only training data reader for JSON, JSONL, and plain-text files."""

from __future__ import annotations

import json
from pathlib import Path

#: Suffixes parsed as structured records carrying a prompt ``column``.
STRUCTURED_SUFFIXES = frozenset({".json", ".jsonl"})
#: Suffixes whose every non-empty line is one prompt, verbatim.
RAW_LINE_SUFFIXES = frozenset({".txt", ".text", ".prompts", ""})


def _prompt_from_record(record: object, column: str) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict) and isinstance(record.get(column), str):
        return record[column]
    raise ValueError(f"Expected a string or an object with a string {column!r} field")


def validate_prompt_file(path: str | Path, *, allow_raw_lines: bool = False) -> Path:
    """Reject prompt files this reader cannot parse, before anything is read.

    A tabular file (``.csv``/``.tsv``) has no parser here: it falls through to the
    raw-line branch, where every whole row -- header included -- becomes a training
    prompt.  That silently happened to the SDXL residual-bias run, so an unknown
    suffix is now an error unless the caller opts in with ``allow_raw_lines``.
    """

    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    if suffix in STRUCTURED_SUFFIXES or suffix in RAW_LINE_SUFFIXES or allow_raw_lines:
        return source
    raise ValueError(
        f"unsupported prompt-file suffix {suffix!r} for {source}: read_prompts parses "
        f"{sorted(STRUCTURED_SUFFIXES)} as records and {sorted(RAW_LINE_SUFFIXES - {''})} "
        "as one prompt per line. A tabular file would be read as whole raw lines including "
        "its header; pass allow_raw_lines=True (trainers: --allow-raw-line-prompts) only if "
        "each raw line really is the prompt you want to train on."
    )


def read_prompts(
    path: str | Path,
    *,
    column: str = "prompt",
    limit: int | None = None,
    allow_raw_lines: bool = False,
) -> list[str]:
    source = validate_prompt_file(path, allow_raw_lines=allow_raw_lines)
    text = source.read_text(encoding="utf-8")
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON prompt file must contain a list")
        prompts = [_prompt_from_record(item, column) for item in payload]
    elif suffix == ".jsonl":
        prompts = [_prompt_from_record(json.loads(line), column) for line in text.splitlines() if line.strip()]
    else:
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {source}")
    return prompts if limit is None else prompts[:limit]


__all__ = [
    "RAW_LINE_SUFFIXES",
    "STRUCTURED_SUFFIXES",
    "read_prompts",
    "validate_prompt_file",
]
