#!/usr/bin/env python3
"""Generate paired SD3.5 Medium base/student images under an isolated contract."""

from __future__ import annotations

from reseed.evaluation.sd3m_generate import (
    SD35M_EVALUATION_CONTRACT,
    run_sd3_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    return run_sd3_evaluation(SD35M_EVALUATION_CONTRACT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
