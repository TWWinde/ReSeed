"""Auxiliary objective used in ReSeed diagnostics."""

from .feature_diversity import (
    FrozenDINOStateEncoder,
    capture_parameter_gradients,
    merge_quality_and_diversity_gradients,
    teacher_guided_diversity_terms,
)

__all__ = [
    "FrozenDINOStateEncoder",
    "capture_parameter_gradients",
    "merge_quality_and_diversity_gradients",
    "teacher_guided_diversity_terms",
]
