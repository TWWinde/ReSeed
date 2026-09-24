"""Reproducible runtime contracts for Diffusers SD3-family pipelines.

Only small JSON configuration files are fingerprinted here.  A pinned Hub
revision supplies the base-weight identity, while distilled student weights
are hashed by evaluation manifests.  This module prevents a locally cached
SD3 snapshot (or a loaded pipeline) from being silently mistaken for a
different SD3-family architecture or FlowMatch schedule; it does not claim to
hash every base-model tensor byte.

Heavy Diffusers imports are intentionally avoided.  Contract validation works
with already-loaded pipelines and with lightweight fakes used by unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


_CONFIG_PATHS = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "vae/config.json",
)
_COMPONENT_CONFIG_PATHS = {
    "pipeline": "model_index.json",
    "scheduler": "scheduler/scheduler_config.json",
    "transformer": "transformer/config.json",
    "vae": "vae/config.json",
}


class SD3ContractError(ValueError):
    """Raised when a pipeline does not satisfy its declared SD3 contract."""


@dataclass(frozen=True)
class SD3FamilyContract:
    family: str
    qk_norm: str | None
    dual_attention_layers: tuple[int, ...]
    pos_embed_max_size: int
    shift: float = 3.0
    num_train_timesteps: int = 1000
    use_dynamic_shifting: bool = False
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 24
    attention_head_dim: int = 64
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    caption_projection_dim: int = 1536
    pooled_projection_dim: int = 2048
    patch_size: int = 2
    vae_scale_factor: int = 8


_SD35_MEDIUM_CONTRACT = SD3FamilyContract(
    family="sd3_5_medium",
    qk_norm="rms_norm",
    dual_attention_layers=tuple(range(13)),
    pos_embed_max_size=384,
)


SD3_FAMILY_CONTRACTS: Mapping[str, SD3FamilyContract] = {
    # The original SD3 Medium config predates the explicit qk_norm and
    # dual_attention_layers keys; absent values normalize to None and ().
    "sd3_medium": SD3FamilyContract(
        family="sd3_medium",
        qk_norm=None,
        dual_attention_layers=(),
        pos_embed_max_size=192,
    ),
    # SD3.5 Medium's MMDiT-X adds RMS QK normalization and dual attention in
    # transformer blocks 0 through 12 (the first 13 blocks).
    "sd3_5_medium": _SD35_MEDIUM_CONTRACT,
}


def _read_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"Required SD3 pipeline config is missing: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SD3ContractError(f"Invalid JSON in SD3 pipeline config {path}: {error}") from error
    if not isinstance(value, dict):
        raise SD3ContractError(f"SD3 pipeline config must contain a JSON object: {path}")
    return value, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _required_config_value(config: Any, key: str, owner: str) -> Any:
    sentinel = object()
    value = _config_value(config, key, sentinel)
    if value is sentinel:
        raise SD3ContractError(f"{owner} config is missing required field {key!r}")
    return value


def _declared_class(config: Any, owner: str) -> str:
    value = _required_config_value(config, "_class_name", owner)
    if not isinstance(value, str) or not value:
        raise SD3ContractError(f"{owner} config _class_name must be a non-empty string")
    return value


def _runtime_class_name(value: Any) -> str:
    """Return the real runtime class, with an explicit hook for test doubles."""

    override = getattr(value, "__sd3_contract_class_name__", None)
    if override is not None:
        if not isinstance(override, str) or not override:
            raise SD3ContractError("__sd3_contract_class_name__ must be a non-empty string")
        return override
    return type(value).__name__


def _vae_scale_factor(config: Any) -> int:
    channels = _required_config_value(config, "block_out_channels", "VAE")
    if not isinstance(channels, (list, tuple)) or not channels:
        raise SD3ContractError("VAE block_out_channels must be a non-empty list")
    return 2 ** (len(channels) - 1)


def _normalize_dual_attention_layers(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise SD3ContractError("transformer dual_attention_layers must be a list")
    try:
        normalized = tuple(int(index) for index in value)
    except (TypeError, ValueError) as error:
        raise SD3ContractError("transformer dual_attention_layers must contain integers") from error
    if len(normalized) != len(set(normalized)) or any(index < 0 for index in normalized):
        raise SD3ContractError("transformer dual_attention_layers must contain unique non-negative indices")
    return normalized


def _config_bool(config: Any, key: str, owner: str, *, default: bool) -> bool:
    value = _config_value(config, key, default)
    if not isinstance(value, bool):
        raise SD3ContractError(f"{owner} config {key} must be a boolean")
    return value


def _infer_revision(root: Path, configs: Mapping[str, dict[str, Any]]) -> str | None:
    revisions = {
        value
        for config in configs.values()
        for key in ("_commit_hash", "commit_hash")
        if isinstance((value := config.get(key)), str) and value
    }
    if len(revisions) > 1:
        raise SD3ContractError(f"Snapshot configs declare conflicting revisions: {sorted(revisions)}")
    configured = next(iter(revisions), None)
    path_revision = root.name if root.parent.name == "snapshots" else None
    if configured is not None and path_revision is not None and configured != path_revision:
        raise SD3ContractError(
            f"Snapshot path revision {path_revision!r} does not match config revision {configured!r}"
        )
    return configured or path_revision


def _snapshot_summary(configs: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    model_index = configs["model_index.json"]
    scheduler = configs["scheduler/scheduler_config.json"]
    transformer = configs["transformer/config.json"]
    vae = configs["vae/config.json"]
    return {
        "classes": {
            "pipeline": _declared_class(model_index, "pipeline"),
            "scheduler": _declared_class(scheduler, "scheduler"),
            "transformer": _declared_class(transformer, "transformer"),
            "vae": _declared_class(vae, "VAE"),
        },
        "scheduler": {
            "shift": float(_required_config_value(scheduler, "shift", "scheduler")),
            "num_train_timesteps": int(
                _required_config_value(scheduler, "num_train_timesteps", "scheduler")
            ),
            "use_dynamic_shifting": bool(
                _config_bool(scheduler, "use_dynamic_shifting", "scheduler", default=False)
            ),
        },
        "transformer": {
            "in_channels": int(_required_config_value(transformer, "in_channels", "transformer")),
            "out_channels": int(_required_config_value(transformer, "out_channels", "transformer")),
            "num_layers": int(_required_config_value(transformer, "num_layers", "transformer")),
            "attention_head_dim": int(
                _required_config_value(transformer, "attention_head_dim", "transformer")
            ),
            "num_attention_heads": int(
                _required_config_value(transformer, "num_attention_heads", "transformer")
            ),
            "joint_attention_dim": int(
                _required_config_value(transformer, "joint_attention_dim", "transformer")
            ),
            "caption_projection_dim": int(
                _required_config_value(transformer, "caption_projection_dim", "transformer")
            ),
            "pooled_projection_dim": int(
                _required_config_value(transformer, "pooled_projection_dim", "transformer")
            ),
            "patch_size": int(_required_config_value(transformer, "patch_size", "transformer")),
            "qk_norm": _config_value(transformer, "qk_norm"),
            "dual_attention_layers": list(
                _normalize_dual_attention_layers(_config_value(transformer, "dual_attention_layers", []))
            ),
            "pos_embed_max_size": int(
                _required_config_value(transformer, "pos_embed_max_size", "transformer")
            ),
        },
        "vae": {"scale_factor": _vae_scale_factor(vae)},
    }


def fingerprint_sd3_snapshot(
    model_path: str | Path,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Read and fingerprint the four identity-bearing SD3 pipeline configs.

    ``model_path`` must be a local Diffusers pipeline snapshot.  The aggregate
    ``config_sha256`` hashes ordered relative paths and raw file contents, so it
    is stable across machines and changes when any relevant config changes.
    If the path is under ``snapshots/<commit>``, that commit is recorded as the
    resolved revision.
    """

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SD3 pipeline snapshot does not exist: {root}")

    configs: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    for relative in _CONFIG_PATHS:
        config, raw = _read_json_object(root / relative)
        configs[relative] = config
        files[relative] = {"sha256": _sha256(raw), "size_bytes": len(raw)}
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(raw)
        aggregate.update(b"\0")

    inferred_revision = _infer_revision(root, configs)
    if revision is not None:
        if not isinstance(revision, str) or not revision:
            raise SD3ContractError("revision must be a non-empty string")
        if inferred_revision is not None and inferred_revision != revision:
            raise SD3ContractError(
                f"Requested revision {revision!r} does not match local snapshot revision {inferred_revision!r}"
            )
        resolved_revision = revision
    else:
        resolved_revision = inferred_revision

    fingerprint: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sd3_pipeline_config_fingerprint",
        "source_path": str(root),
        "revision": resolved_revision,
        "config_sha256": aggregate.hexdigest(),
        "files": files,
    }
    fingerprint.update(_snapshot_summary(configs))
    # Keep this invariant explicit: callers may safely persist the result in a
    # checkpoint or evaluation manifest without a custom encoder.
    json.dumps(fingerprint, sort_keys=True)
    return fingerprint


def _expect_equal(owner: str, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise SD3ContractError(f"{owner} {field}={actual!r}; expected {expected!r}")


def _expect_float(owner: str, field: str, actual: Any, expected: float) -> None:
    try:
        numeric = float(actual)
    except (TypeError, ValueError) as error:
        raise SD3ContractError(f"{owner} {field} must be numeric, got {actual!r}") from error
    if not math.isclose(numeric, expected, rel_tol=0.0, abs_tol=1e-9):
        raise SD3ContractError(f"{owner} {field}={numeric!r}; expected {expected!r}")


def _validate_summary(summary: Mapping[str, Any], contract: SD3FamilyContract, owner: str) -> None:
    classes = summary["classes"]
    _expect_equal(owner, "pipeline class", classes["pipeline"], "StableDiffusion3Pipeline")
    _expect_equal(owner, "transformer class", classes["transformer"], "SD3Transformer2DModel")
    _expect_equal(owner, "VAE class", classes["vae"], "AutoencoderKL")
    _expect_equal(
        owner,
        "scheduler class",
        classes["scheduler"],
        "FlowMatchEulerDiscreteScheduler",
    )
    scheduler = summary["scheduler"]
    _expect_float(owner, "scheduler shift", scheduler["shift"], contract.shift)
    _expect_equal(
        owner,
        "scheduler num_train_timesteps",
        int(scheduler["num_train_timesteps"]),
        contract.num_train_timesteps,
    )
    _expect_equal(
        owner,
        "scheduler use_dynamic_shifting",
        bool(scheduler["use_dynamic_shifting"]),
        contract.use_dynamic_shifting,
    )
    transformer = summary["transformer"]
    _expect_equal(owner, "transformer in_channels", int(transformer["in_channels"]), contract.in_channels)
    _expect_equal(owner, "transformer out_channels", int(transformer["out_channels"]), contract.out_channels)
    for field in (
        "num_layers",
        "attention_head_dim",
        "num_attention_heads",
        "joint_attention_dim",
        "caption_projection_dim",
        "pooled_projection_dim",
        "patch_size",
    ):
        _expect_equal(
            owner,
            f"transformer {field}",
            int(transformer[field]),
            getattr(contract, field),
        )
    _expect_equal(owner, "transformer qk_norm", transformer["qk_norm"], contract.qk_norm)
    _expect_equal(
        owner,
        "transformer dual_attention_layers",
        tuple(transformer["dual_attention_layers"]),
        contract.dual_attention_layers,
    )
    _expect_equal(
        owner,
        "transformer pos_embed_max_size",
        int(transformer["pos_embed_max_size"]),
        contract.pos_embed_max_size,
    )
    _expect_equal(owner, "VAE scale factor", int(summary["vae"]["scale_factor"]), contract.vae_scale_factor)


def _runtime_summary(pipe: Any) -> dict[str, Any]:
    for component in ("scheduler", "transformer", "vae"):
        if not hasattr(pipe, component):
            raise SD3ContractError(f"pipeline is missing required component {component!r}")
    scheduler_config = pipe.scheduler.config
    transformer_config = pipe.transformer.config
    vae_config = pipe.vae.config
    declared_vae_scale = getattr(pipe, "vae_scale_factor", None)
    computed_vae_scale = _vae_scale_factor(vae_config)
    if declared_vae_scale is not None and int(declared_vae_scale) != computed_vae_scale:
        raise SD3ContractError(
            f"pipeline vae_scale_factor={declared_vae_scale!r} does not match VAE config {computed_vae_scale!r}"
        )
    return {
        "classes": {
            "pipeline": _runtime_class_name(pipe),
            "scheduler": _runtime_class_name(pipe.scheduler),
            "transformer": _runtime_class_name(pipe.transformer),
            "vae": _runtime_class_name(pipe.vae),
        },
        "scheduler": {
            "shift": float(_required_config_value(scheduler_config, "shift", "scheduler")),
            "num_train_timesteps": int(
                _required_config_value(scheduler_config, "num_train_timesteps", "scheduler")
            ),
            "use_dynamic_shifting": bool(
                _config_bool(
                    scheduler_config,
                    "use_dynamic_shifting",
                    "scheduler",
                    default=False,
                )
            ),
        },
        "transformer": {
            "in_channels": int(
                _required_config_value(transformer_config, "in_channels", "transformer")
            ),
            "out_channels": int(
                _required_config_value(transformer_config, "out_channels", "transformer")
            ),
            "num_layers": int(
                _required_config_value(transformer_config, "num_layers", "transformer")
            ),
            "attention_head_dim": int(
                _required_config_value(transformer_config, "attention_head_dim", "transformer")
            ),
            "num_attention_heads": int(
                _required_config_value(transformer_config, "num_attention_heads", "transformer")
            ),
            "joint_attention_dim": int(
                _required_config_value(transformer_config, "joint_attention_dim", "transformer")
            ),
            "caption_projection_dim": int(
                _required_config_value(transformer_config, "caption_projection_dim", "transformer")
            ),
            "pooled_projection_dim": int(
                _required_config_value(transformer_config, "pooled_projection_dim", "transformer")
            ),
            "patch_size": int(
                _required_config_value(transformer_config, "patch_size", "transformer")
            ),
            "qk_norm": _config_value(transformer_config, "qk_norm"),
            "dual_attention_layers": list(
                _normalize_dual_attention_layers(
                    _config_value(transformer_config, "dual_attention_layers", [])
                )
            ),
            "pos_embed_max_size": int(
                _required_config_value(transformer_config, "pos_embed_max_size", "transformer")
            ),
        },
        "vae": {"scale_factor": computed_vae_scale},
    }


def validate_sd3_pipeline_contract(
    pipe: Any,
    *,
    family: str,
    model_path: str | Path | None = None,
    expected_config_sha256: str | None = None,
    expected_component_sha256: Mapping[str, str] | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Validate a loaded pipeline and return a JSON-serializable fingerprint.

    When ``model_path`` is supplied, runtime component values are also matched
    against the four local snapshot configs.  Expected aggregate SHA and
    revision checks require ``model_path`` so provenance cannot be asserted
    without evidence.
    """

    try:
        contract = SD3_FAMILY_CONTRACTS[family]
    except KeyError as error:
        supported = ", ".join(sorted(SD3_FAMILY_CONTRACTS))
        raise SD3ContractError(f"Unsupported SD3 family {family!r}; expected one of: {supported}") from error

    runtime = _runtime_summary(pipe)
    _validate_summary(runtime, contract, "runtime")

    if model_path is None and (
        expected_config_sha256 is not None
        or expected_component_sha256 is not None
        or expected_revision is not None
    ):
        raise SD3ContractError(
            "model_path is required when expected config hashes or revision are provided"
        )
    snapshot = (
        validate_sd3_snapshot_contract(
            model_path,
            family=family,
            expected_config_sha256=expected_config_sha256,
            expected_component_sha256=expected_component_sha256,
            expected_revision=expected_revision,
        )
        if model_path is not None
        else None
    )
    if snapshot is not None:
        for section in ("classes", "scheduler", "transformer", "vae"):
            _expect_equal("runtime", section, runtime[section], snapshot[section])

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sd3_runtime_contract",
        "family": family,
        "runtime": runtime,
        "snapshot": snapshot,
        "config_sha256": snapshot["config_sha256"] if snapshot is not None else None,
        "revision": snapshot["revision"] if snapshot is not None else None,
    }
    json.dumps(result, sort_keys=True)
    return result


def validate_sd3_snapshot_contract(
    model_path: str | Path,
    *,
    family: str,
    expected_config_sha256: str | None = None,
    expected_component_sha256: Mapping[str, str] | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Validate a local snapshot without importing Torch or Diffusers."""

    try:
        contract = SD3_FAMILY_CONTRACTS[family]
    except KeyError as error:
        supported = ", ".join(sorted(SD3_FAMILY_CONTRACTS))
        raise SD3ContractError(f"Unsupported SD3 family {family!r}; expected one of: {supported}") from error
    snapshot = fingerprint_sd3_snapshot(model_path)
    _validate_summary(snapshot, contract, "snapshot")
    if expected_config_sha256 is not None:
        _expect_equal(
            "snapshot", "config_sha256", snapshot["config_sha256"], expected_config_sha256
        )
    if expected_component_sha256 is not None:
        unknown = set(expected_component_sha256) - set(_COMPONENT_CONFIG_PATHS)
        if unknown:
            raise SD3ContractError(
                f"Unknown expected_component_sha256 keys: {sorted(unknown)}"
            )
        if not expected_component_sha256:
            raise SD3ContractError("expected_component_sha256 must not be empty")
        for component, expected_sha in expected_component_sha256.items():
            relative = _COMPONENT_CONFIG_PATHS[component]
            _expect_equal(
                "snapshot",
                f"{component} config sha256",
                snapshot["files"][relative]["sha256"],
                expected_sha,
            )
    if expected_revision is not None:
        _expect_equal("snapshot", "revision", snapshot["revision"], expected_revision)
    return snapshot


__all__ = [
    "SD3ContractError",
    "SD3FamilyContract",
    "SD3_FAMILY_CONTRACTS",
    "fingerprint_sd3_snapshot",
    "validate_sd3_snapshot_contract",
    "validate_sd3_pipeline_contract",
]
