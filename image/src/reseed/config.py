"""Strict, dependency-free experiment configuration contracts.

The legacy trainers retain their original flags.  Native future engines should
use these JSON contracts so every run has one resolved, validated description.
JSON was selected intentionally for the first integration version to avoid
adding a configuration framework to the validated training environment.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


_TOP_LEVEL = {"schema_version", "run", "model", "data", "train", "schedule", "objective", "checkpoint", "evaluation"}
_PIXART_OBJECTIVE_KEYS = {"kind", "gad_nsp", "crra"}
_SD3_OBJECTIVE_KEYS = {
    "kind", "gad_nsp", "dmd_tdm", "crra", "antipodal_teacher_reg",
    "data_forcing", "dp_dmd",
}
_DP_DMD_KEYS = {"k_anchor", "anchor_weight", "teacher_inference_steps"}
_DMD_TDM_KEYS = {
    "method", "lambda_fake_teacher_reg", "huber_base_c", "implementation",
    "sigma_sampling", "cfg_control", "student_gradient_scope",
    "student_fake_sample_coupling", "loss_weighting",
    "allow_unanchored_frequency_training", "fake_score_tempering",
}
FAKE_SCORE_TEMPERING_DEFAULTS = {
    "strength": 0.0,
    "sigma_start": 0.6,
    "sigma_full": 0.9,
    "start_student_update": 200,
    "warmup_student_updates": 200,
}
_SD3_MODEL_KEYS = {
    "family", "pretrained_path", "student_initialization", "prediction_type", "resolution",
    "max_sequence_length", "fake_score_initialization",
}
_SD35_MODEL_KEYS = _SD3_MODEL_KEYS | {
    "source_model_id", "revision", "pipeline_class", "transformer_class", "latent_channels", "vae_scale_factor",
    "qk_norm", "dual_attention_layers", "pos_embed_max_size", "config_sha256",
}
_SD35_CONFIG_SHA_KEYS = {"pipeline", "transformer", "scheduler", "vae"}
_SD3_TRAIN_KEYS = {
    "max_steps", "batch_size_per_device", "gradient_accumulation_steps", "mixed_precision",
    "learning_rate", "fake_learning_rate", "fake_updates_per_student", "adam_betas",
    "adam_epsilon", "weight_decay", "student_max_grad_norm", "fake_max_grad_norm",
    "gradient_checkpointing", "trainable_blocks", "initial_rollout_noise",
    "initial_rollout_noise_lowpass_cutoff", "initial_rollout_noise_lowpass_fraction",
}
_FLOW_SCHEDULE_KEYS = {
    "kind", "num_train_timesteps", "student_steps", "shift", "teacher_guidance_scale",
    "student_guidance_scale", "teacher_cfg_threshold_a", "teacher_cfg_symmetric_span",
}
_SD35_FLOW_SCHEDULE_KEYS = _FLOW_SCHEDULE_KEYS | {
    "reference_inference_steps", "guidance_mode", "use_dynamic_shifting", "student_sigma_grid",
}
_CRRA_KEYS = {
    "loss_scale", "teacher_solver_steps", "interval_indices", "lambda_tf", "lambda_op",
    "lambda_crra", "trust_kappa", "start_student_update", "warmup_student_updates",
    "every_student_updates", "state_distance", "lpips_net",
}
_GAD_NSP_KEYS = {"lambda_nsp", "epsilon", "loss", "huber_c"}
_ANTIPODAL_TEACHER_REG_KEYS = {
    "scale", "pair_lambda", "teacher_solver_steps", "interval_index",
    "start_student_update", "warmup_student_updates", "every_student_updates",
    "loss", "huber_c",
}
_DATA_FORCING_KEYS = {
    "enabled", "reference_manifest", "replacement_fraction", "sigma_gate",
    "curriculum_easy_steps", "curriculum_medium_steps", "distribution",
}


SD35_MEDIUM_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
SD35_MEDIUM_CONFIG_SHA256 = {
    "pipeline": "b5e97ca0d3c0cd9a1e3ed91a71bbdd489de7941b23fe038b3aaa9acf36fd1543",
    "transformer": "b6ea8128277df286573ce1937c36ce42a5a84e976331091ffccb1d05893dc299",
    "scheduler": "677e1ed36ca0de4ac8cc766ca58869e8649673d5f1d47eb8f8aef511a18ae8f7",
    "vae": "58557f2439dfa867450caef425b5d11160be8aa9c34d60dbf23a94a6a94cb060",
}


class ConfigError(ValueError):
    pass


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {name} keys: {', '.join(unknown)}")


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in {source}: {error}") from error
    return validate_config(payload)


def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the cross-method invariants needed for fair experiments."""
    payload = deepcopy(_mapping(payload, "config"))
    _reject_unknown(payload, _TOP_LEVEL, "top-level")
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2, 3}:
        raise ConfigError("schema_version must be 1, 2, or 3")

    for name in ("run", "model", "data", "train", "schedule", "objective", "checkpoint"):
        _mapping(payload.get(name), name)

    if schema_version == 2:
        return _validate_sd3_flow_config(payload)
    if schema_version == 3:
        return _validate_sd35_flow_config(payload)

    return _validate_pixart_v1_config(payload)


def _validate_pixart_v1_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the original PixArt epsilon/two-optimizer contract unchanged."""

    model = _mapping(payload["model"], "model")
    train = _mapping(payload["train"], "train")
    schedule = _mapping(payload["schedule"], "schedule")
    objective = _mapping(payload["objective"], "objective")
    _reject_unknown(objective, _PIXART_OBJECTIVE_KEYS, "objective")

    if model.get("prediction_type") != "epsilon":
        raise ConfigError("model.prediction_type must be 'epsilon'")
    if train.get("mixed_precision") != "fp16":
        raise ConfigError("The current PixArt two-optimizer implementations require train.mixed_precision='fp16'")
    if train.get("gradient_accumulation_steps") != 1:
        raise ConfigError("train.gradient_accumulation_steps must be 1 for the current two-optimizer implementations")

    total_steps = schedule.get("total_steps")
    student_steps = schedule.get("student_steps")
    if not isinstance(total_steps, int) or not 0 < total_steps <= 1000:
        raise ConfigError("schedule.total_steps must be an integer in [1, 1000]")
    if not isinstance(student_steps, int) or student_steps <= 0:
        raise ConfigError("schedule.student_steps must be a positive integer")
    if total_steps % student_steps != 0:
        raise ConfigError("schedule.total_steps must be divisible by schedule.student_steps")

    kind = objective.get("kind")
    if kind not in {"tdm", "gad_nsp", "crra"}:
        raise ConfigError("objective.kind must be one of: tdm, gad_nsp, crra")
    gad = _mapping(objective.get("gad_nsp", {}), "objective.gad_nsp")
    crra = _mapping(objective.get("crra", {}), "objective.crra")
    nsp = _positive_number(
        gad.get("lambda_nsp", 0.0), "objective.gad_nsp.lambda_nsp", allow_zero=True
    )
    crra_active = float(crra.get("loss_scale", 0.0)) > 0.0

    if kind == "tdm" and (nsp != 0.0 or crra_active):
        raise ConfigError("objective.kind=tdm requires lambda_nsp=0 and crra.loss_scale=0")
    if kind == "gad_nsp":
        if nsp <= 0.0:
            raise ConfigError("objective.kind=gad_nsp requires gad_nsp.lambda_nsp > 0")
        if crra_active:
            raise ConfigError("GAD NSP and CRRA are mutually exclusive")
    if kind == "crra":
        if nsp != 0.0:
            raise ConfigError("CRRA requires gad_nsp.lambda_nsp=0")
        if not crra_active:
            raise ConfigError("objective.kind=crra requires crra.loss_scale > 0")
        _validate_crra_state_distance(crra)
        teacher_steps = crra.get("teacher_solver_steps")
        if not isinstance(teacher_steps, int) or teacher_steps <= 0 or teacher_steps % student_steps != 0:
            raise ConfigError("crra.teacher_solver_steps must be positive and divisible by schedule.student_steps")
        intervals = crra.get("interval_indices", [])
        if not isinstance(intervals, list) or not intervals:
            raise ConfigError("crra.interval_indices must be a non-empty list")
        invalid = [index for index in intervals if not isinstance(index, int) or index < 0 or index >= student_steps]
        if invalid:
            raise ConfigError(f"crra.interval_indices outside [0, {student_steps - 1}]: {invalid}")
    return payload


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must be numeric") from error
    invalid = number < 0 if allow_zero else number <= 0
    if not math.isfinite(number) or invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{name} must be {qualifier}")
    return number


def _validate_crra_state_distance(crra: dict[str, Any]) -> None:
    distance = crra.get("state_distance", "mse")
    if distance not in {"mse", "lpips"}:
        raise ConfigError("objective.crra.state_distance must be 'mse' or 'lpips'")
    if distance == "lpips":
        if crra.get("lpips_net", "alex") not in {"alex", "vgg", "squeeze"}:
            raise ConfigError("objective.crra.lpips_net must be 'alex', 'vgg', or 'squeeze'")
    elif "lpips_net" in crra:
        raise ConfigError("objective.crra.lpips_net is only valid with state_distance='lpips'")


def _validate_fake_score_tempering(
    dmd_tdm: dict[str, Any], *, model_family: str, objective_kind: str
) -> None:
    # Do not inject disabled fields into old configs: their canonical JSON and
    # checkpoint hashes must remain unchanged. An explicitly supplied object is
    # resolved in full so the opt-in experiment records every schedule parameter.
    if "fake_score_tempering" not in dmd_tdm:
        return
    name = "objective.dmd_tdm.fake_score_tempering"
    tempering = _mapping(dmd_tdm["fake_score_tempering"], name)
    _reject_unknown(tempering, set(FAKE_SCORE_TEMPERING_DEFAULTS), name)
    for key, default in FAKE_SCORE_TEMPERING_DEFAULTS.items():
        tempering.setdefault(key, default)
    for key in ("strength", "sigma_start", "sigma_full"):
        if isinstance(tempering[key], bool):
            raise ConfigError(f"{name}.{key} must be numeric, not boolean")
        tempering[key] = _positive_number(tempering[key], f"{name}.{key}", allow_zero=True)
    if not 0.0 <= tempering["sigma_start"] < tempering["sigma_full"] < 1.0:
        raise ConfigError(f"{name} requires 0 <= sigma_start < sigma_full < 1")
    for key in ("start_student_update", "warmup_student_updates"):
        value = tempering[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"{name}.{key} must be a non-negative integer")
    if tempering["strength"] > 0.0 and (
        model_family != "sd3_medium" or objective_kind not in {"tdm", "dmd"}
    ):
        raise ConfigError(f"{name} with positive strength requires SD3 Medium tdm/dmd")
    # Positive strength changes the fake-score target; it is NOT exact vanilla
    # sd_distill target equivalence. The implementation identifier below still
    # describes the unchanged base sampler/loss contract, not this opt-in target.


def _validate_sd3_flow_config(
    payload: dict[str, Any],
    *,
    expected_family: str = "sd3_medium",
    model_keys: set[str] = _SD3_MODEL_KEYS,
    schedule_keys: set[str] = _FLOW_SCHEDULE_KEYS,
    model_label: str = "SD3 Medium",
    backend_label: str = "native SD3M",
) -> dict[str, Any]:
    """Validate native SD3M classic Flow-DMD/TDM, optionally with CRRA."""

    model = _mapping(payload["model"], "model")
    train = _mapping(payload["train"], "train")
    schedule = _mapping(payload["schedule"], "schedule")
    objective = _mapping(payload["objective"], "objective")
    data = _mapping(payload["data"], "data")
    _reject_unknown(model, model_keys, "model")
    _reject_unknown(train, _SD3_TRAIN_KEYS, "train")
    _reject_unknown(schedule, schedule_keys, "schedule")
    _reject_unknown(objective, _SD3_OBJECTIVE_KEYS, "objective")

    if model.get("family") != expected_family:
        if expected_family == "sd3_medium":
            raise ConfigError("schema v2 currently requires model.family='sd3_medium'")
        raise ConfigError(f"model.family must be {expected_family!r}")
    if model.get("prediction_type") != "flow":
        raise ConfigError(f"{model_label} requires model.prediction_type='flow'")
    if not isinstance(model.get("pretrained_path"), str) or not model["pretrained_path"]:
        raise ConfigError("model.pretrained_path must be a non-empty string")
    for name in ("student_initialization", "fake_score_initialization"):
        value = model.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            raise ConfigError(f"model.{name} must be a non-empty string when provided")
    resolution = model.get("resolution")
    if not isinstance(resolution, int) or resolution <= 0 or resolution % 8 != 0:
        raise ConfigError("model.resolution must be a positive multiple of 8")

    if train.get("mixed_precision") not in {"bf16", "fp16"}:
        raise ConfigError(f"{backend_label} requires train.mixed_precision='bf16' or 'fp16'")
    if train.get("gradient_accumulation_steps") != 1:
        raise ConfigError(f"{backend_label} currently requires train.gradient_accumulation_steps=1")
    if train.get("batch_size_per_device") != 1:
        raise ConfigError(f"{backend_label} currently requires train.batch_size_per_device=1")
    max_steps = train.get("max_steps")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ConfigError("train.max_steps must be a positive integer")
    _positive_number(train.get("learning_rate"), "train.learning_rate")
    _positive_number(train.get("fake_learning_rate"), "train.fake_learning_rate")
    fake_updates = train.get("fake_updates_per_student")
    if not isinstance(fake_updates, int) or fake_updates <= 0:
        raise ConfigError("train.fake_updates_per_student must be a positive integer")
    _positive_number(train.get("student_max_grad_norm", 2.0), "train.student_max_grad_norm")
    _positive_number(train.get("fake_max_grad_norm", 1.0), "train.fake_max_grad_norm")
    trainable_blocks = train.get("trainable_blocks", "all")
    if trainable_blocks != "all" and (
        not isinstance(trainable_blocks, list)
        or not trainable_blocks
        or any(not isinstance(index, int) for index in trainable_blocks)
    ):
        raise ConfigError("train.trainable_blocks must be 'all' or a non-empty integer list")
    initial_rollout_noise = train.get("initial_rollout_noise", "gaussian")
    if initial_rollout_noise not in {
        "gaussian",
        "lowpass",
        "mixed",
        "paired",
        "repulsive",
    }:
        raise ConfigError(
            "train.initial_rollout_noise must be one of "
            "'gaussian', 'lowpass', 'mixed', 'paired', or 'repulsive'"
        )
    lowpass_cutoff = _positive_number(
        train.get("initial_rollout_noise_lowpass_cutoff", 0.20),
        "train.initial_rollout_noise_lowpass_cutoff",
    )
    if lowpass_cutoff > 1.0:
        raise ConfigError("train.initial_rollout_noise_lowpass_cutoff must be <= 1")
    lowpass_fraction = _positive_number(
        train.get("initial_rollout_noise_lowpass_fraction", 0.50),
        "train.initial_rollout_noise_lowpass_fraction",
    )
    if lowpass_fraction >= 1.0:
        raise ConfigError("train.initial_rollout_noise_lowpass_fraction must be < 1")
    if initial_rollout_noise == "paired" and not math.isclose(
        lowpass_fraction, 0.50, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ConfigError("train.initial_rollout_noise='paired' requires lowpass_fraction=0.5")

    if schedule.get("kind") != "flow_matching":
        raise ConfigError(f"{model_label} requires schedule.kind='flow_matching'")
    num_train_timesteps = schedule.get("num_train_timesteps")
    if not isinstance(num_train_timesteps, int) or num_train_timesteps <= 0:
        raise ConfigError("schedule.num_train_timesteps must be a positive integer")
    student_steps = schedule.get("student_steps")
    if not isinstance(student_steps, int) or student_steps <= 0:
        raise ConfigError("schedule.student_steps must be a positive integer")
    if num_train_timesteps % student_steps != 0:
        raise ConfigError("schedule.num_train_timesteps must be divisible by schedule.student_steps")
    _positive_number(schedule.get("shift"), "schedule.shift")
    _positive_number(schedule.get("teacher_guidance_scale"), "schedule.teacher_guidance_scale")
    teacher_cfg_threshold = schedule.get("teacher_cfg_threshold_a")
    if teacher_cfg_threshold is not None:
        teacher_cfg_threshold = _positive_number(
            teacher_cfg_threshold, "schedule.teacher_cfg_threshold_a"
        )
        if teacher_cfg_threshold >= 1.0:
            raise ConfigError("schedule.teacher_cfg_threshold_a must be in (0, 1)")
    teacher_cfg_symmetric_span = schedule.get("teacher_cfg_symmetric_span")
    if teacher_cfg_symmetric_span is not None:
        teacher_cfg_symmetric_span = _positive_number(
            teacher_cfg_symmetric_span, "schedule.teacher_cfg_symmetric_span"
        )
        if teacher_cfg_symmetric_span >= float(schedule["teacher_guidance_scale"]):
            raise ConfigError(
                "schedule.teacher_cfg_symmetric_span must be smaller than teacher_guidance_scale"
            )
        if teacher_cfg_threshold is not None:
            raise ConfigError(
                "teacher_cfg_symmetric_span and teacher_cfg_threshold_a are mutually exclusive"
            )
    student_guidance = _positive_number(
        schedule.get("student_guidance_scale"), "schedule.student_guidance_scale"
    )
    if student_guidance != 1.0:
        raise ConfigError(f"{backend_label} requires schedule.student_guidance_scale=1 for one student forward")

    kind = objective.get("kind")
    if kind not in {"dmd", "tdm", "gad_nsp", "crra", "dp_dmd"}:
        raise ConfigError(
            f"{backend_label} objective.kind must be 'dmd', 'tdm', 'gad_nsp', 'crra', or 'dp_dmd'"
        )
    gad = _mapping(objective.get("gad_nsp", {}), "objective.gad_nsp")
    _reject_unknown(gad, _GAD_NSP_KEYS, "objective.gad_nsp")
    nsp = _positive_number(
        gad.get("lambda_nsp", 0.0), "objective.gad_nsp.lambda_nsp", allow_zero=True
    )
    if kind != "gad_nsp" and nsp != 0.0:
        raise ConfigError(f"{backend_label} non-GAD objectives require gad_nsp.lambda_nsp=0")
    if kind == "gad_nsp":
        if nsp <= 0.0:
            raise ConfigError(f"{backend_label} objective.kind=gad_nsp requires lambda_nsp > 0")
        _positive_number(gad.get("epsilon", 0.01), "objective.gad_nsp.epsilon")
        if gad.get("loss", "pseudo_huber") not in {"mse", "pseudo_huber"}:
            raise ConfigError("objective.gad_nsp.loss must be 'mse' or 'pseudo_huber'")
        _positive_number(gad.get("huber_c", 0.001), "objective.gad_nsp.huber_c")
    dp_dmd = _mapping(objective.get("dp_dmd", {}), "objective.dp_dmd")
    _reject_unknown(dp_dmd, _DP_DMD_KEYS, "objective.dp_dmd")
    if kind == "dp_dmd":
        # DP-DMD (Wu et al. 2026): the first student step regresses onto the
        # teacher's k_anchor-step secant; DMD trains only the later intervals.
        k_anchor = dp_dmd.get("k_anchor")
        if not isinstance(k_anchor, int) or isinstance(k_anchor, bool) or k_anchor <= 0:
            raise ConfigError("objective.dp_dmd.k_anchor must be a positive integer")
        teacher_inference_steps = dp_dmd.get(
            "teacher_inference_steps", schedule.get("reference_inference_steps", 28)
        )
        if (
            not isinstance(teacher_inference_steps, int)
            or isinstance(teacher_inference_steps, bool)
            or teacher_inference_steps <= 0
        ):
            raise ConfigError("objective.dp_dmd.teacher_inference_steps must be a positive integer")
        if "reference_inference_steps" in schedule and schedule["reference_inference_steps"] != teacher_inference_steps:
            raise ConfigError(
                "objective.dp_dmd.teacher_inference_steps must equal schedule.reference_inference_steps"
            )
        if k_anchor > teacher_inference_steps:
            raise ConfigError("objective.dp_dmd.k_anchor cannot exceed the teacher inference step count")
        _positive_number(dp_dmd.get("anchor_weight"), "objective.dp_dmd.anchor_weight")
        dp_dmd["teacher_inference_steps"] = teacher_inference_steps
    elif dp_dmd:
        raise ConfigError("objective.dp_dmd is only valid with objective.kind=dp_dmd")
    dmd_tdm = _mapping(objective.get("dmd_tdm"), "objective.dmd_tdm")
    _reject_unknown(dmd_tdm, _DMD_TDM_KEYS, "objective.dmd_tdm")
    _validate_fake_score_tempering(dmd_tdm, model_family=expected_family, objective_kind=kind)
    if dmd_tdm.get("method") != "classic":
        raise ConfigError(f"{backend_label} requires objective.dmd_tdm.method='classic'")
    # Resolve every behavior that historically differed from sd_distill.  The
    # values are injected into the canonical config so checkpoint hashes and
    # manifests cannot silently mix the old native approximation with the
    # reference Stage-1 objective.
    reference_contract = {
        "implementation": "sd_distill_classic_v1",
        "sigma_sampling": "points_mode_c",
        "cfg_control": 0.96,
        "student_gradient_scope": "local_interval",
        "student_fake_sample_coupling": "reuse_last_fake_rollout",
        "loss_weighting": "channel_height_mean_keep_width",
    }
    for name, expected in reference_contract.items():
        dmd_tdm.setdefault(name, expected)
        actual = dmd_tdm[name]
        if isinstance(expected, float):
            try:
                matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise ConfigError(
                f"objective.dmd_tdm.{name} must be {expected!r} for exact sd_distill parity"
            )
    if num_train_timesteps != 1000:
        raise ConfigError("exact sd_distill classic Stage-1 requires schedule.num_train_timesteps=1000")
    fake_teacher_reg = _positive_number(
        dmd_tdm.get("lambda_fake_teacher_reg", 0.0),
        "objective.dmd_tdm.lambda_fake_teacher_reg",
        allow_zero=True,
    )
    allow_unanchored_frequency_training = dmd_tdm.get(
        "allow_unanchored_frequency_training", False
    )
    if not isinstance(allow_unanchored_frequency_training, bool):
        raise ConfigError(
            "objective.dmd_tdm.allow_unanchored_frequency_training must be boolean"
        )
    if kind in {"dmd", "dp_dmd"} and fake_teacher_reg != 0.0:
        raise ConfigError(f"objective.kind={kind} requires lambda_fake_teacher_reg=0")
    # CRRA is orthogonal to the choice of DMD base.  A positive coefficient
    # gives the TDM-regularized base used by the original SD3M experiments;
    # zero gives the DMD-only base + CRRA ablation.  TDM and GAD labels still
    # require the teacher regularizer so their existing semantics cannot drift.
    if kind in {"tdm", "gad_nsp"} and fake_teacher_reg <= 0.0:
        raise ConfigError(f"objective.kind={kind} requires lambda_fake_teacher_reg > 0")
    # A hard low-pass transform removes high-frequency degrees of freedom; RMS
    # matching restores only scalar energy, not the isotropic Gaussian prior
    # assumed by the SD3 Flow-Matching teacher.  The original SD3M-1024
    # DMD-only/low-pass experiment consequently admitted a self-consistent but
    # visually collapsed student/fake-score fixed point.  Keep pure low-pass
    # available in evaluation diagnostics, but reject it as a training prior.
    if initial_rollout_noise == "lowpass":
        raise ConfigError(
            "native SD3 frequency training forbids 100% low-pass rollout noise; "
            "use 'mixed' or 'paired' so the training prior retains Gaussian support"
        )
    # Frequency-shaped marginals need the same fake-score teacher anchor used
    # by sd_distill's lambda_reg branch.  A zero coefficient is the intentional
    # Gaussian DMD-only baseline, not a safe frequency-training recipe.
    if (
        initial_rollout_noise in {"mixed", "paired"}
        and fake_teacher_reg <= 0.0
        and not allow_unanchored_frequency_training
    ):
        raise ConfigError(
            "native SD3 frequency-shaped rollout noise requires "
            "objective.dmd_tdm.lambda_fake_teacher_reg > 0 (TDM anchor), or an "
            "explicit allow_unanchored_frequency_training=true experiment override"
        )
    _positive_number(dmd_tdm.get("huber_base_c"), "objective.dmd_tdm.huber_base_c")

    crra = _mapping(objective.get("crra", {}), "objective.crra")
    _reject_unknown(crra, _CRRA_KEYS, "objective.crra")
    loss_scale = _positive_number(
        crra.get("loss_scale", 0.0), "objective.crra.loss_scale", allow_zero=True
    )

    antipodal = _mapping(
        objective.get("antipodal_teacher_reg", {}),
        "objective.antipodal_teacher_reg",
    )
    _reject_unknown(
        antipodal,
        _ANTIPODAL_TEACHER_REG_KEYS,
        "objective.antipodal_teacher_reg",
    )
    antipodal_scale = _positive_number(
        antipodal.get("scale", 0.0),
        "objective.antipodal_teacher_reg.scale",
        allow_zero=True,
    )
    if antipodal_scale > 0.0:
        if kind not in {"dmd", "tdm"}:
            raise ConfigError(
                "SD3 antipodal teacher regularization requires objective.kind='dmd' or 'tdm'"
            )
        _positive_number(
            antipodal.get("pair_lambda", 1.0),
            "objective.antipodal_teacher_reg.pair_lambda",
            allow_zero=True,
        )
        teacher_steps = antipodal.get("teacher_solver_steps", 28)
        if not isinstance(teacher_steps, int) or teacher_steps <= 0:
            raise ConfigError(
                "objective.antipodal_teacher_reg.teacher_solver_steps must be a positive integer"
            )
        if teacher_steps % student_steps:
            raise ConfigError(
                "antipodal teacher_solver_steps must be divisible by schedule.student_steps"
            )
        interval_index = antipodal.get("interval_index", 0)
        if interval_index != 0:
            raise ConfigError(
                "objective.antipodal_teacher_reg.interval_index must be 0 so z and -z "
                "remain exact antipodes at the input of the measured student transition"
            )
        for name, default, positive in (
            ("start_student_update", 0, False),
            ("warmup_student_updates", 0, False),
            ("every_student_updates", 1, True),
        ):
            value = antipodal.get(name, default)
            if not isinstance(value, int) or value < int(positive):
                qualifier = "positive" if positive else "non-negative"
                raise ConfigError(
                    f"objective.antipodal_teacher_reg.{name} must be a {qualifier} integer"
                )
        if antipodal.get("loss", "huber") not in {"mse", "huber"}:
            raise ConfigError("objective.antipodal_teacher_reg.loss must be 'mse' or 'huber'")
        _positive_number(
            antipodal.get("huber_c", 1e-3),
            "objective.antipodal_teacher_reg.huber_c",
        )

    data_forcing = _mapping(
        objective.get("data_forcing", {}), "objective.data_forcing"
    )
    _reject_unknown(
        data_forcing, _DATA_FORCING_KEYS, "objective.data_forcing"
    )
    data_forcing_enabled = data_forcing.get("enabled", False)
    if not isinstance(data_forcing_enabled, bool):
        raise ConfigError("objective.data_forcing.enabled must be boolean")
    if data_forcing_enabled:
        if expected_family != "sd3_medium" or kind not in {"dmd", "tdm"}:
            raise ConfigError("data forcing requires SD3 Medium dmd/tdm")
        reference_manifest = data_forcing.get("reference_manifest")
        if not isinstance(reference_manifest, str) or not reference_manifest:
            raise ConfigError(
                "objective.data_forcing.reference_manifest must be a non-empty string"
            )
        replacement_fraction = _positive_number(
            data_forcing.get("replacement_fraction"),
            "objective.data_forcing.replacement_fraction",
        )
        if replacement_fraction > 1.0:
            raise ConfigError(
                "objective.data_forcing.replacement_fraction must be <= 1"
            )
        sigma_gate = _positive_number(
            data_forcing.get("sigma_gate"),
            "objective.data_forcing.sigma_gate",
            allow_zero=True,
        )
        if sigma_gate >= 1.0:
            raise ConfigError("objective.data_forcing.sigma_gate must be in [0, 1)")
        easy_steps = data_forcing.get("curriculum_easy_steps")
        medium_steps = data_forcing.get("curriculum_medium_steps")
        if not isinstance(easy_steps, int) or easy_steps < 0:
            raise ConfigError(
                "objective.data_forcing.curriculum_easy_steps must be non-negative"
            )
        if not isinstance(medium_steps, int) or medium_steps < easy_steps:
            raise ConfigError(
                "objective.data_forcing.curriculum_medium_steps must be >= easy_steps"
            )
        distribution = data_forcing.get("distribution", "teacher_query_only")
        if distribution not in {"teacher_query_only", "official_multistep"}:
            raise ConfigError(
                "objective.data_forcing.distribution must be "
                "'teacher_query_only' or 'official_multistep'"
            )
        if distribution == "official_multistep" and sigma_gate != 0.0:
            raise ConfigError(
                "official_multistep data forcing requires sigma_gate=0"
            )
    if kind in {"dmd", "tdm", "gad_nsp", "dp_dmd"}:
        if loss_scale != 0.0:
            raise ConfigError(f"objective.kind={kind} requires objective.crra.loss_scale=0")
        if not isinstance(data.get("prompt_file"), str) or not data["prompt_file"]:
            raise ConfigError("data.prompt_file must be a non-empty string")
        return payload
    if loss_scale <= 0.0:
        raise ConfigError("objective.kind=crra requires objective.crra.loss_scale > 0")
    _validate_crra_state_distance(crra)
    teacher_steps = crra.get("teacher_solver_steps")
    if not isinstance(teacher_steps, int) or teacher_steps <= 0 or teacher_steps % student_steps != 0:
        raise ConfigError(
            "objective.crra.teacher_solver_steps must be positive and divisible by schedule.student_steps"
        )
    intervals = crra.get("interval_indices")
    if not isinstance(intervals, list) or not intervals:
        raise ConfigError("objective.crra.interval_indices must be a non-empty list")
    invalid = [index for index in intervals if not isinstance(index, int) or index < 0 or index >= student_steps]
    if invalid:
        raise ConfigError(f"objective.crra.interval_indices outside [0, {student_steps - 1}]: {invalid}")
    for name in ("lambda_tf", "lambda_op", "lambda_crra"):
        _positive_number(crra.get(name), f"objective.crra.{name}", allow_zero=True)
    if not any(float(crra[name]) > 0 for name in ("lambda_tf", "lambda_op", "lambda_crra")):
        raise ConfigError(f"at least one {model_label} CRRA loss coefficient must be positive")
    _positive_number(crra.get("trust_kappa"), "objective.crra.trust_kappa")
    for name in ("start_student_update", "warmup_student_updates"):
        value = crra.get(name, 0)
        if not isinstance(value, int) or value < 0:
            raise ConfigError(f"objective.crra.{name} must be a non-negative integer")
    every = crra.get("every_student_updates", 1)
    if not isinstance(every, int) or every <= 0:
        raise ConfigError("objective.crra.every_student_updates must be a positive integer")

    if not isinstance(data.get("prompt_file"), str) or not data["prompt_file"]:
        raise ConfigError("data.prompt_file must be a non-empty string")
    return payload


def _validate_sd35_flow_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the pinned SD3.5 Medium MMDiT-X FlowMatch contract.

    Schema v3 is deliberately additive.  It does not loosen schema v2 or let a
    moving Hub revision silently change the architecture/scheduler underneath
    a run.
    """

    run = _mapping(payload["run"], "run")
    if run.get("backend") != "sd35m-native":
        raise ConfigError("schema v3 requires run.backend='sd35m-native'")

    model = _mapping(payload["model"], "model")
    if model.get("source_model_id") != "stabilityai/stable-diffusion-3.5-medium":
        raise ConfigError(
            "SD3.5 Medium requires model.source_model_id='stabilityai/stable-diffusion-3.5-medium'"
        )
    if model.get("revision") != SD35_MEDIUM_REVISION:
        raise ConfigError(f"SD3.5 Medium requires model.revision={SD35_MEDIUM_REVISION!r}")
    if model.get("pipeline_class") != "StableDiffusion3Pipeline":
        raise ConfigError("SD3.5 Medium requires model.pipeline_class='StableDiffusion3Pipeline'")
    if model.get("transformer_class") != "SD3Transformer2DModel":
        raise ConfigError("SD3.5 Medium requires model.transformer_class='SD3Transformer2DModel'")
    if model.get("latent_channels") != 16:
        raise ConfigError("SD3.5 Medium requires model.latent_channels=16")
    if model.get("vae_scale_factor") != 8:
        raise ConfigError("SD3.5 Medium requires model.vae_scale_factor=8")
    if model.get("qk_norm") != "rms_norm":
        raise ConfigError("SD3.5 Medium requires model.qk_norm='rms_norm'")
    if model.get("dual_attention_layers") != list(range(13)):
        raise ConfigError("SD3.5 Medium requires model.dual_attention_layers=[0, ..., 12]")
    if model.get("pos_embed_max_size") != 384:
        raise ConfigError("SD3.5 Medium requires model.pos_embed_max_size=384")
    config_sha256 = _mapping(model.get("config_sha256"), "model.config_sha256")
    _reject_unknown(config_sha256, _SD35_CONFIG_SHA_KEYS, "model.config_sha256")
    if config_sha256 != SD35_MEDIUM_CONFIG_SHA256:
        raise ConfigError("SD3.5 Medium model.config_sha256 does not match the pinned component configs")

    schedule = _mapping(payload["schedule"], "schedule")
    if schedule.get("num_train_timesteps") != 1000:
        raise ConfigError("SD3.5 Medium requires schedule.num_train_timesteps=1000")
    if schedule.get("reference_inference_steps") != 40:
        raise ConfigError("SD3.5 Medium requires schedule.reference_inference_steps=40")
    # schedule.teacher_guidance_scale is the *distillation* teacher's CFG, which
    # the teacher-CFG scale study sweeps exactly as it does on SD3-Medium. It is
    # not the evaluation reference CFG: that one stays pinned at 4.5 by the
    # immutable SD3.5M evaluation protocols and by reference_inference_steps=40
    # above, and nothing here can move it.
    try:
        teacher_guidance = float(schedule.get("teacher_guidance_scale"))
    except (TypeError, ValueError) as error:
        raise ConfigError("schedule.teacher_guidance_scale must be numeric") from error
    if not math.isfinite(teacher_guidance) or teacher_guidance <= 0.0:
        raise ConfigError(
            "SD3.5 Medium requires a finite positive schedule.teacher_guidance_scale"
        )
    if schedule.get("guidance_mode") != "classic_cfg_no_slg":
        raise ConfigError("SD3.5 Medium requires schedule.guidance_mode='classic_cfg_no_slg'")
    if schedule.get("use_dynamic_shifting") is not False:
        raise ConfigError("SD3.5 Medium requires schedule.use_dynamic_shifting=false")
    if schedule.get("student_sigma_grid") != "uniform_unshifted":
        raise ConfigError("SD3.5 Medium requires schedule.student_sigma_grid='uniform_unshifted'")
    try:
        shift = float(schedule.get("shift"))
    except (TypeError, ValueError) as error:
        raise ConfigError("schedule.shift must be numeric") from error
    if not math.isfinite(shift) or shift != 3.0:
        raise ConfigError("SD3.5 Medium requires schedule.shift=3.0")

    validated = _validate_sd3_flow_config(
        payload,
        expected_family="sd3_5_medium",
        model_keys=_SD35_MODEL_KEYS,
        schedule_keys=_SD35_FLOW_SCHEDULE_KEYS,
        model_label="SD3.5 Medium",
        backend_label="native SD3.5M",
    )
    crra = validated["objective"].get("crra", {})
    if validated["objective"].get("kind") == "crra" and crra.get("teacher_solver_steps") != 40:
        raise ConfigError("SD3.5 Medium CRRA requires objective.crra.teacher_solver_steps=40")
    return validated


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
