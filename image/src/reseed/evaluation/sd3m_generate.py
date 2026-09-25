#!/usr/bin/env python3
"""Generate paired SD3-family base/student images from identical CPU latents."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from reseed.diffusion.frequency_noise import (
    lowpass_match_sample_rms,
    pink_noise_match_sample_rms,
)

from reseed.config import SD35_MEDIUM_CONFIG_SHA256, SD35_MEDIUM_REVISION
from reseed.evaluation.generation_contract import (
    prepare_generation_request,
    sha256_file,
)
from reseed.models.sd3 import (
    load_sd3_pipeline,
    load_sd3_transformer,
    validate_sd3_pipeline_contract,
)
from reseed.models.sd3_residual_scale import load_sd3_residual_scale_adapter
from reseed.models.sd3_residual_bias import load_sd3_residual_bias_adapter
from reseed.paths import project_root


_OBJECTIVES = ("dmd", "tdm", "gad_nsp", "crra", "dp_dmd")


@dataclass(frozen=True)
class SD3EvaluationContract:
    mode_prefix: str
    backend: str
    model_family: str
    display_name: str
    evaluation_protocol: str
    base_inference_steps: int
    teacher_guidance_scale: float
    resolution: int
    max_sequence_length: int
    immutable_request: bool = True
    strict_protocol: bool = False
    expected_prompt_count: int = 5000
    source_model_id: str | None = None
    model_revision: str | None = None
    component_config_sha256: dict[str, str] | None = None
    prompt_csv_sha256: str | None = None
    protocol_relative_path: str | None = None
    protocol_sha256: str | None = None

    @property
    def base_mode(self) -> str:
        return f"{self.mode_prefix}-base"

    @property
    def student_modes(self) -> frozenset[str]:
        return frozenset(f"{self.mode_prefix}-{objective}" for objective in _OBJECTIVES)

    @property
    def modes(self) -> tuple[str, ...]:
        return (self.base_mode, *sorted(self.student_modes))


SD3M_EVALUATION_CONTRACT = SD3EvaluationContract(
    mode_prefix="sd3m",
    backend="sd3m-native",
    model_family="sd3_medium",
    display_name="SD3 Medium",
    evaluation_protocol="coco2017_val5k_sd3m_seeded_v1",
    base_inference_steps=28,
    teacher_guidance_scale=7.0,
    resolution=512,
    max_sequence_length=256,
    # Preserve the historical SD3M artifact behavior.  SD3.5 starts with the
    # stronger immutable request contract from its first result.
    immutable_request=False,
)

SD35M_EVALUATION_CONTRACT = SD3EvaluationContract(
    mode_prefix="sd35m",
    backend="sd35m-native",
    model_family="sd3_5_medium",
    display_name="SD3.5 Medium",
    evaluation_protocol="coco2017_val5k_sd35m_seeded_v1",
    base_inference_steps=40,
    teacher_guidance_scale=4.5,
    resolution=512,
    max_sequence_length=256,
    strict_protocol=True,
    source_model_id="stabilityai/stable-diffusion-3.5-medium",
    model_revision=SD35_MEDIUM_REVISION,
    component_config_sha256=SD35_MEDIUM_CONFIG_SHA256,
    prompt_csv_sha256="a4a1d16e09eefabf0ad237eed8e224d684996e85221797cbcf557167c795e7da",
    protocol_relative_path="configs/protocols/coco2017_val5k_sd35m_seeded_v1.json",
    protocol_sha256="5624d7c52871672bd57fa17e559f99784b060ecac584896533b0d44fca3fe911",
)

# Native-resolution SD3.5M protocol.  This is an additional, separately named
# and separately hashed immutable protocol, not a loosening of the 512 one:
# every sampler field (40/4 steps, CFG 4.5/1.0, shift 3, classic CFG without
# SLG, sequence length 256, batch size 1, gaussian latents) is unchanged and
# only the sampled resolution and the resulting latent shape differ.
SD35M_1024_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_EVALUATION_CONTRACT,
    evaluation_protocol="coco2017_val5k_1024_sd35m_seeded_v1",
    resolution=1024,
    protocol_relative_path="configs/protocols/coco2017_val5k_1024_sd35m_seeded_v1.json",
    protocol_sha256="b8c8309b2a63ccad91fedfde32b1ac4b375940ed0cfddc669242a596567152e2",
)

# Shared 100 prompt x 10 seed diversity bank at the same 1024 sampler contract.
# The COCO5K protocols fix exactly 5000 prompts, so the diversity bank needs its
# own named protocol rather than a mutable prompt count.
SD35M_1024_DIVERSITY_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_1024_EVALUATION_CONTRACT,
    evaluation_protocol="sd35m1024_diversity_100x10_v1",
    expected_prompt_count=1000,
    prompt_csv_sha256="1a50e75544f17972487a37154bba24c61b784005b03712cf0a2a1f6821f57360",
    protocol_relative_path="configs/protocols/sd35m1024_diversity_100x10_v1.json",
    protocol_sha256="99bca9255404b5e0cc6279e881935aa06b5aab4830cf721163d7f765c72ec107",
)

# Fixed qualitative bank used for the paper's matched-noise portrait panels.
# It keeps the native 1024 sampler contract and changes only the immutable
# prompt/seed bank and image count.
SD35M_1024_PORTRAIT_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_1024_EVALUATION_CONTRACT,
    evaluation_protocol="sd35m1024_portrait_3prompts_25seeds_v1",
    expected_prompt_count=75,
    prompt_csv_sha256="c0ea31aa56518f53d86a505b5f761bb168521b8442fa510c02ae3467e5315f52",
    protocol_relative_path="configs/protocols/sd35m1024_portrait_3prompts_25seeds_v1.json",
    protocol_sha256="81a41d17e525740ff08b72d5fe19c80529d5c54ca04a4eb2a253ee516cbe4ca1",
)

# The same 75 portrait cases, seeds and latents, with the reference teacher
# sampling 60 steps instead of 40; students stay pinned at four steps.
SD35M_1024_PORTRAIT_TEACHER60_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_1024_PORTRAIT_EVALUATION_CONTRACT,
    evaluation_protocol="sd35m1024_portrait_3prompts_25seeds_teacher60_v1",
    base_inference_steps=60,
    protocol_relative_path="configs/protocols/sd35m1024_portrait_3prompts_25seeds_teacher60_v1.json",
    protocol_sha256="8fabcf4516b598b4b85ed2ea0b9b217fa0faa2bf6c2e3926f0fb2e9ea22cdfac",
)

# Deterministic nested COCO1K protocol used for checkpoint-wise ablations.  It
# remains strict: the exact first-1K prompt CSV and every sampler field are
# pinned independently from the full COCO5K protocol.
SD35M_1024_COCO1K_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_1024_EVALUATION_CONTRACT,
    evaluation_protocol="coco2017_val1k_1024_sd35m_seeded_v1",
    expected_prompt_count=1000,
    prompt_csv_sha256="1a45411ed112e737ec2950d327b7c0a0f6fc1360b27ec8aed08986ca34f563ed",
    protocol_relative_path="configs/protocols/coco2017_val1k_1024_sd35m_seeded_v1.json",
    protocol_sha256="1dc50b84b8333577d236577ad4860ad0193dc229d72e68aeccae8dd039fe8db7",
)

# Official GenEval prompt bank (553 prompts x 4 samples = 2,212 cases) at the
# same 1024 sampler contract.  The cases CSV is byte-identical to the one the
# SD3-M GenEval run used, so the two model families are scored on the same
# prompts, the same case order and the same seeds; only the sampler contract
# differs.  Like the diversity bank it needs its own named protocol because the
# COCO5K protocols fix exactly 5000 prompts.
SD35M_1024_GENEVAL_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_1024_EVALUATION_CONTRACT,
    evaluation_protocol="geneval_553x4_sd35m1024_v1",
    expected_prompt_count=2212,
    prompt_csv_sha256="791e718c15f4ee13269c5ee678fda8b49b8f18a435531488ddb9421737c04adf",
    protocol_relative_path="configs/protocols/geneval_553x4_sd35m1024_v1.json",
    protocol_sha256="8a0edf538d09c2ab4608efc24f1119f6ef3ce81fde89db1e1f00fd2a4898e30a",
)

# The same official GenEval prompts with ten samples each (5,530 cases).  One
# generation pass then answers both questions the paper asks of a prompt bank:
# the mmdetection scorer and the same-prompt diversity scorers read the same ten
# images.  Its seeds are the GenEval recipe at ten samples, so they are not
# the 553x4 bank's seeds and the two GenEval numbers are not byte-comparable.
SD35M_1024_GENEVAL_10SAMPLE_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_1024_EVALUATION_CONTRACT,
    evaluation_protocol="geneval_553x10_sd35m1024_v1",
    expected_prompt_count=5530,
    prompt_csv_sha256="6d70eace8dbcbd071e0926dd82215f7a2d4b45abd84ded19ad8a830d7e9c4aed",
    protocol_relative_path="configs/protocols/geneval_553x10_sd35m1024_v1.json",
    protocol_sha256="2d2e1a770790825e3cfa02dba3f3b5b69c0faa5bfa2644c1c1985b577831ff76",
)

# The full published PartiPrompts table, one image per prompt, as a quality
# bank.  FID has no meaning here because PartiPrompts ships no reference images.
SD35M_1024_PARTIPROMPTS_QUALITY_CONTRACT = dataclasses.replace(
    SD35M_1024_EVALUATION_CONTRACT,
    evaluation_protocol="partiprompts_quality_1632x1_sd35m1024_v1",
    expected_prompt_count=1632,
    prompt_csv_sha256="5f9c47cc007e9a33978cc6a3e50cc9d086277b485b1221882ee3e805c87ca3a4",
    protocol_relative_path="configs/protocols/partiprompts_quality_1632x1_sd35m1024_v1.json",
    protocol_sha256="4f31bf2df532c014f7be01080a277be6b13b8e32dd423c37093a88378143c9db",
)

# Six scenic prompts x six samples, the matched-noise bank behind the
# qualitative five-system comparison figures. It carries no metrics: the five
# systems draw the same 36 initial latents so the figures can be read cell by
# cell.
SD35M_1024_SCENIC_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_1024_EVALUATION_CONTRACT,
    evaluation_protocol="sd35m1024_scenic_6prompts_6seeds_v1",
    expected_prompt_count=36,
    prompt_csv_sha256="ecac0aa671d5e244e9ce9bc7b07adaba1ad23d87e7fba65c172aa51c109ec802",
    protocol_relative_path="configs/protocols/sd35m1024_scenic_6prompts_6seeds_v1.json",
    protocol_sha256="c67da1c4497608ac41b0049ae7cebd15d8ca202b25c21f8b4347b1e4133737e2",
)

# The same qualitative bank shape for pets against cluttered backgrounds, where
# a collapsed student repeats one arrangement of the clutter.
SD35M_1024_PETS_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_1024_EVALUATION_CONTRACT,
    evaluation_protocol="sd35m1024_pets_6prompts_6seeds_v1",
    expected_prompt_count=36,
    prompt_csv_sha256="fb9725b3e68e1eb8a6e96913a6d000965286d868710bef2ed52d44652daf0e8d",
    protocol_relative_path="configs/protocols/sd35m1024_pets_6prompts_6seeds_v1.json",
    protocol_sha256="48cd6490168c6b72fc0112f2aef0da32bb7d6b73f65028823482125dcde9eeb2",
)

# Prompt 86 of the shared diversity bank at its own six seeds: the cases behind
# the paper's portrait comparison. The bank's images were pruned to a visual
# sample after scoring and this prompt was not among the ones kept, so the
# figure is rebuilt by regenerating these six cases, which are deterministic.
SD35M_1024_PORTRAIT086_EVALUATION_CONTRACT = dataclasses.replace(
    SD35M_1024_EVALUATION_CONTRACT,
    evaluation_protocol="sd35m1024_portrait086_1prompt_6seeds_v1",
    expected_prompt_count=6,
    prompt_csv_sha256="80f83b721e55e9694cf68c5537812274208a233aae571f2f306735c15c10fe28",
    protocol_relative_path="configs/protocols/sd35m1024_portrait086_1prompt_6seeds_v1.json",
    protocol_sha256="384fe2e7bbaff0a92ab7fc4909adbedf93f749dbf9de6d03f20bc1d30ce2679f",
)

# Every strict SD3.5M protocol this entry point will accept.  An unknown
# --evaluation-protocol keeps the default contract and therefore still fails the
# immutable protocol-name check below.
SD35M_PROTOCOL_CONTRACTS: dict[str, SD3EvaluationContract] = {
    contract.evaluation_protocol: contract
    for contract in (
        SD35M_EVALUATION_CONTRACT,
        SD35M_1024_EVALUATION_CONTRACT,
        SD35M_1024_COCO1K_EVALUATION_CONTRACT,
        SD35M_1024_DIVERSITY_EVALUATION_CONTRACT,
        SD35M_1024_PORTRAIT_EVALUATION_CONTRACT,
        SD35M_1024_PORTRAIT_TEACHER60_EVALUATION_CONTRACT,
        SD35M_1024_GENEVAL_EVALUATION_CONTRACT,
        SD35M_1024_GENEVAL_10SAMPLE_EVALUATION_CONTRACT,
        SD35M_1024_PARTIPROMPTS_QUALITY_CONTRACT,
        SD35M_1024_SCENIC_EVALUATION_CONTRACT,
        SD35M_1024_PETS_EVALUATION_CONTRACT,
        SD35M_1024_PORTRAIT086_EVALUATION_CONTRACT,
    )
}


def resolve_sd3_evaluation_contract(
    contract: SD3EvaluationContract, protocol_name: str
) -> SD3EvaluationContract:
    """Select the named strict SD3.5M protocol variant, if there is one."""

    if contract.model_family != "sd3_5_medium":
        return contract
    return SD35M_PROTOCOL_CONTRACTS.get(protocol_name, contract)


# Backward-compatible import used by existing callers/tests.
_STUDENT_MODES = set(SD3M_EVALUATION_CONTRACT.student_modes)


def _records(path: Path, expected_count: int | None) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"expected {expected_count} prompts, found {len(records)}")
    required = {"case_number", "evaluation_seed", "prompt"}
    missing = required - set(records[0] if records else {})
    if missing:
        raise ValueError(f"prompt CSV is missing columns: {sorted(missing)}")
    for record in records:
        record["case_number"] = int(record["case_number"])
        record["evaluation_seed"] = int(record["evaluation_seed"])
        if "coco_id" in record and record["coco_id"]:
            record["coco_id"] = int(record["coco_id"])
    case_numbers = [record["case_number"] for record in records]
    if len(case_numbers) != len(set(case_numbers)):
        raise ValueError("case_number values must be unique")
    return records


def _image_path(root: Path, record: dict[str, Any]) -> Path:
    return root / "images" / (
        f"case_{record['case_number']:05d}_seed_{record['evaluation_seed']}.png"
    )


def _image_identity_path(path: Path) -> Path:
    """Legacy per-image sidecar. Written by runs before the append-only log."""
    return path.with_suffix(path.suffix + ".identity.json")


def _identity_log_path(root: Path, shard_id: int = 0, num_shards: int = 1) -> Path:
    """One append-only identity log per shard, replacing per-image sidecars.

    A 5,000-image COCO5K directory used to hold 5,000 sidecars, and the image
    pruner deleted only the PNGs, so 90 per-checkpoint evaluations left roughly
    half a million files behind and exhausted the project's file-count quota
    mid-run. The log carries exactly the same records in one file per shard.
    """
    suffix = "" if num_shards == 1 else f".shard_{shard_id:05d}_of_{num_shards:05d}"
    return root / f"image_identity{suffix}.jsonl"


def _load_identity_log(root: Path) -> dict[str, dict[str, Any]]:
    """Every identity record on disk, keyed by image file name.

    Reads every shard's log, because a merge step runs after all shards and a
    resumed shard may have been split differently. A crash can leave the last
    line half-written; that line is dropped and its image is simply
    regenerated, which is what a missing sidecar did before.
    """
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("image_identity*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn trailing write
            name = item.get("image_path")
            if isinstance(name, str):
                records[name] = item
    return records


def _identity_record(
    path: Path, identities: dict[str, dict[str, Any]] | None
) -> dict[str, Any] | None:
    """The identity for one image: from the log, else the legacy sidecar.

    The fallback is what keeps evaluation directories written before the log
    resumable and mergeable.
    """
    if identities is not None:
        record = identities.get(path.name)
        if record is not None:
            return record
    sidecar = _image_identity_path(path)
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _valid_image(
    path: Path,
    *,
    expected_latent_sha256: str | None = None,
    expected_request_sha256: str | None = None,
    require_identity: bool = False,
    identities: dict[str, dict[str, Any]] | None = None,
) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        if not require_identity:
            return True
        identity = _identity_record(path, identities)
        if identity is None:
            return False
        with Image.open(path) as image:
            width, height = image.size
        return (
            identity.get("schema_version") == 1
            and identity.get("image_sha256") == sha256_file(path)
            and identity.get("width") == width
            and identity.get("height") == height
            and identity.get("latent_sha256") == expected_latent_sha256
            and identity.get("generation_request_sha256") == expected_request_sha256
        )
    except Exception:
        return False


def _noise(
    record: dict[str, Any],
    shape: tuple[int, int, int],
    *,
    initial_noise_mode: str = "gaussian",
    lowpass_cutoff: float = 0.20,
    pink_beta: float = 0.3,
    constant_magnitude: float = 10.0,
) -> tuple[torch.Tensor, str]:
    generator = torch.Generator(device="cpu").manual_seed(record["evaluation_seed"])
    if initial_noise_mode in ("gaussian", "lowpass", "pink", "antithetic"):
        latent = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
    elif initial_noise_mode == "zeros":
        latent = torch.zeros(shape, device="cpu", dtype=torch.float32)
    elif initial_noise_mode == "ones":
        latent = torch.ones(shape, device="cpu", dtype=torch.float32)
    elif initial_noise_mode == "constant_positive":
        latent = torch.full(shape, constant_magnitude, device="cpu", dtype=torch.float32)
    elif initial_noise_mode == "constant_negative":
        latent = torch.full(shape, -constant_magnitude, device="cpu", dtype=torch.float32)
    else:
        raise ValueError(f"unsupported initial noise mode: {initial_noise_mode}")
    if initial_noise_mode == "lowpass":
        # Match SD3M training: the rollout noise enters the frequency helper in
        # BF16, while FFT/filtering and RMS restoration are internally FP32.
        latent = latent.to(torch.bfloat16)
        latent, _ = lowpass_match_sample_rms(latent.unsqueeze(0), cutoff=lowpass_cutoff)
        latent = latent.squeeze(0).contiguous()
    elif initial_noise_mode == "pink":
        latent = latent.to(torch.bfloat16)
        latent, _ = pink_noise_match_sample_rms(latent.unsqueeze(0), beta=pink_beta)
        latent = latent.squeeze(0).contiguous()
    elif initial_noise_mode == "antithetic":
        # The same evaluation seed therefore gives the exact pair z and -z.
        latent = -latent
    digest_tensor = latent.view(torch.int16) if latent.dtype == torch.bfloat16 else latent
    digest = hashlib.sha256(digest_tensor.contiguous().numpy().tobytes()).hexdigest()
    return latent, digest


def apply_residual_bias_interval(
    adapter: Any, interval_index: int, enabled_intervals: Sequence[int]
) -> None:
    """Gate a residual-bias adapter for the interval about to be sampled.

    ``enabled_intervals`` is the configured interval set; the adapter is active
    on exactly those student intervals.  The default set ``[0]`` therefore keeps
    first-interval-only inference bitwise unchanged.
    """

    if adapter is None:
        return
    if interval_index in enabled_intervals:
        adapter.enable()
    else:
        adapter.disable()


def _unshifted_inference_sigmas(steps: int) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    return [1.0 - index / steps for index in range(steps)]


def _piecewise_cfg_schedule(
    inference_steps: int,
    start_fraction: float,
    late_guidance_scale: float,
) -> tuple[list[float], int]:
    """Return exact per-step CFG values and the first guided step index."""

    if inference_steps <= 0:
        raise ValueError("inference_steps must be positive")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("teacher CFG start fraction must be in [0, 1)")
    if start_fraction == 0.0:
        return [late_guidance_scale] * inference_steps, 0
    switch_step = int(round(inference_steps * start_fraction))
    switch_step = min(max(switch_step, 1), inference_steps - 1)
    return (
        [
            1.0 if step_index < switch_step else late_guidance_scale
            for step_index in range(inference_steps)
        ],
        switch_step,
    )


def _threshold_cfg_schedule(
    timesteps: torch.Tensor,
    threshold_a: float,
    late_guidance_scale: float,
    *,
    num_train_timesteps: int,
) -> tuple[list[float], int]:
    """Enable CFG once the descending normalized FlowMatch time reaches ``a``."""

    if not 0.0 < threshold_a < 1.0:
        raise ValueError("teacher CFG threshold a must be in (0, 1)")
    normalized = [float(value) / float(num_train_timesteps) for value in timesteps]
    schedule = [
        1.0 if timestep > threshold_a else late_guidance_scale
        for timestep in normalized
    ]
    guided = [index for index, scale in enumerate(schedule) if scale != 1.0]
    if not guided:
        raise ValueError("teacher CFG threshold never enables guidance on this schedule")
    return schedule, guided[0]


def _symmetric_cfg_schedule(
    timesteps: torch.Tensor,
    mean_guidance_scale: float,
    span: float,
) -> tuple[list[float], float]:
    """Return the timestep-normalized symmetric TV-CFG schedule.

    The unnormalized scale rises linearly from ``omega - s`` to
    ``omega + s`` and then falls symmetrically. Following the official TV-CFG
    implementation, ``A`` is chosen so that the descending-timestep weighted
    mean remains ``omega``. For the paper/default setting ``s=omega-1``, the
    raw endpoints are therefore one and ``2*omega-1``.
    """

    from reseed.diffusion.tvcfg import symmetric_tvcfg_schedule

    schedule, normalization = symmetric_tvcfg_schedule(
        timesteps, mean_guidance_scale, span
    )
    return [float(value) for value in schedule], normalization


def _load_sd3_pipeline_with_forced_cfg(model: str, *, torch_dtype: torch.dtype) -> Any:
    """Load a pipeline that always prepares both CFG embedding branches.

    Scheduled diagnostics need both branches even at CFG=1. Constant
    subunit-CFG teachers also need them: the stock pipeline's CFG>1 switch
    would otherwise silently replace u + w * (c - u) with c.
    """

    from diffusers import StableDiffusion3Pipeline

    class _ForcedCFGStableDiffusion3Pipeline(StableDiffusion3Pipeline):
        @property
        def do_classifier_free_guidance(self) -> bool:
            return True

    return _ForcedCFGStableDiffusion3Pipeline.from_pretrained(
        str(model), torch_dtype=torch_dtype
    )


def _requires_forced_cfg(args: argparse.Namespace, contract: SD3EvaluationContract) -> bool:
    """Preserve native student/CFG=1 paths while enabling true subunit CFG."""

    if args.mode != contract.base_mode:
        return False
    return (
        0.0 < args.teacher_guidance_scale < 1.0
        or float(getattr(args, "teacher_cfg_start_fraction", 0.0)) > 0.0
        or getattr(args, "teacher_cfg_threshold_a", None) is not None
        or getattr(args, "teacher_cfg_symmetric_span", None) is not None
    )


def _load_sd3_evaluation_pipeline(
    args: argparse.Namespace,
    contract: SD3EvaluationContract,
    *,
    torch_dtype: torch.dtype,
) -> Any:
    loader = (
        _load_sd3_pipeline_with_forced_cfg
        if _requires_forced_cfg(args, contract)
        else load_sd3_pipeline
    )
    return loader(args.model, torch_dtype=torch_dtype)


def _protocol_identity(contract: SD3EvaluationContract) -> dict[str, Any] | None:
    if contract.protocol_relative_path is None:
        return None
    path = project_root() / contract.protocol_relative_path
    if not path.is_file():
        raise ValueError(f"missing evaluation protocol contract: {path}")
    digest = sha256_file(path)
    if contract.protocol_sha256 is not None and digest != contract.protocol_sha256:
        raise ValueError(
            f"evaluation protocol SHA256 {digest} does not match {contract.protocol_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("name") != contract.evaluation_protocol:
        raise ValueError(
            f"evaluation protocol file name={payload.get('name')!r}; "
            f"expected {contract.evaluation_protocol!r}"
        )
    return {
        "path": contract.protocol_relative_path,
        "sha256": digest,
    }


def _runtime_versions() -> dict[str, Any]:
    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "diffusers": package_version("diffusers"),
        "transformers": package_version("transformers"),
        "pillow": package_version("Pillow"),
    }


def _implementation_identity(contract: SD3EvaluationContract) -> list[dict[str, str]]:
    root = project_root()
    relatives = [
        "src/reseed/evaluation/sd3m_generate.py",
        "src/reseed/evaluation/generation_contract.py",
        "src/reseed/diffusion/frequency_noise.py",
        "src/reseed/models/sd3.py",
        "src/reseed/models/sd3_contract.py",
    ]
    if contract.model_family == "sd3_5_medium":
        relatives.append("src/reseed/evaluation/sd35m_generate.py")
    identities = []
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing evaluation implementation file: {path}")
        identities.append({"path": relative, "sha256": sha256_file(path)})
    return identities


def _latent_bank_identity(
    records: list[dict[str, Any]],
    shape: tuple[int, int, int],
    *,
    initial_noise_mode: str = "gaussian",
    lowpass_cutoff: float = 0.20,
    pink_beta: float = 0.3,
    constant_magnitude: float = 10.0,
) -> tuple[dict[int, str], str]:
    hashes: dict[int, str] = {}
    aggregate = hashlib.sha256()
    for record in records:
        case_number = int(record["case_number"])
        _, digest = _noise(
            record,
            shape,
            initial_noise_mode=initial_noise_mode,
            lowpass_cutoff=lowpass_cutoff,
            pink_beta=pink_beta,
            constant_magnitude=constant_magnitude,
        )
        hashes[case_number] = digest
        aggregate.update(str(case_number).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
    return hashes, aggregate.hexdigest()


def _checkpoint_student_identity(checkpoint: Path) -> dict[str, Any]:
    student = checkpoint / "student"
    if not student.is_dir() and (checkpoint / "transformer").is_dir():
        # Externally published students (see tools/wrap_external_sd3m_student.py)
        # ship a bare ``transformer/`` directory.  Locally trained checkpoints keep
        # the ``student/`` layout, so their identity payload is unchanged.
        student = checkpoint / "transformer"
    files = sorted(student.glob("*.safetensors"))
    if not files:
        raise ValueError(f"checkpoint contains no student safetensors: {student}")
    config = student / "config.json"
    if not config.is_file():
        raise ValueError(f"checkpoint contains no student config: {config}")
    indexes = sorted(student.glob("*.index.json"))
    return {
        "config": {
            "path": str(config.relative_to(checkpoint)),
            "size": config.stat().st_size,
            "sha256": sha256_file(config),
        },
        "weights": [
            {
                "path": str(path.relative_to(checkpoint)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "indexes": [
            {
                "path": str(path.relative_to(checkpoint)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in indexes
        ],
    }


def _generation_request(
    args: argparse.Namespace,
    contract: SD3EvaluationContract,
    latent_shape: tuple[int, int, int],
    checkpoint_manifest: dict[str, Any] | None,
    model_runtime_contract: dict[str, Any] | None,
    protocol_identity: dict[str, Any] | None,
    latent_bank_sha256: str,
) -> dict[str, Any]:
    initial_noise_mode = getattr(args, "initial_noise_mode", "gaussian")
    lowpass_cutoff = getattr(args, "initial_noise_lowpass_cutoff", 0.20)
    pink_beta = getattr(args, "initial_noise_pink_beta", 0.3)
    constant_magnitude = getattr(args, "initial_noise_constant_magnitude", 10.0)
    checkpoint_root = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    model_root = Path(args.model).expanduser().resolve()
    config_identities = {}
    for name, relative in {
        "model_index": "model_index.json",
        "scheduler": "scheduler/scheduler_config.json",
        "transformer": "transformer/config.json",
        "vae": "vae/config.json",
    }.items():
        path = model_root / relative
        config_identities[name] = (
            {"path": str(path), "sha256": sha256_file(path)} if path.is_file() else None
        )
    return {
        "schema_version": 1,
        "evaluation_protocol": args.evaluation_protocol,
        "evaluation_protocol_contract": protocol_identity,
        "mode": args.mode,
        "backend": contract.backend if args.mode in contract.student_modes else None,
        "model_family": contract.model_family,
        "model": str(model_root),
        "model_configs": config_identities,
        "model_runtime_contract": model_runtime_contract,
        "checkpoint": str(checkpoint_root) if checkpoint_root else None,
        "checkpoint_manifest_sha256": (
            sha256_file(checkpoint_root / "manifest.json") if checkpoint_root else None
        ),
        "checkpoint_config_sha256": (
            checkpoint_manifest.get("config_sha256") if checkpoint_manifest else None
        ),
        "student": (
            _checkpoint_student_identity(checkpoint_root) if checkpoint_root else None
        ),
        "prompt_csv": str(args.prompts.resolve()),
        "prompt_csv_sha256": sha256_file(args.prompts),
        "num_prompts": args.expected_count,
        "inference_steps": args.inference_steps,
        "teacher_guidance_scale": args.teacher_guidance_scale,
        "teacher_cfg_symmetric_span": getattr(args, "teacher_cfg_symmetric_span", None),
        "student_guidance_scale": args.student_guidance_scale,
        "guidance_mode": "classic_cfg_no_slg",
        "negative_prompt": "",
        "resolution": args.resolution,
        "max_sequence_length": args.max_sequence_length,
        "latent_shape": list(latent_shape),
        "latent_source": (
            "per-case CPU torch.Generator; float32; evaluation_seed; "
            f"initial_noise_mode={initial_noise_mode}"
        ),
        "initial_noise_mode": initial_noise_mode,
        "initial_noise_lowpass_cutoff": (
            lowpass_cutoff if initial_noise_mode == "lowpass" else None
        ),
        "initial_noise_pink_beta": pink_beta if initial_noise_mode == "pink" else None,
        "initial_noise_constant_magnitude": (
            constant_magnitude
            if initial_noise_mode in ("constant_positive", "constant_negative")
            else None
        ),
        "latent_bank_sha256": latent_bank_sha256,
        "batch_size": args.batch_size,
        "weight_dtype": "bfloat16",
        "latent_cast_dtype": "bfloat16",
        "vae_decode_dtype": "bfloat16",
        "runtime_versions": _runtime_versions(),
        "implementation": _implementation_identity(contract),
    }


def _protocol(
    args: argparse.Namespace,
    pipe: Any,
    latent_shape: tuple[int, int, int],
    contract: SD3EvaluationContract = SD3M_EVALUATION_CONTRACT,
    checkpoint_manifest: dict[str, Any] | None = None,
    model_runtime_contract: dict[str, Any] | None = None,
    request_path: Path | None = None,
    request_sha256: str | None = None,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    initial_noise_mode = getattr(args, "initial_noise_mode", "gaussian")
    lowpass_cutoff = getattr(args, "initial_noise_lowpass_cutoff", 0.20)
    pink_beta = getattr(args, "initial_noise_pink_beta", 0.3)
    constant_magnitude = getattr(args, "initial_noise_constant_magnitude", 10.0)
    aligned_base_sigmas = (
        _unshifted_inference_sigmas(args.inference_steps)
        if args.mode in contract.student_modes
        else None
    )
    shift = float(pipe.scheduler.config.shift)
    effective_guidance = (
        args.teacher_guidance_scale if args.mode == contract.base_mode else args.student_guidance_scale
    )
    aligned_shifted_boundaries = (
        [shift * value / (1.0 + (shift - 1.0) * value) for value in [*aligned_base_sigmas, 0.0]]
        if aligned_base_sigmas is not None
        else None
    )
    cfg_fraction = float(getattr(args, "teacher_cfg_start_fraction", 0.0))
    cfg_threshold_a = getattr(args, "teacher_cfg_threshold_a", None)
    cfg_symmetric_span = getattr(args, "teacher_cfg_symmetric_span", None)
    protocol = {
        "schema_version": 1,
        "evaluation_protocol": args.evaluation_protocol,
        "model_family": contract.model_family,
        "mode": args.mode,
        "model": str(Path(args.model).expanduser()),
        "checkpoint": str(Path(args.checkpoint).expanduser()) if args.checkpoint else None,
        "checkpoint_objective": checkpoint_manifest.get("objective") if checkpoint_manifest else None,
        "checkpoint_config_sha256": checkpoint_manifest.get("config_sha256") if checkpoint_manifest else None,
        "prediction_type": "flow",
        "scheduler": pipe.scheduler.__class__.__name__,
        "scheduler_shift": float(pipe.scheduler.config.shift),
        "num_inference_steps": args.inference_steps,
        "explicit_unshifted_sigmas": aligned_base_sigmas,
        "effective_shifted_boundaries": aligned_shifted_boundaries,
        "guidance_scale": effective_guidance,
        "teacher_guidance_scale": args.teacher_guidance_scale,
        "teacher_cfg_start_fraction": cfg_fraction,
        "teacher_cfg_threshold_a": cfg_threshold_a,
        "teacher_cfg_symmetric_span": cfg_symmetric_span,
        "teacher_cfg_switch_step": (
            int(round(args.inference_steps * cfg_fraction))
            if cfg_fraction > 0
            else 0
        ),
        "teacher_cfg_schedule": (
            "symmetric_tvcfg_timestep_normalized"
            if cfg_symmetric_span is not None
            else "normalized_time_threshold_cfg1_then_configured"
            if cfg_threshold_a is not None
            else "step_fraction_cfg1_then_configured"
            if cfg_fraction > 0
            else "constant"
        ),
        "student_guidance_scale": args.student_guidance_scale,
        "student_forwards_per_step": 1 if args.mode in contract.student_modes else None,
        "resolution": args.resolution,
        "latent_shape": list(latent_shape),
        "latent_source": (
            "per-case CPU torch.Generator seeded by evaluation_seed; float32 hash before cast; "
            f"initial_noise_mode={initial_noise_mode}"
        ),
        "initial_noise_mode": initial_noise_mode,
        "initial_noise_lowpass_cutoff": (
            lowpass_cutoff if initial_noise_mode == "lowpass" else None
        ),
        "initial_noise_pink_beta": pink_beta if initial_noise_mode == "pink" else None,
        "initial_noise_constant_magnitude": (
            constant_magnitude
            if initial_noise_mode in ("constant_positive", "constant_negative")
            else None
        ),
        "prompt_csv": str(args.prompts.resolve()),
        "prompt_csv_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "num_prompts": args.expected_count,
    }
    if contract.immutable_request:
        protocol.update(
            {
                "backend": contract.backend if args.mode in contract.student_modes else None,
                "model_runtime_contract": model_runtime_contract,
                "guidance_mode": "classic_cfg_no_slg",
                "negative_prompt": "",
                "generation_request": str(request_path) if request_path else None,
                "generation_request_sha256": request_sha256,
                "weight_dtype": "bfloat16",
                "latent_cast_dtype": "bfloat16",
                "vae_decode_dtype": "bfloat16",
            }
        )
    if request_payload is not None:
        # Keep the generic scorer's request/protocol identity comparison exact.
        for key in (
            "model",
            "checkpoint",
            "student",
            "latent_bank_sha256",
        ):
            protocol[key] = request_payload.get(key)
        protocol["evaluation_protocol_contract"] = request_payload.get(
            "evaluation_protocol_contract"
        )
        protocol["runtime_versions"] = request_payload.get("runtime_versions")
        protocol["implementation"] = request_payload.get("implementation")
    if (
        contract.mode_prefix == "sd3m"
        and args.mode == contract.base_mode
        and 0.0 < effective_guidance <= 1.0
        and cfg_fraction == 0.0
        and cfg_threshold_a is None
        and cfg_symmetric_span is None
    ):
        # Version the newly supported diagnostic semantics without changing
        # historical student, >1 teacher, or immutable SD3.5 protocols.
        forced_cfg = _requires_forced_cfg(args, contract)
        protocol.update(
            {
                "teacher_cfg_actual_constant_scale": effective_guidance,
                "teacher_cfg_forced_branches": forced_cfg,
                "teacher_cfg_backend": (
                    "diffusers_forced_cfg" if forced_cfg else "diffusers_conditional_only"
                ),
            }
        )
    return protocol


def _write_outputs(
    root: Path,
    records: list[dict[str, Any]],
    latent_hashes: dict[int, str],
    protocol: dict[str, Any],
    contract: SD3EvaluationContract = SD3M_EVALUATION_CONTRACT,
    *,
    shard_id: int = 0,
    num_shards: int = 1,
) -> None:
    suffix = "" if num_shards == 1 else f".shard_{shard_id:05d}_of_{num_shards:05d}"
    manifest = root / f"generation_manifest{suffix}.jsonl"
    identities = _load_identity_log(root) if contract.immutable_request else None
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            item = dict(record)
            image_path = _image_path(root, record)
            item.update(
                {
                    "image_path": str(image_path),
                    "latent_sha256": latent_hashes[record["case_number"]],
                    "mode": protocol["mode"],
                    "model_family": contract.model_family,
                }
            )
            if contract.immutable_request:
                identity = _identity_record(image_path, identities)
                if identity is None:
                    raise ValueError(
                        f"missing image identity for {image_path}: neither "
                        f"{_identity_log_path(root, shard_id, num_shards).name} nor a "
                        "legacy .identity.json sidecar carries it"
                    )
                with Image.open(image_path) as opened:
                    actual_size = opened.size
                if (
                    identity.get("image_sha256") != sha256_file(image_path)
                    or identity.get("latent_sha256") != latent_hashes[record["case_number"]]
                    or identity.get("generation_request_sha256")
                    != protocol.get("generation_request_sha256")
                    or actual_size != (identity.get("width"), identity.get("height"))
                ):
                    raise ValueError(f"image identity mismatch: {image_path}")
                item.update(
                    {
                        "image_sha256": identity["image_sha256"],
                        "image_width": identity["width"],
                        "image_height": identity["height"],
                    }
                )
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (root / f"generation_protocol{suffix}.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if num_shards > 1:
        (root / f"generation_complete{suffix}").write_text("complete\n", encoding="utf-8")


def _shard_suffix(shard_id: int, num_shards: int) -> str:
    return f".shard_{shard_id:05d}_of_{num_shards:05d}"


def _finalize_shards(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    contract: SD3EvaluationContract,
) -> None:
    """Validate disjoint completed shards and materialize one canonical manifest."""

    expected = {int(record["case_number"]): record for record in records}
    combined: dict[int, dict[str, Any]] = {}
    canonical_protocol: dict[str, Any] | None = None
    ignored_protocol_keys = {"generation_shard"}

    for shard_id in range(args.num_shards):
        suffix = _shard_suffix(shard_id, args.num_shards)
        manifest_path = args.output / f"generation_manifest{suffix}.jsonl"
        protocol_path = args.output / f"generation_protocol{suffix}.json"
        complete_path = args.output / f"generation_complete{suffix}"
        for path in (manifest_path, protocol_path, complete_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing completed generation shard artifact: {path}")

        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("generation_shard") != {
            "shard_id": shard_id,
            "num_shards": args.num_shards,
            "assignment": "record_index_mod_num_shards",
        }:
            raise ValueError(f"invalid generation shard contract: {protocol_path}")
        comparable = {
            key: value for key, value in protocol.items() if key not in ignored_protocol_keys
        }
        if canonical_protocol is None:
            canonical_protocol = comparable
        elif comparable != canonical_protocol:
            raise ValueError(f"generation protocol differs across shards: {protocol_path}")

        with manifest_path.open(encoding="utf-8") as handle:
            shard_records = [json.loads(line) for line in handle if line.strip()]
        expected_shard_count = len(records[shard_id :: args.num_shards])
        if len(shard_records) != expected_shard_count:
            raise ValueError(
                f"shard {shard_id} has {len(shard_records)} records, "
                f"expected {expected_shard_count}"
            )
        for item in shard_records:
            case_number = int(item["case_number"])
            if case_number in combined or case_number not in expected:
                raise ValueError(f"duplicate or unexpected case_number={case_number}")
            source = expected[case_number]
            for key in ("evaluation_seed", "prompt"):
                if item.get(key) != source.get(key):
                    raise ValueError(f"shard record {case_number} differs on {key}")
            image_path = Path(item["image_path"])
            if not image_path.is_file() or not _valid_image(image_path):
                raise ValueError(f"missing or invalid generated image: {image_path}")
            combined[case_number] = item

    if set(combined) != set(expected):
        missing = sorted(set(expected) - set(combined))
        raise ValueError(f"incomplete merged shard set; missing case numbers: {missing[:10]}")
    if canonical_protocol is None:
        raise ValueError("no shard protocols were loaded")

    manifest = args.output / "generation_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(combined[int(record["case_number"])], ensure_ascii=False) + "\n")
    (args.output / "generation_protocol.json").write_text(
        json.dumps(canonical_protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _save_image_with_identity(
    image: Image.Image,
    path: Path,
    *,
    latent_sha256: str,
    generation_request_sha256: str | None = None,
    write_identity: bool,
    identity_log: Path | None = None,
) -> None:
    """Save the image, then append its identity to the shard log.

    The PNG is written first and the record second, so a crash in between
    leaves an image with no record, which resume treats as absent and
    regenerates. Each line is flushed and fsynced, because the log is the only
    thing standing between a killed job and a full regeneration.
    """
    image.save(path)
    if not write_identity or identity_log is None:
        return
    width, height = image.size
    record = {
        "schema_version": 1,
        "image_path": path.name,
        "image_sha256": sha256_file(path),
        "width": width,
        "height": height,
        "latent_sha256": latent_sha256,
        "generation_request_sha256": generation_request_sha256,
    }
    identity_log.parent.mkdir(parents=True, exist_ok=True)
    with identity_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parser(contract: SD3EvaluationContract = SD3M_EVALUATION_CONTRACT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=contract.modes, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--first-interval-lora-checkpoint",
        type=Path,
        help=(
            "Optional PEFT/Diffusers LoRA checkpoint enabled for student interval "
            "zero only and disabled for every suffix interval."
        ),
    )
    parser.add_argument(
        "--first-interval-transformer-checkpoint",
        type=Path,
        help=(
            "Optional complete transformer used for student interval zero only; "
            "the transformer from --checkpoint is restored for suffix intervals."
        ),
    )
    parser.add_argument(
        "--first-interval-residual-scale-checkpoint",
        type=Path,
        help=(
            "Optional block-residual scalar checkpoint enabled for student interval "
            "zero only and disabled for every suffix interval."
        ),
    )
    parser.add_argument(
        "--first-interval-residual-bias-checkpoint",
        type=Path,
        help=(
            "Optional block-output channel-bias checkpoint enabled for student "
            "interval zero only and disabled for every suffix interval."
        ),
    )
    parser.add_argument(
        "--first-interval-residual-bias-intervals",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Student interval indices on which the residual-bias adapter is enabled. "
            "The default '0' reproduces first-interval-only inference exactly."
        ),
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluation-protocol", default=contract.evaluation_protocol)
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--teacher-guidance-scale", type=float, default=contract.teacher_guidance_scale)
    parser.add_argument(
        "--teacher-cfg-start-fraction",
        type=float,
        default=0.0,
        help=(
            "Inference-only base-model diagnostic: use CFG=1 before this fraction "
            "of sampling steps, then --teacher-guidance-scale. Zero preserves "
            "the historical constant-CFG behavior."
        ),
    )
    parser.add_argument(
        "--teacher-cfg-threshold-a",
        type=float,
        help=(
            "Inference-only base-model diagnostic: while normalized descending "
            "FlowMatch timestep t > a use CFG=1; when t <= a use "
            "--teacher-guidance-scale. Mutually exclusive with other CFG schedules."
        ),
    )
    parser.add_argument(
        "--teacher-cfg-symmetric-span",
        type=float,
        help=(
            "Inference-only base-model TV-CFG diagnostic. Use a symmetric "
            "triangular schedule around --teacher-guidance-scale with half-span s, "
            "then normalize over actual scheduler timestep intervals so its weighted "
            "mean remains the configured teacher scale. The paper/default choice "
            "is s=omega-1 (for omega=7, pass 6)."
        ),
    )
    parser.add_argument("--student-guidance-scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=contract.resolution)
    parser.add_argument("--max-sequence-length", type=int, default=contract.max_sequence_length)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--finalize-shards",
        action="store_true",
        help="Validate completed shards and write the canonical manifest without model loading.",
    )
    parser.add_argument(
        "--initial-noise-mode",
        choices=(
            "gaussian",
            "lowpass",
            "pink",
            "antithetic",
            "zeros",
            "ones",
            "constant_positive",
            "constant_negative",
        ),
        default="gaussian",
        help="Initial latent construction used for this inference diagnostic.",
    )
    parser.add_argument(
        "--initial-noise-lowpass-cutoff",
        type=float,
        default=0.20,
        help="Normalized radial cutoff used when --initial-noise-mode=lowpass.",
    )
    parser.add_argument(
        "--initial-noise-pink-beta",
        type=float,
        default=0.3,
        help="Power-spectrum exponent beta used when --initial-noise-mode=pink.",
    )
    parser.add_argument(
        "--initial-noise-constant-magnitude",
        type=float,
        default=10.0,
        help="Absolute value for constant_positive/constant_negative (default: 10 sigma).",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def run_sd3_evaluation(
    contract: SD3EvaluationContract,
    argv: list[str] | None = None,
) -> int:
    args = _parser(contract).parse_args(argv)
    contract = resolve_sd3_evaluation_contract(contract, args.evaluation_protocol)
    if not 0.0 < args.teacher_guidance_scale < float("inf"):
        raise ValueError("--teacher-guidance-scale must be finite and positive")
    if not 0.0 < args.initial_noise_lowpass_cutoff <= 1.0:
        raise ValueError("--initial-noise-lowpass-cutoff must be in (0, 1]")
    if not 0.0 <= args.initial_noise_pink_beta < float("inf"):
        raise ValueError("--initial-noise-pink-beta must be finite and non-negative")
    if not 0.0 < args.initial_noise_constant_magnitude < float("inf"):
        raise ValueError("--initial-noise-constant-magnitude must be finite and positive")
    if args.mode in contract.student_modes and not args.checkpoint:
        raise ValueError(f"--checkpoint is required for a {contract.display_name} student mode")
    if args.mode == contract.base_mode and args.checkpoint:
        raise ValueError(f"--checkpoint is not valid for {contract.base_mode}")
    if args.first_interval_lora_checkpoint is not None:
        if args.mode not in contract.student_modes:
            raise ValueError("--first-interval-lora-checkpoint requires a student mode")
        if args.inference_steps is not None and args.inference_steps != 4:
            raise ValueError("first-interval LoRA evaluation requires exactly four student steps")
        lora_weights = args.first_interval_lora_checkpoint / "pytorch_lora_weights.safetensors"
        if not lora_weights.is_file():
            raise ValueError(f"missing first-interval LoRA weights: {lora_weights}")
    if args.first_interval_transformer_checkpoint is not None:
        if args.mode not in contract.student_modes:
            raise ValueError("--first-interval-transformer-checkpoint requires a student mode")
        if args.inference_steps is not None and args.inference_steps != 4:
            raise ValueError("first-interval transformer evaluation requires exactly four student steps")
        transformer_config = args.first_interval_transformer_checkpoint / "transformer" / "config.json"
        if not transformer_config.is_file():
            raise ValueError(f"missing first-interval transformer config: {transformer_config}")
    if args.first_interval_residual_scale_checkpoint is not None:
        if args.mode not in contract.student_modes:
            raise ValueError("--first-interval-residual-scale-checkpoint requires a student mode")
        if args.first_interval_lora_checkpoint is not None:
            raise ValueError("choose only one first-interval correction")
        residual_weights = (
            args.first_interval_residual_scale_checkpoint / "residual_scales.pt"
        )
        if not residual_weights.is_file():
            raise ValueError(f"missing first-interval residual scales: {residual_weights}")
    if args.first_interval_residual_bias_checkpoint is not None:
        if args.mode not in contract.student_modes:
            raise ValueError("--first-interval-residual-bias-checkpoint requires a student mode")
        residual_bias_weights = (
            args.first_interval_residual_bias_checkpoint / "residual_biases.pt"
        )
        if not residual_bias_weights.is_file():
            raise ValueError(f"missing first-interval residual biases: {residual_bias_weights}")
    residual_bias_intervals = sorted(set(args.first_interval_residual_bias_intervals))
    if residual_bias_intervals != [0] and args.first_interval_residual_bias_checkpoint is None:
        raise ValueError(
            "--first-interval-residual-bias-intervals requires "
            "--first-interval-residual-bias-checkpoint"
        )
    if len(residual_bias_intervals) != len(args.first_interval_residual_bias_intervals):
        raise ValueError("--first-interval-residual-bias-intervals must not repeat an index")
    if not residual_bias_intervals or any(
        index < 0 or index > 3 for index in residual_bias_intervals
    ):
        raise ValueError("--first-interval-residual-bias-intervals must be a subset of 0..3")
    selected_first_interval_corrections = sum(
        value is not None
        for value in (
            args.first_interval_lora_checkpoint,
            args.first_interval_transformer_checkpoint,
            args.first_interval_residual_scale_checkpoint,
            args.first_interval_residual_bias_checkpoint,
        )
    )
    if selected_first_interval_corrections > 1:
        raise ValueError("choose only one first-interval correction")
    if args.student_guidance_scale != 1.0:
        raise ValueError(
            f"{contract.display_name} student guidance must be exactly 1.0 for one forward per step"
        )
    if not 0.0 <= args.teacher_cfg_start_fraction < 1.0:
        raise ValueError("--teacher-cfg-start-fraction must be in [0, 1)")
    if args.teacher_cfg_threshold_a is not None and not 0.0 < args.teacher_cfg_threshold_a < 1.0:
        raise ValueError("--teacher-cfg-threshold-a must be in (0, 1)")
    if args.teacher_cfg_symmetric_span is not None and not (
        0.0 < args.teacher_cfg_symmetric_span < args.teacher_guidance_scale
    ):
        raise ValueError("--teacher-cfg-symmetric-span must satisfy 0 < s < teacher CFG")
    selected_cfg_schedules = sum(
        (
            args.teacher_cfg_start_fraction > 0,
            args.teacher_cfg_threshold_a is not None,
            args.teacher_cfg_symmetric_span is not None,
        )
    )
    if selected_cfg_schedules > 1:
        raise ValueError("choose exactly one optional teacher CFG schedule")
    scheduled_cfg_requested = selected_cfg_schedules == 1
    if scheduled_cfg_requested and args.mode != contract.base_mode:
        raise ValueError("piecewise teacher CFG is only valid for the frozen base model")
    if contract.strict_protocol and scheduled_cfg_requested:
        raise ValueError("piecewise teacher CFG is outside the immutable strict protocol")
    checkpoint_manifest = None
    if args.mode in contract.student_modes:
        checkpoint_root = Path(args.checkpoint).expanduser()
        manifest_path = checkpoint_root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing {contract.display_name} checkpoint manifest: {manifest_path}")
        checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_objective = args.mode.removeprefix(f"{contract.mode_prefix}-")
        if checkpoint_manifest.get("backend") != contract.backend:
            raise ValueError(
                f"evaluation mode {args.mode} requires checkpoint backend={contract.backend}, "
                f"got {checkpoint_manifest.get('backend')!r}"
            )
        if checkpoint_manifest.get("model_family") != contract.model_family:
            raise ValueError(
                f"evaluation mode {args.mode} requires checkpoint model_family={contract.model_family}, "
                f"got {checkpoint_manifest.get('model_family')!r}"
            )
        if checkpoint_manifest.get("prediction_type") != "flow":
            raise ValueError(
                f"evaluation mode {args.mode} requires checkpoint prediction_type='flow', "
                f"got {checkpoint_manifest.get('prediction_type')!r}"
            )
        if (
            contract.source_model_id is not None
            and checkpoint_manifest.get("source_model_id") != contract.source_model_id
        ):
            raise ValueError(
                f"evaluation mode {args.mode} requires checkpoint source_model_id="
                f"{contract.source_model_id!r}, got {checkpoint_manifest.get('source_model_id')!r}"
            )
        if contract.model_revision is not None and checkpoint_manifest.get("model_revision") != contract.model_revision:
            raise ValueError(
                f"evaluation mode {args.mode} requires checkpoint model_revision={contract.model_revision}, "
                f"got {checkpoint_manifest.get('model_revision')!r}"
            )
        if (
            contract.component_config_sha256 is not None
            and checkpoint_manifest.get("model_config_sha256") != contract.component_config_sha256
        ):
            raise ValueError("checkpoint model component hashes do not match the SD3.5M protocol")
        if checkpoint_manifest.get("objective") != expected_objective:
            raise ValueError(
                f"evaluation mode {args.mode} requires checkpoint objective={expected_objective}, "
                f"got {checkpoint_manifest.get('objective')!r}"
            )
    default_steps = contract.base_inference_steps if args.mode == contract.base_mode else 4
    if args.inference_steps is None:
        args.inference_steps = default_steps
    if args.first_interval_lora_checkpoint is not None and args.inference_steps != 4:
        raise ValueError("first-interval LoRA evaluation requires exactly four student steps")
    if args.first_interval_residual_scale_checkpoint is not None and args.inference_steps != 4:
        raise ValueError("first-interval residual-scale evaluation requires exactly four student steps")
    if args.first_interval_transformer_checkpoint is not None and args.inference_steps != 4:
        raise ValueError("first-interval transformer evaluation requires exactly four student steps")
    if args.first_interval_residual_bias_checkpoint is not None and args.inference_steps != 4:
        raise ValueError("first-interval residual-bias evaluation requires exactly four student steps")
    if contract.strict_protocol:
        expected_steps = contract.base_inference_steps if args.mode == contract.base_mode else 4
        strict_values = {
            "evaluation protocol": (args.evaluation_protocol, contract.evaluation_protocol),
            "inference steps": (args.inference_steps, expected_steps),
            "student guidance scale": (args.student_guidance_scale, 1.0),
            "resolution": (args.resolution, contract.resolution),
            "max sequence length": (
                args.max_sequence_length,
                contract.max_sequence_length,
            ),
            "prompt count": (args.expected_count, contract.expected_prompt_count),
            "generation batch size": (args.batch_size, 1),
            "initial noise mode": (args.initial_noise_mode, "gaussian"),
        }
        if args.mode == contract.base_mode:
            # The reference run really samples at the protocol's frozen teacher
            # CFG, so it stays pinned.  A four-step student always samples at
            # CFG 1 (see effective_guidance below), so for student modes
            # --teacher-guidance-scale is not a sampler field at all: it names
            # the frozen-teacher CFG the checkpoint was distilled with and is
            # checked against the checkpoint's own flow contract further down.
            strict_values["teacher guidance scale"] = (
                args.teacher_guidance_scale,
                contract.teacher_guidance_scale,
            )
        mismatches = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in strict_values.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(
                f"{contract.evaluation_protocol} is immutable; mismatched fields: {mismatches}"
            )
    if args.batch_size <= 0 or args.expected_count <= 0 or args.inference_steps <= 0:
        raise ValueError("batch size, expected count, and inference steps must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_id < num_shards")
    if contract.prompt_csv_sha256 is not None:
        actual_prompt_sha256 = sha256_file(args.prompts)
        if actual_prompt_sha256 != contract.prompt_csv_sha256:
            raise ValueError(
                f"prompt CSV SHA256 {actual_prompt_sha256} does not match "
                f"{contract.evaluation_protocol} ({contract.prompt_csv_sha256})"
            )
    records = _records(args.prompts, args.expected_count)
    if args.finalize_shards:
        if args.num_shards == 1:
            raise ValueError("--finalize-shards requires --num-shards greater than one")
        _finalize_shards(args, records, contract)
        return 0

    protocol_identity = _protocol_identity(contract)
    if not torch.cuda.is_available():
        raise RuntimeError(f"{contract.display_name} generation requires CUDA")

    dtype = torch.bfloat16
    scheduled_cfg = scheduled_cfg_requested
    pipe = _load_sd3_evaluation_pipeline(args, contract, torch_dtype=dtype)
    if args.mode in contract.student_modes:
        pipe.transformer = load_sd3_transformer(args.checkpoint, torch_dtype=dtype)
    suffix_transformer = pipe.transformer
    first_interval_transformer = None
    if args.first_interval_transformer_checkpoint is not None:
        first_interval_transformer = load_sd3_transformer(
            args.first_interval_transformer_checkpoint,
            torch_dtype=dtype,
        )
    if args.first_interval_lora_checkpoint is not None:
        pipe.load_lora_weights(
            str(args.first_interval_lora_checkpoint), adapter_name="first_interval"
        )
    residual_scale_adapter = None
    residual_scale_payload = None
    if args.first_interval_residual_scale_checkpoint is not None:
        residual_scale_adapter, residual_scale_payload = load_sd3_residual_scale_adapter(
            pipe.transformer, args.first_interval_residual_scale_checkpoint
        )
    residual_bias_adapter = None
    residual_bias_payload = None
    if args.first_interval_residual_bias_checkpoint is not None:
        residual_bias_adapter, residual_bias_payload = load_sd3_residual_bias_adapter(
            pipe.transformer, args.first_interval_residual_bias_checkpoint
        )
    model_runtime_contract = None
    if contract.model_revision is not None:
        model_runtime_contract = validate_sd3_pipeline_contract(
            pipe,
            family=contract.model_family,
            model_path=args.model,
            expected_component_sha256=contract.component_config_sha256,
            expected_revision=contract.model_revision,
        )
    pipe.to("cuda")
    if first_interval_transformer is not None:
        first_interval_transformer.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    shift = float(pipe.scheduler.config.shift)
    if not abs(shift - 3.0) < 1e-7:
        raise ValueError(f"expected {contract.display_name} FlowMatch shift=3, got {shift}")
    vae_scale = int(pipe.vae_scale_factor)
    latent_shape = (
        int(pipe.transformer.config.in_channels),
        args.resolution // vae_scale,
        args.resolution // vae_scale,
    )
    if checkpoint_manifest is not None and checkpoint_manifest.get("method") == "external":
        # Externally published students carry no local training contract.  Check
        # the sampler the publisher documented instead (see
        # tools/wrap_external_sd3m_student.py), leaving trained checkpoints on
        # the unchanged flow-contract path below.
        external_required = {
            "scheduler_shift": shift,
            "inference_steps": args.inference_steps,
            "student_guidance_scale": args.student_guidance_scale,
        }
        for key, expected in external_required.items():
            actual = checkpoint_manifest.get(key)
            if actual != expected:
                raise ValueError(
                    f"external checkpoint manifest {key}={actual!r}, expected {expected!r}"
                )
    elif checkpoint_manifest is not None:
        flow_contract = checkpoint_manifest.get("flow_contract", {})
        required = {
            "shift": shift,
            "student_steps": args.inference_steps,
            "teacher_guidance_scale": args.teacher_guidance_scale,
            "student_guidance_scale": 1.0,
            "student_forwards_per_step": 1,
        }
        if contract.strict_protocol:
            required.update(
                {
                    "reference_inference_steps": contract.base_inference_steps,
                    "guidance_mode": "classic_cfg_no_slg",
                    "use_dynamic_shifting": False,
                    "student_sigma_grid": "uniform_unshifted",
                }
            )
        for key, expected in required.items():
            actual = flow_contract.get(key)
            if actual != expected:
                raise ValueError(
                    f"checkpoint flow contract {key}={actual!r}, expected {expected!r}"
                )
    latent_hashes, latent_bank_sha256 = _latent_bank_identity(
        records,
        latent_shape,
        initial_noise_mode=args.initial_noise_mode,
        lowpass_cutoff=args.initial_noise_lowpass_cutoff,
        pink_beta=args.initial_noise_pink_beta,
        constant_magnitude=args.initial_noise_constant_magnitude,
    )
    request_path = None
    request_sha256 = None
    request = None
    if contract.immutable_request:
        request = _generation_request(
            args,
            contract,
            latent_shape,
            checkpoint_manifest,
            model_runtime_contract,
            protocol_identity,
            latent_bank_sha256,
        )
        request_path, request_sha256 = prepare_generation_request(
            args.output, request, resume=args.resume
        )
    else:
        args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "images").mkdir(exist_ok=True)
    protocol = _protocol(
        args,
        pipe,
        latent_shape,
        contract,
        checkpoint_manifest,
        model_runtime_contract,
        request_path,
        request_sha256,
        request,
    )
    if args.num_shards > 1:
        protocol["generation_shard"] = {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "assignment": "record_index_mod_num_shards",
        }
    effective_guidance = (
        args.teacher_guidance_scale if args.mode == contract.base_mode else args.student_guidance_scale
    )
    cfg_normalization = None
    if args.teacher_cfg_symmetric_span is not None:
        pipe.scheduler.set_timesteps(args.inference_steps, device="cpu")
        guidance_schedule, cfg_normalization = _symmetric_cfg_schedule(
            pipe.scheduler.timesteps,
            args.teacher_guidance_scale,
            args.teacher_cfg_symmetric_span,
        )
        cfg_switch_step = None
    elif args.teacher_cfg_threshold_a is not None:
        pipe.scheduler.set_timesteps(args.inference_steps, device="cpu")
        guidance_schedule, cfg_switch_step = _threshold_cfg_schedule(
            pipe.scheduler.timesteps,
            args.teacher_cfg_threshold_a,
            args.teacher_guidance_scale,
            num_train_timesteps=int(pipe.scheduler.config.num_train_timesteps),
        )
    else:
        guidance_schedule, cfg_switch_step = _piecewise_cfg_schedule(
            args.inference_steps,
            args.teacher_cfg_start_fraction,
            args.teacher_guidance_scale,
        )
    protocol["teacher_cfg_switch_step"] = cfg_switch_step
    protocol["teacher_cfg_realized_guidance_schedule"] = guidance_schedule
    protocol["teacher_cfg_normalization_A"] = cfg_normalization
    if args.first_interval_lora_checkpoint is not None:
        protocol.update(
            {
                "first_interval_lora_checkpoint": str(
                    args.first_interval_lora_checkpoint.resolve()
                ),
                "lora_enabled_intervals": [0],
                "lora_disabled_intervals": [1, 2, 3],
            }
        )
    if args.first_interval_transformer_checkpoint is not None:
        protocol.update(
            {
                "first_interval_transformer_checkpoint": str(
                    args.first_interval_transformer_checkpoint.resolve()
                ),
                "full_transformer_enabled_intervals": [0],
                "frozen_s7_enabled_intervals": [1, 2, 3],
            }
        )
    if args.first_interval_residual_scale_checkpoint is not None:
        protocol.update(
            {
                "first_interval_residual_scale_checkpoint": str(
                    args.first_interval_residual_scale_checkpoint.resolve()
                ),
                "residual_scale_blocks": residual_scale_payload["blocks"],
                "residual_scale_enabled_intervals": [0],
                "residual_scale_disabled_intervals": [1, 2, 3],
            }
        )
    if args.first_interval_residual_bias_checkpoint is not None:
        protocol.update(
            {
                "first_interval_residual_bias_checkpoint": str(
                    args.first_interval_residual_bias_checkpoint.resolve()
                ),
                "residual_bias_blocks": residual_bias_payload["blocks"],
                "residual_bias_enabled_intervals": residual_bias_intervals,
                "residual_bias_disabled_intervals": [
                    index for index in range(4) if index not in residual_bias_intervals
                ],
                "residual_bias_trained_interval_index": residual_bias_payload.get(
                    "interval_index", 0
                ),
            }
        )

    def step_end_callback(
        pipeline: Any, step_index: int, _timestep: Any, callback_kwargs: dict
    ) -> dict:
        # The scheduler has already applied interval zero when this callback is
        # first invoked. Disabling here leaves all three suffix evaluations
        # bitwise on the frozen S7 weights.
        if args.first_interval_lora_checkpoint is not None and step_index == 0:
            pipeline.transformer.disable_adapters()
        if first_interval_transformer is not None and step_index == 0:
            pipeline.transformer = suffix_transformer
        if residual_scale_adapter is not None and step_index == 0:
            residual_scale_adapter.disable()
        # Enable the adapter for the interval the scheduler is about to apply.
        # With the default interval set {0} this disables it after interval zero
        # and leaves it disabled, exactly as the first-interval-only path did.
        apply_residual_bias_interval(
            residual_bias_adapter, step_index + 1, residual_bias_intervals
        )
        next_step = step_index + 1
        if scheduled_cfg and next_step < len(guidance_schedule):
            pipeline._guidance_scale = guidance_schedule[next_step]
        return callback_kwargs
    protocol_suffix = (
        ""
        if args.num_shards == 1
        else _shard_suffix(args.shard_id, args.num_shards)
    )
    protocol_path = args.output / f"generation_protocol{protocol_suffix}.json"
    if args.resume and protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("--resume protocol differs from the existing generation protocol")

    selected_records = records[args.shard_id :: args.num_shards]
    # One read of the shard logs, instead of one sidecar open per candidate image.
    identities = _load_identity_log(args.output) if contract.immutable_request else None
    identity_log = _identity_log_path(args.output, args.shard_id, args.num_shards)
    pending = []
    for record in selected_records:
        digest = latent_hashes[record["case_number"]]
        if not (
            args.resume
            and _valid_image(
                _image_path(args.output, record),
                expected_latent_sha256=digest,
                expected_request_sha256=request_sha256,
                require_identity=contract.immutable_request,
                identities=identities,
            )
        ):
            pending.append(record)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        if args.first_interval_lora_checkpoint is not None:
            # The preceding batch finishes with the adapter disabled.
            pipe.transformer.enable_adapters()
        if first_interval_transformer is not None:
            pipe.transformer = first_interval_transformer
        if residual_scale_adapter is not None:
            # The preceding batch finishes with the residual scales disabled.
            residual_scale_adapter.enable()
        # The preceding batch leaves the adapter in its final per-step state.
        apply_residual_bias_interval(residual_bias_adapter, 0, residual_bias_intervals)
        batch_latents = torch.stack(
            [
                _noise(
                    record,
                    latent_shape,
                    initial_noise_mode=args.initial_noise_mode,
                    lowpass_cutoff=args.initial_noise_lowpass_cutoff,
                    pink_beta=args.initial_noise_pink_beta,
                    constant_magnitude=args.initial_noise_constant_magnitude,
                )[0]
                for record in batch
            ]
        ).to(device="cuda", dtype=dtype)
        inference_schedule = (
            {"sigmas": _unshifted_inference_sigmas(args.inference_steps)}
            if args.mode in contract.student_modes
            else {"num_inference_steps": args.inference_steps}
        )
        pipeline_guidance = guidance_schedule[0] if scheduled_cfg else effective_guidance
        result = pipe(
            prompt=[record["prompt"] for record in batch],
            prompt_2=None,
            prompt_3=None,
            negative_prompt=[""] * len(batch),
            height=args.resolution,
            width=args.resolution,
            guidance_scale=pipeline_guidance,
            latents=batch_latents,
            output_type="pil",
            max_sequence_length=args.max_sequence_length,
            callback_on_step_end=(
                step_end_callback
                if scheduled_cfg
                or args.first_interval_lora_checkpoint is not None
                or residual_scale_adapter is not None
                or residual_bias_adapter is not None
                else None
            ),
            callback_on_step_end_tensor_inputs=["latents"],
            **inference_schedule,
        )
        for image, record in zip(result.images, batch):
            _save_image_with_identity(
                image,
                _image_path(args.output, record),
                latent_sha256=latent_hashes[record["case_number"]],
                generation_request_sha256=request_sha256,
                write_identity=contract.immutable_request,
                identity_log=identity_log,
            )
        print(
            f"[{contract.mode_prefix}-generate] "
            f"{min(start + len(batch), len(pending))}/{len(pending)} new",
            flush=True,
        )

    _write_outputs(
        args.output,
        selected_records,
        latent_hashes,
        protocol,
        contract,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Backward-compatible SD3 Medium evaluation entry point."""

    return run_sd3_evaluation(SD3M_EVALUATION_CONTRACT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
