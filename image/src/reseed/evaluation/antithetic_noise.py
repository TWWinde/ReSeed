#!/usr/bin/env python3
"""Evaluate PixArt responses to exact antithetic initial-noise pairs.

The evaluator is inference-only.  It builds an auditable latent bank containing
``z``, ``-z`` and a matched independent pair for every prompt/pair id, delegates
generation to the repository's existing PixArt paired generator, and scores the
resulting manifests.  DINO/CLIP feature-space measurements are primary; raw
pixel or final-latent Pearson correlations are auxiliary diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    # The registry executes this file by path; expose repository-native tools
    # without requiring callers to add the repository root to PYTHONPATH.
    sys.path.insert(0, str(PROJECT_ROOT))


PAIR_ROLES = ("pos", "neg", "rand1", "rand2")
LATENT_PROTOCOL = "pixart_antithetic_noise_latents_v1"
EVALUATION_PROTOCOL = "pixart_antithetic_noise_metric_v1"


def safe_pearson(left: Any, right: Any, *, eps: float = 1e-12) -> float | np.ndarray:
    """Compute row-wise Pearson correlation without propagating invalid rows.

    One-dimensional inputs return a float.  Inputs with leading batch
    dimensions are flattened per first-dimension row and return ``[B]``.
    PyTorch tensors and NumPy-compatible arrays are accepted.
    """

    if hasattr(left, "detach"):
        left = left.detach().cpu().numpy()
    if hasattr(right, "detach"):
        right = right.detach().cpu().numpy()
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Pearson inputs must have identical shapes: {a.shape} != {b.shape}")
    scalar = a.ndim == 1
    if scalar:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
    elif a.ndim < 2:
        raise ValueError("Pearson inputs must contain at least one vector dimension")
    else:
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
    result = np.full(a.shape[0], np.nan, dtype=np.float64)
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if finite.any():
        x = a[finite] - a[finite].mean(axis=1, keepdims=True)
        y = b[finite] - b[finite].mean(axis=1, keepdims=True)
        x_var = np.mean(x * x, axis=1)
        y_var = np.mean(y * y, axis=1)
        valid = (x_var > eps) & (y_var > eps)
        values = np.full(x.shape[0], np.nan, dtype=np.float64)
        if valid.any():
            numerator = np.sum(x[valid] * y[valid], axis=1)
            denominator = np.linalg.norm(x[valid], axis=1) * np.linalg.norm(y[valid], axis=1)
            values[valid] = np.clip(numerator / denominator, -1.0, 1.0)
        result[np.flatnonzero(finite)] = values
    return float(result[0]) if scalar else result


def affine_antisymmetry_score(correlation: float) -> float:
    """Map perfect positive/negative correlation to zero/one respectively."""

    if not math.isfinite(correlation):
        return math.nan
    return (1.0 - max(-1.0, min(1.0, correlation))) / 2.0


def variance_reduction(
    pos: Any,
    neg: Any,
    rand1: Any,
    rand2: Any,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return correlation-derived and empirical antithetic VRF values."""

    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in (pos, neg, rand1, rand2)]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("VRF inputs must have identical shapes")
    if arrays[0].size < 2 or not all(np.isfinite(array).all() for array in arrays):
        return {
            "pair_corr": math.nan,
            "vrf_from_corr": math.nan,
            "pair_mean_variance": math.nan,
            "independent_variance": math.nan,
            "vrf_empirical": math.nan,
        }
    rho = float(safe_pearson(arrays[0], arrays[1], eps=eps))
    pair_mean = (arrays[0] + arrays[1]) / 2.0
    random_mean = (arrays[2] + arrays[3]) / 2.0
    pair_variance = float(np.var(pair_mean, ddof=1))
    independent_variance = float(np.var(random_mean, ddof=1))
    return {
        "pair_corr": rho,
        "vrf_from_corr": 1.0 / (1.0 + rho) if math.isfinite(rho) and 1.0 + rho > eps else math.nan,
        "pair_mean_variance": pair_variance,
        "independent_variance": independent_variance,
        "vrf_empirical": independent_variance / pair_variance if pair_variance > eps else math.nan,
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    result: dict[str, float | int] = {
        "valid_count": int(valid.size),
        "invalid_count": int(array.size - valid.size),
    }
    for key in ("mean", "std", "median", "q05", "q25", "q75", "q95"):
        result[key] = math.nan
    if not valid.size:
        return result
    result.update(
        {
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
            "median": float(np.median(valid)),
            "q05": float(np.quantile(valid, 0.05)),
            "q25": float(np.quantile(valid, 0.25)),
            "q75": float(np.quantile(valid, 0.75)),
            "q95": float(np.quantile(valid, 0.95)),
        }
    )
    return result


def json_safe(value: Any) -> Any:
    """Convert non-finite floating values to JSON null recursively."""

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def cosine_distance(left: np.ndarray, right: np.ndarray, *, eps: float = 1e-12) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.nan
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= eps:
        return math.nan
    return 1.0 - float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def relative_error(value: np.ndarray, target: np.ndarray, *, eps: float = 1e-12) -> float:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if not np.isfinite(value).all() or not np.isfinite(target).all():
        return math.nan
    return float(np.linalg.norm(value - target) / (np.linalg.norm(target) + eps))


def read_prompts(path: Path, limit: int) -> list[dict[str, Any]]:
    from tools.feature_diversity_eval import read_prompts as existing_reader

    return existing_reader(path, limit)


def build_antithetic_bank(
    prompts: list[dict[str, Any]],
    *,
    pairs_per_prompt: int,
    base_seed: int,
    latent_size: int,
    prompt_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Create exact ``z,-z`` pairs plus equally sized independent baselines."""

    import torch

    if pairs_per_prompt < 2:
        raise ValueError("pairs_per_prompt must be at least 2")
    records: list[dict[str, Any]] = []
    latents = []
    for prompt_row in prompts:
        prompt_id = int(prompt_row["prompt_id"])
        for pair_id in range(pairs_per_prompt):
            noise_seed = base_seed + (prompt_id * pairs_per_prompt + pair_id) * 3
            generated = []
            for offset in range(3):
                generator = torch.Generator(device="cpu").manual_seed(noise_seed + offset)
                generated.append(
                    torch.randn((4, latent_size, latent_size), generator=generator).to(torch.float16)
                )
            role_latents = (generated[0], -generated[0], generated[1], generated[2])
            first_case = len(records)
            for role, latent in zip(PAIR_ROLES, role_latents):
                case_number = len(records)
                latent = latent.contiguous()
                records.append(
                    {
                        "case_number": case_number,
                        "source": "antithetic-noise-metric",
                        "prompt": prompt_row["prompt"],
                        "prompt_id": prompt_id,
                        "coco_id": prompt_id,
                        "pair_id": pair_id,
                        "pair_role": role,
                        "sample_id": pair_id * 4 + PAIR_ROLES.index(role),
                        "evaluation_seed": noise_seed if role in ("pos", "neg") else noise_seed + PAIR_ROLES.index(role) - 1,
                        "noise_seed": noise_seed,
                        "antithetic_of_case": first_case if role == "neg" else None,
                        "latent_sha256": hashlib.sha256(latent.numpy().tobytes()).hexdigest(),
                    }
                )
                latents.append(latent)
    tensor = torch.stack(latents)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "protocol": LATENT_PROTOCOL,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_count": len(prompts),
        "pairs_per_prompt": pairs_per_prompt,
        "samples_per_pair": 4,
        "base_seed": base_seed,
        "seed_recipe": "base + (prompt_id * pairs_per_prompt + pair_id) * 3",
        "records": records,
        "latents": tensor,
    }
    torch.save(payload, output)
    return payload


def load_manifest(path: Path, *, pairs_per_prompt: int | None = None) -> tuple[list[dict[str, Any]], dict[int, dict[int, dict[str, int]]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("generation manifest is empty")
    records.sort(key=lambda row: int(row["case_number"]))
    groups: dict[int, dict[int, dict[str, int]]] = {}
    for index, record in enumerate(records):
        for field in ("case_number", "prompt", "prompt_id", "pair_id", "pair_role", "latent_sha256", "image_path"):
            if field not in record:
                raise ValueError(f"manifest record is missing {field!r}")
        if int(record["case_number"]) != index:
            raise ValueError("case_number must be contiguous and ordered")
        prompt_id = int(record["prompt_id"])
        pair_id = int(record["pair_id"])
        role = str(record["pair_role"])
        if role not in PAIR_ROLES:
            raise ValueError(f"unsupported pair role: {role}")
        slot = groups.setdefault(prompt_id, {}).setdefault(pair_id, {})
        if role in slot:
            raise ValueError(f"duplicate role {role} for prompt={prompt_id}, pair={pair_id}")
        slot[role] = index
    expected_prompt_ids = list(range(len(groups)))
    if sorted(groups) != expected_prompt_ids:
        raise ValueError("prompt ids must be contiguous")
    for prompt_id, prompt_pairs in groups.items():
        if pairs_per_prompt is not None and len(prompt_pairs) != pairs_per_prompt:
            raise ValueError(
                f"prompt {prompt_id} has {len(prompt_pairs)} pairs, expected {pairs_per_prompt}"
            )
        if sorted(prompt_pairs) != list(range(len(prompt_pairs))):
            raise ValueError(f"pair ids for prompt {prompt_id} must be contiguous")
        prompts = {records[index]["prompt"] for pair in prompt_pairs.values() for index in pair.values()}
        if len(prompts) != 1:
            raise ValueError(f"prompt text differs within prompt group {prompt_id}")
        for pair_id, roles in prompt_pairs.items():
            if set(roles) != set(PAIR_ROLES):
                raise ValueError(f"prompt={prompt_id}, pair={pair_id} does not contain all roles")
            pos = records[roles["pos"]]
            neg = records[roles["neg"]]
            if pos["latent_sha256"] == neg["latent_sha256"]:
                raise ValueError("positive and negative latent identities are equal")
            if len({records[roles["pos"]]["latent_sha256"], records[roles["rand1"]]["latent_sha256"], records[roles["rand2"]]["latent_sha256"]}) != 3:
                raise ValueError("source and independent baseline latents are not independent")
    return records, groups


def validate_aligned_manifests(
    student: list[dict[str, Any]], teacher: list[dict[str, Any]]
) -> None:
    if len(student) != len(teacher):
        raise ValueError("teacher and student manifests have different lengths")
    identity = ("case_number", "prompt", "prompt_id", "pair_id", "pair_role", "latent_sha256")
    for index, (left, right) in enumerate(zip(student, teacher)):
        if any(left[key] != right[key] for key in identity):
            raise ValueError(f"teacher/student manifest mismatch at case {index}")


def validate_initial_noise_bank(manifest: Path, records: list[dict[str, Any]]) -> Path:
    """Verify that manifest roles refer to exact z/-z and independent noise."""

    import torch

    protocol_path = manifest.parent / "generation_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"initial-noise validation requires generation_protocol.json next to {manifest}"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bank_path = Path(protocol.get("latent_bank", ""))
    if not bank_path.is_file():
        raise FileNotFoundError(f"generation protocol latent bank is missing: {bank_path}")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    if bank.get("protocol") != LATENT_PROTOCOL:
        raise ValueError(f"unexpected antithetic latent protocol: {bank.get('protocol')!r}")
    bank_records = bank.get("records", [])
    latents = bank.get("latents")
    if len(bank_records) != len(records) or not isinstance(latents, torch.Tensor):
        raise ValueError("latent bank record/tensor count does not match generation manifest")
    for first in range(0, len(records), 4):
        roles = tuple(records[first + offset]["pair_role"] for offset in range(4))
        if roles != PAIR_ROLES:
            raise ValueError(f"latent bank pair roles are not ordered at case {first}: {roles}")
        pos, neg, rand1, rand2 = (latents[first + offset] for offset in range(4))
        if not torch.equal(neg, -pos):
            raise ValueError(f"negative latent is not the exact antipode at case {first}")
        if torch.equal(pos, rand1) or torch.equal(pos, rand2) or torch.equal(rand1, rand2):
            raise ValueError(f"independent baseline contains a duplicate at case {first}")
        for offset in range(4):
            bank_record = bank_records[first + offset]
            manifest_record = records[first + offset]
            for key in ("case_number", "prompt_id", "pair_id", "pair_role", "latent_sha256"):
                if bank_record[key] != manifest_record[key]:
                    raise ValueError(f"latent bank/manifest mismatch at case {first + offset}: {key}")
    return bank_path


def image_vector(record: dict[str, Any]) -> np.ndarray:
    from PIL import Image

    with Image.open(record["image_path"]) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return value.transpose(2, 0, 1).reshape(-1)


def latent_vector(record: dict[str, Any]) -> np.ndarray:
    import torch

    path = record.get("latent_path")
    if not path:
        raise ValueError("latent metric requested but manifest has no latent_path")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"invalid final latent at {path}")
    return value.float().numpy().reshape(-1)


def _space_metrics(
    records: list[dict[str, Any]],
    groups: dict[int, dict[int, dict[str, int]]],
    *,
    space: str,
    matrix: np.ndarray | None = None,
    loader: Callable[[dict[str, Any]], np.ndarray] | None = None,
    per_pair: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    anti, centralized, random, anti_distance, random_distance = [], [], [], [], []
    for prompt_id, prompt_pairs in groups.items():
        indexes = sorted({index for roles in prompt_pairs.values() for index in roles.values()})
        values = {
            index: np.asarray(matrix[index] if matrix is not None else loader(records[index]), dtype=np.float32).reshape(-1)
            for index in indexes
        }
        center = np.stack(
            [values[roles[role]] for roles in prompt_pairs.values() for role in ("pos", "neg")]
        ).mean(axis=0)
        for pair_id, roles in prompt_pairs.items():
            pos, neg = values[roles["pos"]], values[roles["neg"]]
            rand1, rand2 = values[roles["rand1"]], values[roles["rand2"]]
            raw_corr = float(safe_pearson(pos, neg))
            centered_corr = float(safe_pearson(pos - center, neg - center))
            random_corr = float(safe_pearson(rand1, rand2))
            anti_dist = cosine_distance(pos, neg)
            random_dist = cosine_distance(rand1, rand2)
            anti.append(raw_corr)
            centralized.append(centered_corr)
            random.append(random_corr)
            anti_distance.append(anti_dist)
            random_distance.append(random_dist)
            row = per_pair.setdefault(
                (prompt_id, pair_id),
                {
                    "prompt_id": prompt_id,
                    "pair_id": pair_id,
                    "prompt": records[roles["pos"]]["prompt"],
                },
            )
            row.update(
                {
                    f"{space}_antithetic_corr": raw_corr,
                    f"{space}_centralized_corr": centered_corr,
                    f"{space}_affine_antisymmetry": affine_antisymmetry_score(centered_corr),
                    f"{space}_random_corr": random_corr,
                    f"{space}_antithetic_cosine_distance": anti_dist,
                    f"{space}_random_cosine_distance": random_dist,
                }
            )
    anti_summary = summarize(anti)
    centralized_summary = summarize(centralized)
    random_summary = summarize(random)
    anti_distance_summary = summarize(anti_distance)
    random_distance_summary = summarize(random_distance)
    random_mean = float(random_summary["mean"])
    anti_mean = float(anti_summary["mean"])
    distance_denominator = abs(float(random_distance_summary["mean"]))
    return {
        "antithetic_corr": anti_summary,
        "centralized_antithetic_corr": centralized_summary,
        "affine_antisymmetry": summarize([affine_antisymmetry_score(value) for value in centralized]),
        "random_corr": random_summary,
        "corr_gap": anti_mean - random_mean if math.isfinite(anti_mean) and math.isfinite(random_mean) else math.nan,
        "antithetic_cosine_distance": anti_distance_summary,
        "random_cosine_distance": random_distance_summary,
        "antithetic_diversity_gain": (
            (float(anti_distance_summary["mean"]) - float(random_distance_summary["mean"]))
            / max(distance_denominator, 1e-12)
        ),
    }


def _teacher_student_metrics(
    student_records: list[dict[str, Any]],
    teacher_records: list[dict[str, Any]],
    groups: dict[int, dict[int, dict[str, int]]],
    *,
    space: str,
    student_matrix: np.ndarray | None = None,
    teacher_matrix: np.ndarray | None = None,
    loader: Callable[[dict[str, Any]], np.ndarray] | None = None,
    per_pair: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        key: []
        for key in (
            "teacher_anc",
            "student_anc",
            "anti_gap",
            "pair_difference_cosine",
            "pair_difference_relative_error",
            "pair_difference_mse",
            "pair_mean_error",
            "pointwise_error",
        )
    }
    for prompt_id, prompt_pairs in groups.items():
        for pair_id, roles in prompt_pairs.items():
            def get(records, matrix, role):
                index = roles[role]
                source = matrix[index] if matrix is not None else loader(records[index])
                return np.asarray(source, dtype=np.float32).reshape(-1)

            sp, sn = get(student_records, student_matrix, "pos"), get(student_records, student_matrix, "neg")
            tp, tn = get(teacher_records, teacher_matrix, "pos"), get(teacher_records, teacher_matrix, "neg")
            s_delta, t_delta = sp - sn, tp - tn
            metrics = {
                "teacher_anc": float(safe_pearson(tp, tn)),
                "student_anc": float(safe_pearson(sp, sn)),
                "pair_difference_cosine": 1.0 - cosine_distance(s_delta, t_delta),
                "pair_difference_relative_error": relative_error(s_delta, t_delta),
                "pair_difference_mse": float(np.mean((s_delta - t_delta) ** 2)),
                "pair_mean_error": float(np.mean((((sp + sn) / 2.0) - ((tp + tn) / 2.0)) ** 2)),
                "pointwise_error": float(np.mean((sp - tp) ** 2)),
            }
            metrics["anti_gap"] = abs(metrics["student_anc"] - metrics["teacher_anc"])
            for key, value in metrics.items():
                values[key].append(value)
                per_pair[(prompt_id, pair_id)][f"{space}_{key}"] = value
    return {key: summarize(metric_values) for key, metric_values in values.items()}


def image_statistics(record: dict[str, Any]) -> dict[str, float]:
    from PIL import Image

    with Image.open(record["image_path"]) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    brightness = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    row_weights = brightness.sum(axis=1)
    denominator = float(row_weights.sum())
    centroid = (
        float(np.dot(np.linspace(0.0, 1.0, brightness.shape[0]), row_weights) / denominator)
        if denominator > 1e-12
        else math.nan
    )
    midpoint = brightness.shape[0] // 2
    return {
        "mean_pixel": float(array.mean()),
        "brightness": float(brightness.mean()),
        "vertical_brightness_imbalance": float(brightness[:midpoint].mean() - brightness[midpoint:].mean()),
        "vertical_brightness_centroid": centroid,
    }


def _statistics_metrics(
    records: list[dict[str, Any]], groups: dict[int, dict[int, dict[str, int]]]
) -> dict[str, Any]:
    names = tuple(image_statistics(records[0]))
    prompt_results = {name: [] for name in names}
    for prompt_pairs in groups.values():
        role_values = {name: {role: [] for role in PAIR_ROLES} for name in names}
        for roles in prompt_pairs.values():
            for role, index in roles.items():
                stats = image_statistics(records[index])
                for name in names:
                    role_values[name][role].append(stats[name])
        for name in names:
            values = role_values[name]
            prompt_results[name].append(
                variance_reduction(values["pos"], values["neg"], values["rand1"], values["rand2"])
            )
    return {
        name: {
            field: summarize([float(row[field]) for row in rows])
            for field in (
                "pair_corr",
                "vrf_from_corr",
                "pair_mean_variance",
                "independent_variance",
                "vrf_empirical",
            )
        }
        for name, rows in prompt_results.items()
    }


def _lpips_metrics(
    records: list[dict[str, Any]],
    groups: dict[int, dict[int, dict[str, int]]],
    *,
    device: str,
    net: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    try:
        import lpips
    except ModuleNotFoundError:
        warnings.warn("LPIPS was requested but the optional lpips package is unavailable; skipping it.")
        return {"status": "skipped", "reason": "lpips package is unavailable"}

    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    model = lpips.LPIPS(net=net).eval().requires_grad_(False).to(resolved_device)
    pairs: list[tuple[str, Path, Path]] = []
    for prompt_pairs in groups.values():
        for roles in prompt_pairs.values():
            pairs.append(("anti", Path(records[roles["pos"]]["image_path"]), Path(records[roles["neg"]]["image_path"])))
            pairs.append(("random", Path(records[roles["rand1"]]["image_path"]), Path(records[roles["rand2"]]["image_path"])))
    output = {"anti": [], "random": []}

    def load(path: Path):
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0)

    with torch.inference_mode():
        for first in range(0, len(pairs), batch_size):
            chunk = pairs[first : first + batch_size]
            left = torch.stack([load(row[1]) for row in chunk]).to(resolved_device)
            right = torch.stack([load(row[2]) for row in chunk]).to(resolved_device)
            scores = model(left, right).flatten().float().cpu().numpy()
            for row, score in zip(chunk, scores):
                output[row[0]].append(float(score))
    anti, random = summarize(output["anti"]), summarize(output["random"])
    denominator = max(abs(float(random["mean"])), 1e-12)
    return {
        "network": net,
        "antithetic": anti,
        "random": random,
        "antithetic_diversity_gain": (float(anti["mean"]) - float(random["mean"])) / denominator,
    }


def _feature_matrix(
    records: list[dict[str, Any]],
    *,
    encoder: str,
    model_id: str,
    device: str,
    dtype: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from tools.feature_diversity_eval import _encode_images

    tensor, metadata = _encode_images(
        records,
        encoder=encoder,
        model_id=model_id,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    return tensor.numpy(), metadata


def score(args: argparse.Namespace) -> Path:
    student_records, groups = load_manifest(
        args.student_manifest, pairs_per_prompt=args.pairs_per_prompt
    )
    validated_bank = None
    if getattr(args, "validate_initial_noise", True):
        validated_bank = validate_initial_noise_bank(args.student_manifest, student_records)
    teacher_records = None
    if args.teacher_manifest:
        teacher_records, teacher_groups = load_manifest(
            args.teacher_manifest, pairs_per_prompt=args.pairs_per_prompt
        )
        if teacher_groups.keys() != groups.keys():
            raise ValueError("teacher/student prompt groups differ")
        validate_aligned_manifests(student_records, teacher_records)
        if getattr(args, "validate_initial_noise", True):
            teacher_bank = validate_initial_noise_bank(args.teacher_manifest, teacher_records)
            if teacher_bank.resolve() != validated_bank.resolve():
                raise ValueError("teacher and student were not generated from the same latent bank")

    per_pair: dict[tuple[int, int], dict[str, Any]] = {}
    spaces: dict[str, Any] = {}
    loader = image_vector if args.metric_space == "image" else latent_vector
    spaces[args.metric_space] = _space_metrics(
        student_records,
        groups,
        space=args.metric_space,
        loader=loader,
        per_pair=per_pair,
    )
    teacher_student: dict[str, Any] = {}
    if teacher_records is not None:
        teacher_student[args.metric_space] = _teacher_student_metrics(
            student_records,
            teacher_records,
            groups,
            space=args.metric_space,
            loader=loader,
            per_pair=per_pair,
        )

    feature_metadata: dict[str, Any] = {}
    requested_features = [] if args.feature_spaces == "none" else args.feature_spaces.split(",")
    for encoder in requested_features:
        encoder = encoder.strip()
        if encoder not in ("dino", "clip"):
            raise ValueError(f"unsupported feature space: {encoder}")
        model_id = args.dino_model if encoder == "dino" else args.clip_model
        dtype = args.dino_dtype if encoder == "dino" else args.clip_dtype
        encoder_records = (
            student_records
            if teacher_records is None
            else [*student_records, *teacher_records]
        )
        combined_matrix, metadata = _feature_matrix(
            encoder_records,
            encoder=encoder,
            model_id=model_id,
            device=args.device,
            dtype=dtype,
            batch_size=args.feature_batch_size,
        )
        student_matrix = combined_matrix[: len(student_records)]
        feature_metadata[encoder] = metadata
        spaces[encoder] = _space_metrics(
            student_records,
            groups,
            space=encoder,
            matrix=student_matrix,
            per_pair=per_pair,
        )
        if teacher_records is not None:
            teacher_matrix = combined_matrix[len(student_records) :]
            teacher_student[encoder] = _teacher_student_metrics(
                student_records,
                teacher_records,
                groups,
                space=encoder,
                student_matrix=student_matrix,
                teacher_matrix=teacher_matrix,
                per_pair=per_pair,
            )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "student_manifest": str(args.student_manifest.resolve()),
        "teacher_manifest": str(args.teacher_manifest.resolve()) if args.teacher_manifest else None,
        "prompt_count": len(groups),
        "pairs_per_prompt": args.pairs_per_prompt,
        "pair_count": len(groups) * args.pairs_per_prompt,
        "metric_space": args.metric_space,
        "validated_latent_bank": str(validated_bank.resolve()) if validated_bank else None,
        "feature_models": feature_metadata,
        "spaces": spaces,
        "teacher_student": teacher_student or None,
    }
    if args.compute_statistics:
        payload["image_statistics"] = _statistics_metrics(student_records, groups)
    if args.compute_lpips:
        payload["lpips"] = _lpips_metrics(
            student_records,
            groups,
            device=args.device,
            net=args.lpips_net,
            batch_size=args.lpips_batch_size,
        )
    primary = spaces[args.metric_space]
    flat_metrics: dict[str, Any] = {
        "metrics/antithetic_corr_mean": primary["antithetic_corr"]["mean"],
        "metrics/antithetic_corr_std": primary["antithetic_corr"]["std"],
        "metrics/random_corr_mean": primary["random_corr"]["mean"],
        "metrics/centralized_anc_mean": primary["centralized_antithetic_corr"]["mean"],
        "metrics/corr_gap": primary["corr_gap"],
        "metrics/valid_count": primary["antithetic_corr"]["valid_count"],
    }
    if args.compute_statistics:
        statistic_aliases = {
            "brightness": "brightness",
            "vertical_brightness_imbalance": "contrast",
            "vertical_brightness_centroid": "centroid",
        }
        for source, target in statistic_aliases.items():
            statistic = payload["image_statistics"][source]
            flat_metrics[f"metrics/{target}_corr"] = statistic["pair_corr"]["mean"]
            flat_metrics[f"metrics/{target}_vrf"] = statistic["vrf_from_corr"]["mean"]
    if teacher_student:
        comparison = teacher_student[args.metric_space]
        for source, target in (
            ("teacher_anc", "teacher_anc"),
            ("student_anc", "student_anc"),
            ("anti_gap", "anti_gap"),
            ("pair_difference_mse", "pair_difference_error"),
            ("pair_mean_error", "pair_mean_error"),
            ("pointwise_error", "pointwise_error"),
        ):
            flat_metrics[f"metrics/{target}"] = comparison[source]["mean"]
    payload["flat_metrics"] = flat_metrics
    payload.update(flat_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [per_pair[key] for key in sorted(per_pair)]
    csv_path = args.output_dir / "antithetic_per_pair.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload["per_pair_csv"] = str(csv_path.resolve())
    output = args.output_dir / "antithetic_metrics.json"
    serializable = json_safe(payload)
    output.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=False))
    return output


def _generation_command(
    args: argparse.Namespace,
    *,
    mode: str,
    model: str,
    checkpoint: str | None,
    vae: str | None,
    output: Path,
    expected_count: int,
) -> list[str]:
    project_root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        str(project_root / "legacy" / "gad" / "generate_coco1k_benchmark.py"),
        "--mode",
        mode,
        "--model",
        model,
        "--latent-bank",
        str(args.output.resolve() / "latent_bank.pt"),
        "--output",
        str(output),
        "--batch-size",
        str(args.generation_batch_size),
        "--num-gpus",
        str(args.num_gpus),
        "--teacher-steps",
        str(args.teacher_steps),
        "--guidance",
        str(args.guidance),
        "--gad-steps",
        str(args.student_steps),
        "--total-steps",
        str(args.total_steps),
        "--expected-count",
        str(expected_count),
        "--resolution",
        str(args.resolution),
        "--latent-protocol",
        LATENT_PROTOCOL,
        "--evaluation-protocol",
        EVALUATION_PROTOCOL,
    ]
    if checkpoint:
        command.extend(["--checkpoint", checkpoint])
    if vae:
        command.extend(["--vae", vae])
    if args.metric_space == "latent":
        command.append("--save-final-latents")
    if args.resume:
        command.append("--resume")
    return command


def run_pixart(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    bank_path = args.output / "latent_bank.pt"
    if not bank_path.exists():
        prompts = read_prompts(args.prompts, args.prompt_count)
        bank = build_antithetic_bank(
            prompts,
            pairs_per_prompt=args.pairs_per_prompt,
            base_seed=args.seed,
            latent_size=args.resolution // 8,
            prompt_path=args.prompts,
            output=bank_path,
        )
    else:
        import torch

        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        if bank.get("protocol") != LATENT_PROTOCOL:
            raise ValueError("existing latent bank uses a different protocol")
        if int(bank.get("pairs_per_prompt", -1)) != args.pairs_per_prompt:
            raise ValueError("existing latent bank uses a different pairs-per-prompt")
        if int(bank.get("base_seed", -1)) != args.seed:
            raise ValueError("existing latent bank uses a different seed")
    expected_count = len(bank["records"])
    student_dir = args.output / "student_generation"
    subprocess.run(
        _generation_command(
            args,
            mode=args.student_mode,
            model=args.model,
            checkpoint=args.checkpoint,
            vae=args.vae,
            output=student_dir,
            expected_count=expected_count,
        ),
        check=True,
        cwd=project_root,
    )
    teacher_manifest = None
    if args.teacher_model:
        teacher_dir = args.output / "teacher_generation"
        subprocess.run(
            _generation_command(
                args,
                mode=args.teacher_mode,
                model=args.teacher_model,
                checkpoint=args.teacher_checkpoint,
                vae=args.teacher_vae,
                output=teacher_dir,
                expected_count=expected_count,
            ),
            check=True,
            cwd=project_root,
        )
        teacher_manifest = teacher_dir / "generation_manifest.jsonl"
    score(
        argparse.Namespace(
            student_manifest=student_dir / "generation_manifest.jsonl",
            teacher_manifest=teacher_manifest,
            output_dir=args.output / "metrics",
            pairs_per_prompt=args.pairs_per_prompt,
            metric_space=args.metric_space,
            feature_spaces=args.feature_spaces,
            dino_model=args.dino_model,
            clip_model=args.clip_model,
            device=args.device,
            dino_dtype=args.dino_dtype,
            clip_dtype=args.clip_dtype,
            feature_batch_size=args.feature_batch_size,
            compute_statistics=args.compute_statistics,
            compute_lpips=args.compute_lpips,
            lpips_net=args.lpips_net,
            lpips_batch_size=args.lpips_batch_size,
            validate_initial_noise=True,
        )
    )


def add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pairs-per-prompt", type=int, default=8)
    parser.add_argument("--metric-space", choices=("image", "latent"), default="image")
    parser.add_argument("--feature-spaces", default="dino,clip", help="Comma-separated dino,clip or none")
    parser.add_argument("--dino-model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dino-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--clip-dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--compute-statistics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--lpips-batch-size", type=int, default=16)
    parser.add_argument(
        "--validate-initial-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify exact z/-z and independent baselines against the generation latent bank.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bank = sub.add_parser("bank", help="Build a deterministic antithetic/random latent bank")
    bank.add_argument("--prompts", type=Path, required=True)
    bank.add_argument("--output", type=Path, required=True)
    bank.add_argument("--prompt-count", type=int, default=20)
    bank.add_argument("--pairs-per-prompt", type=int, default=8)
    bank.add_argument("--seed", type=int, default=42)
    bank.add_argument("--latent-size", type=int, default=64)

    score_parser = sub.add_parser("score", help="Score existing antithetic generation manifests")
    score_parser.add_argument("--student-manifest", type=Path, required=True)
    score_parser.add_argument("--teacher-manifest", type=Path)
    score_parser.add_argument("--output-dir", type=Path, required=True)
    add_score_arguments(score_parser)

    run = sub.add_parser("run-pixart", help="Build bank, reuse PixArt generation, then score")
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--prompt-count", type=int, default=20)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--model", required=True)
    run.add_argument("--student-mode", choices=("pixart", "tdm", "gad", "crra"), required=True)
    run.add_argument("--checkpoint")
    run.add_argument("--vae")
    run.add_argument("--teacher-model")
    run.add_argument("--teacher-mode", choices=("pixart", "tdm", "gad", "crra"), default="pixart")
    run.add_argument("--teacher-checkpoint")
    run.add_argument("--teacher-vae")
    run.add_argument("--resolution", type=int, default=512)
    run.add_argument("--teacher-steps", type=int, default=20)
    run.add_argument("--student-steps", type=int, default=4)
    run.add_argument("--total-steps", type=int, default=900)
    run.add_argument("--guidance", type=float, default=4.5)
    run.add_argument("--generation-batch-size", type=int, default=4)
    run.add_argument("--num-gpus", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    add_score_arguments(run)

    args = parser.parse_args()
    if getattr(args, "pairs_per_prompt", 2) < 2:
        parser.error("--pairs-per-prompt must be at least 2")
    if args.command == "bank":
        prompts = read_prompts(args.prompts, args.prompt_count)
        payload = build_antithetic_bank(
            prompts,
            pairs_per_prompt=args.pairs_per_prompt,
            base_seed=args.seed,
            latent_size=args.latent_size,
            prompt_path=args.prompts,
            output=args.output,
        )
        print(json.dumps({"output": str(args.output), "records": len(payload["records"])}, indent=2))
    elif args.command == "score":
        score(args)
    else:
        if args.student_mode != "pixart" and (not args.checkpoint or not args.vae):
            parser.error("distilled student mode requires --checkpoint and --vae")
        if args.teacher_model and args.teacher_mode != "pixart" and (
            not args.teacher_checkpoint or not args.teacher_vae
        ):
            parser.error("distilled teacher mode requires --teacher-checkpoint and --teacher-vae")
        run_pixart(args)


if __name__ == "__main__":
    main()
