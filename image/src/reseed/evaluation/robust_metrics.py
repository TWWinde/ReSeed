#!/usr/bin/env python3
"""Fault-isolated HPSv2, LAION-AeS, and standard FID evaluation.

The suite process never imports metric dependencies.  Each metric is executed
in a fresh child process, writes its own artifacts, and has independent logs.
Consequently an import error, missing checkpoint, CUDA OOM, or malformed result
in one metric cannot prevent the remaining metrics from running.

Metric implementations are delegated to published upstream projects:

* HPSv2: tgxs002/HPSv2, v2.1 checkpoint.
* AeS: PyIQA's ``laion_aes`` adapter for the LAION CLIP+MLP predictor.
* FID: mseitzer/pytorch-fid with the standard 2048-D FID Inception features.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 2
METRICS = ("hps", "aes", "fid")
DEFAULT_HPS_ROOT = Path("/home/dh2121/code/eval_external/HPSv2")
DEFAULT_HPS_CHECKPOINT = Path(
    "/home/dh2121/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/"
    "697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2.1_compressed.pt"
)
DEFAULT_PYIQA_ROOT = Path("/home/dh2121/code/eval_external/IQA-PyTorch")
DEFAULT_PYTHON_DEPS = Path("/home/dh2121/code/eval_external/python_deps")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {result!r}")
    return result


def _summary(values: Sequence[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("metric scores must be a non-empty finite vector")
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std": std,
        "stderr": std / math.sqrt(array.size),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def _load_records(manifest: Path, expected_count: int) -> list[dict[str, Any]]:
    manifest = manifest.expanduser().resolve()
    records: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{manifest}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{manifest}:{line_number}: record is not an object")
            records.append(record)
    if len(records) != expected_count:
        raise ValueError(f"expected {expected_count} records, found {len(records)}")
    cases: set[int] = set()
    image_paths: set[Path] = set()
    for index, record in enumerate(records):
        for field in ("case_number", "prompt", "image_path"):
            if field not in record:
                raise ValueError(f"record {index} is missing {field!r}")
        case = int(record["case_number"])
        if case in cases:
            raise ValueError(f"duplicate case_number={case}")
        cases.add(case)
        if not isinstance(record["prompt"], str) or not record["prompt"].strip():
            raise ValueError(f"record {index} has an empty prompt")
        image = Path(record["image_path"]).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"missing generated image: {image}")
        expected_sha = record.get("image_sha256")
        if expected_sha is not None and _sha256(image) != expected_sha:
            raise ValueError(f"generated image SHA256 mismatch: {image}")
        expected_width = record.get("image_width")
        expected_height = record.get("image_height")
        if (expected_width is None) != (expected_height is None):
            raise ValueError(f"generated image manifest has incomplete dimensions: {image}")
        if expected_width is not None:
            from PIL import Image

            with Image.open(image) as opened:
                if opened.size != (int(expected_width), int(expected_height)):
                    raise ValueError(f"generated image dimensions mismatch: {image}")
        if image in image_paths:
            raise ValueError(f"generated image is reused by multiple records: {image}")
        image_paths.add(image)
        record["case_number"] = case
        record["image_path"] = str(image)
    return sorted(records, key=lambda item: item["case_number"])


def _load_stable_records(
    manifest: Path, expected_count: int
) -> tuple[list[dict[str, Any]], str]:
    manifest = manifest.expanduser().resolve()
    before = _sha256(manifest)
    records = _load_records(manifest, expected_count)
    after = _sha256(manifest)
    if before != after:
        raise ValueError("generation manifest changed while it was being loaded")
    return records, after


def _load_generation_provenance(
    manifest: Path,
    records: Sequence[dict[str, Any]],
    expected_count: int,
    evaluation_protocol: str,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind every extended metric artifact to the exact generation contract."""

    manifest = manifest.expanduser().resolve()
    manifest_sha256 = _sha256(manifest)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError("generation manifest changed before provenance capture")
    protocol_path = manifest.parent / "generation_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing generation protocol beside manifest: {protocol_path}")
    protocol_raw = protocol_path.read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    protocol = json.loads(protocol_raw)
    if protocol.get("evaluation_protocol") != evaluation_protocol:
        raise ValueError(
            "generation/metric protocol mismatch: "
            f"{protocol.get('evaluation_protocol')!r} != {evaluation_protocol!r}"
        )
    if int(protocol.get("num_prompts", -1)) != expected_count:
        raise ValueError("generation protocol prompt count does not match --expected-count")
    modes = {record.get("mode") for record in records}
    if modes != {protocol.get("mode")}:
        raise ValueError(f"manifest modes {modes} do not match generation protocol")

    request_path = manifest.parent / "generation_request.json"
    request_sha256 = None
    declared_request_sha256 = protocol.get("generation_request_sha256")
    if declared_request_sha256 is not None and not request_path.is_file():
        raise FileNotFoundError(
            f"generation protocol declares a request SHA but the request is missing: {request_path}"
        )
    if request_path.is_file():
        request_raw = request_path.read_bytes()
        request = json.loads(request_raw)
        for key in (
            "evaluation_protocol",
            "mode",
            "num_prompts",
            "benchmark_num_prompts",
            "model",
            "vae",
            "checkpoint",
            "student",
            "latent_bank",
            "latent_bank_sha256",
            "latent_bank_identity",
            "student_steps",
            "student_total_steps",
        ):
            if request.get(key) != protocol.get(key):
                raise ValueError(f"generation request/protocol mismatch for {key}")
        request_sha256 = hashlib.sha256(request_raw).hexdigest()
        if declared_request_sha256 != request_sha256:
            raise ValueError("generation request SHA does not match generation protocol")
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "request_path": str(request_path) if request_path.is_file() else None,
        "request_sha256": request_sha256,
    }


def _prepend_import_root(root: Path, expected: Path) -> None:
    root = root.expanduser().resolve()
    if not (root / expected).is_file():
        raise FileNotFoundError(f"missing upstream implementation: {root / expected}")
    sys.path.insert(0, str(root))


def _prepare_hps_dependencies(root: Path, python_deps: Path) -> tuple[Any, Path]:
    """Expose the pinned external dependency root before importing HPSv2."""

    root = root.expanduser().resolve()
    python_deps = python_deps.expanduser().resolve()
    requirements = root / "requirements.txt"
    if python_deps.is_dir():
        sys.path.insert(0, str(python_deps))
    _prepend_import_root(root, Path("hpsv2") / "img_score.py")
    if not requirements.is_file():
        raise FileNotFoundError(f"missing pinned HPSv2 requirements: {requirements}")
    try:
        import clint
    except ImportError as error:
        raise ImportError(
            "HPSv2 requires clint==0.5.1; install it in the active environment "
            "or under --python-deps"
        ) from error
    return clint, requirements


def _write_per_image_csv(
    path: Path,
    records: Sequence[dict[str, Any]],
    metric_name: str,
    scores: Sequence[float],
) -> None:
    if len(records) != len(scores):
        raise AssertionError("record/score count mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    columns = [
        "case_number",
        "coco_id",
        "evaluation_seed",
        "prompt",
        "image_path",
        "mode",
        metric_name,
    ]
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record, score in zip(records, scores):
                writer.writerow(
                    {
                        **{column: record.get(column, "") for column in columns},
                        metric_name: f"{_finite(score, metric_name):.10f}",
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _image_batches(records: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), batch_size):
        yield list(records[start : start + batch_size])


def _run_hps(args: argparse.Namespace, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from PIL import Image

    root = args.hps_root.expanduser().resolve()
    checkpoint = args.hps_checkpoint.expanduser().resolve()
    python_deps = args.python_deps.expanduser().resolve()
    clint, requirements = _prepare_hps_dependencies(root, python_deps)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing pinned HPSv2 checkpoint: {checkpoint}")
    # These are the exact model constructors used by upstream hpsv2/img_score.py.
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    device = torch.device(args.device)
    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14",
        "laion2B-s32B-b79K",
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    tokenizer = get_tokenizer("ViT-H-14")
    scores: list[float] = []
    for batch_index, batch in enumerate(_image_batches(records, args.batch_size), 1):
        images = []
        for record in batch:
            with Image.open(record["image_path"]) as image:
                images.append(preprocess(image.convert("RGB")))
        image_tensor = torch.stack(images).to(device, non_blocking=True)
        text_tensor = tokenizer([record["prompt"] for record in batch]).to(
            device, non_blocking=True
        )
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            outputs = model(image_tensor, text_tensor)
            values = (outputs["image_features"] * outputs["text_features"]).sum(dim=-1)
        scores.extend(values.float().cpu().tolist())
        print(f"[hps] {min(batch_index * args.batch_size, len(records))}/{len(records)}", flush=True)
    csv_path = args.output / "hpsv2_scores.csv"
    _write_per_image_csv(csv_path, records, "hpsv2_score", scores)
    return {
        "metric": "HPSv2",
        "definition": "paired prompt-image score from the official HPSv2 v2.1 model",
        "higher_is_better": True,
        "summary": _summary(scores),
        "scores_csv": str(csv_path),
        "upstream": {
            "repository": "https://github.com/tgxs002/HPSv2",
            "root": str(root),
            "git_head": _git_head(root),
            "implementation": str(root / "hpsv2" / "img_score.py"),
            "implementation_sha256": _sha256(root / "hpsv2" / "img_score.py"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "requirements_sha256": _sha256(requirements),
            "python_deps": str(python_deps),
            "clint_module": str(Path(clint.__file__).resolve()),
            "clint_module_sha256": _sha256(Path(clint.__file__).resolve()),
            "version": "v2.1",
        },
    }


def _run_aes(args: argparse.Namespace, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    root = args.pyiqa_root.expanduser().resolve()
    python_deps = args.python_deps.expanduser().resolve()
    if (python_deps / "pyiqa" / "__init__.py").is_file():
        sys.path.insert(0, str(python_deps))
    elif root.exists():
        _prepend_import_root(root, Path("pyiqa") / "__init__.py")
    try:
        import pyiqa
    except ImportError as error:
        raise ImportError(
            "PyIQA is required for AeS; install it or pass --pyiqa-root to its official checkout"
        ) from error

    device = torch.device(args.device)
    model = pyiqa.create_metric("laion_aes", device=device)
    scores: list[float] = []
    for batch_index, batch in enumerate(_image_batches(records, args.batch_size), 1):
        tensors = []
        for record in batch:
            with Image.open(record["image_path"]) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        images = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode():
            values = model(images).reshape(-1)
        scores.extend(values.float().cpu().tolist())
        print(f"[aes] {min(batch_index * args.batch_size, len(records))}/{len(records)}", flush=True)
    csv_path = args.output / "laion_aes_scores.csv"
    _write_per_image_csv(csv_path, records, "laion_aes", scores)
    try:
        version = importlib.metadata.version("pyiqa")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "metric": "LAION-AeS",
        "definition": "PyIQA laion_aes (CLIP ViT-L/14 plus LAION aesthetic MLP)",
        "higher_is_better": True,
        "summary": _summary(scores),
        "scores_csv": str(csv_path),
        "upstream": {
            "repository": "https://github.com/chaofengc/IQA-PyTorch",
            "root": str(root) if root.exists() else None,
            "git_head": _git_head(root) if root.exists() else None,
            "version": version,
            "base_predictor": "https://github.com/christophschuhmann/improved-aesthetic-predictor",
        },
    }


def _safe_link_view(path: Path, mapping: Sequence[tuple[str, Path]]) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"refusing to replace unsafe metric view: {path}")
        shutil.rmtree(path)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    try:
        for name, source in mapping:
            destination = staging / name
            destination.symlink_to(source)
        os.replace(staging, path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _discard_link_view(path: Path) -> None:
    """Remove a symlink view once the metric that needed it has been computed.

    These views exist only to hand pytorch-fid two directories whose entries it
    can enumerate positionally; nothing reads them afterwards, and the metric
    payload records the manifest and reference hashes that identify the images.
    Each FID run builds 10k of them, which is 10k inodes for a few megabytes, so
    leaving them behind exhausts a file-count quota long before a size quota.

    Every entry is verified to be a symlink first. The view is always built by
    _safe_link_view, so anything else means a caller has started putting real
    files here and deleting the tree would destroy data rather than links.
    """
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"refusing to remove unsafe metric view: {path}")
    for entry in path.rglob("*"):
        if entry.is_dir() and not entry.is_symlink():
            continue
        if not entry.is_symlink():
            raise ValueError(f"metric view holds a real file, refusing to remove: {entry}")
    shutil.rmtree(path)


def _run_fid(args: argparse.Namespace, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if args.fid_reference_dir is None:
        raise ValueError("FID requires --fid-reference-dir with official COCO2017 val images")
    reference_root = args.fid_reference_dir.expanduser().resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(f"missing FID reference directory: {reference_root}")

    import torch

    dependency_root = args.python_deps.expanduser().resolve()
    vendored = dependency_root / "pytorch_fid" / "fid_score.py"
    if vendored.is_file():
        _prepend_import_root(dependency_root, Path("pytorch_fid") / "fid_score.py")
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import FID_WEIGHTS_URL
    except ImportError as error:
        raise ImportError(
            "could not import mseitzer/pytorch-fid; install pinned pytorch-fid==0.3.0"
        ) from error

    generated = [
        (f"{dense_index:05d}.png", Path(record["image_path"]))
        for dense_index, record in enumerate(records)
    ]
    reference_mode = getattr(args, "fid_reference_mode", "matched-records")
    references: list[tuple[str, Path]] = []
    if reference_mode == "matched-records":
        reference_sources: set[Path] = set()
        for dense_index, record in enumerate(records):
            file_name = record.get("file_name")
            if not file_name:
                coco_id = record.get("coco_id")
                if coco_id in (None, ""):
                    raise ValueError("FID records need file_name or coco_id for the real COCO image")
                file_name = f"{int(coco_id):012d}.jpg"
            relative = Path(str(file_name))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe COCO file_name: {file_name!r}")
            source = (reference_root / relative).resolve()
            try:
                source.relative_to(reference_root)
            except ValueError as error:
                raise ValueError(f"COCO reference escaped root: {source}") from error
            if not source.is_file():
                raise FileNotFoundError(f"missing COCO reference image: {source}")
            if source in reference_sources:
                raise ValueError(f"COCO reference image is duplicated: {source}")
            reference_sources.add(source)
            references.append((f"{dense_index:05d}{source.suffix.lower()}", source))
        fid_reference_path = reference_view = args.output / "fid_views" / "coco2017_val"
    elif reference_mode == "full-directory":
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        reference_files = sorted(
            path.resolve()
            for path in reference_root.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        )
        expected_reference_count = getattr(args, "fid_reference_expected_count", None)
        if expected_reference_count is not None and len(reference_files) != expected_reference_count:
            raise ValueError(
                "FID reference count mismatch: "
                f"expected {expected_reference_count}, found {len(reference_files)}"
            )
        if not reference_files:
            raise ValueError(f"FID reference directory has no supported images: {reference_root}")
        references = [(path.name, path) for path in reference_files]
        fid_reference_path = reference_root
        reference_view = None
    else:  # pragma: no cover - argparse enforces this for CLI callers.
        raise ValueError(f"unknown FID reference mode: {reference_mode}")

    view_root = args.output / "fid_views"
    view_root.mkdir(parents=True, exist_ok=True)
    generated_view = view_root / "generated"
    _safe_link_view(generated_view, generated)
    if reference_view is not None:
        _safe_link_view(reference_view, references)
    # Official COCO2017 validation images have different native H/W.  The
    # upstream pytorch-fid dataset applies ToTensor without resizing, so a
    # DataLoader batch larger than one cannot collate those references.  A
    # batch of one preserves upstream preprocessing and the standard FID
    # feature extractor while avoiding any custom crop/resize policy.
    fid_batch_size = 1
    try:
        score = fid_score.calculate_fid_given_paths(
            [str(generated_view), str(fid_reference_path)],
            batch_size=fid_batch_size,
            device=torch.device(args.device),
            dims=2048,
            num_workers=args.num_workers,
        )
    finally:
        # Drop the link farm even when FID raises: it is rebuilt from the
        # manifest on the next attempt, and a failed run must not leave 10k
        # inodes behind.
        _discard_link_view(view_root)
    score = _finite(score, "pytorch-fid")
    implementation = Path(fid_score.__file__).resolve()
    try:
        version = importlib.metadata.version("pytorch-fid")
    except importlib.metadata.PackageNotFoundError:
        version = None
    weights_name = Path(urllib.parse.urlparse(FID_WEIGHTS_URL).path).name
    weights_path = Path(torch.hub.get_dir()).resolve() / "checkpoints" / weights_name
    return {
        "metric": "FID",
        "metric_protocol": (
            "standard_fid_pytorch_fid_0.3.0_generated_vs_full_coco2017_val5k_v1"
            if reference_mode == "full-directory"
            else "standard_fid_pytorch_fid_0.3.0_coco2017_val5k_v1"
        ),
        "definition": "mseitzer/pytorch-fid standard 2048-D Inception-V3 FID",
        "higher_is_better": False,
        "fid": score,
        "generated_count": len(generated),
        "reference_count": len(references),
        "reference_mode": reference_mode,
        "reference_root": str(reference_root),
        "reference_file_names_sha256": hashlib.sha256(
            "\n".join(str(source.relative_to(reference_root)) for _, source in references).encode()
        ).hexdigest(),
        "upstream": {
            "repository": "https://github.com/mseitzer/pytorch-fid",
            "version": version,
            "implementation": str(implementation),
            "implementation_sha256": _sha256(implementation),
            "feature_dimensions": 2048,
            "batch_size": fid_batch_size,
            "batch_size_reason": "native COCO2017 images have variable spatial dimensions",
            "preprocessing": "official pytorch-fid ImagePathDataset and FID InceptionV3",
            "weights_url": FID_WEIGHTS_URL,
            "weights_path": str(weights_path) if weights_path.is_file() else None,
            "weights_sha256": _sha256(weights_path) if weights_path.is_file() else None,
        },
    }


def _worker(args: argparse.Namespace) -> int:
    started = time.monotonic()
    records, manifest_sha256 = _load_stable_records(args.manifest, args.expected_count)
    generation_provenance = _load_generation_provenance(
        args.manifest,
        records,
        args.expected_count,
        args.evaluation_protocol,
        manifest_sha256,
    )
    runners = {"hps": _run_hps, "aes": _run_aes, "fid": _run_fid}
    payload = runners[args.metric](args, records)
    final_records, final_manifest_sha256 = _load_stable_records(
        args.manifest, args.expected_count
    )
    if final_manifest_sha256 != manifest_sha256 or final_records != records:
        raise ValueError("generation manifest or images changed while metric was running")
    final_generation_provenance = _load_generation_provenance(
        args.manifest,
        final_records,
        args.expected_count,
        args.evaluation_protocol,
        final_manifest_sha256,
    )
    if final_generation_provenance != generation_provenance:
        raise ValueError("generation provenance changed while metric was running")
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "metric_id": args.metric,
            "created_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "evaluation_protocol": args.evaluation_protocol,
            "manifest": str(args.manifest.expanduser().resolve()),
            "manifest_sha256": manifest_sha256,
            "generation_provenance": generation_provenance,
            "n": len(records),
        }
    )
    _atomic_json(args.result, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _worker_command(args: argparse.Namespace, metric: str, result: Path) -> list[str]:
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "worker",
        "--metric",
        metric,
        "--manifest",
        str(args.manifest),
        "--output",
        str(args.output),
        "--result",
        str(result),
        "--expected-count",
        str(args.expected_count),
        "--evaluation-protocol",
        args.evaluation_protocol,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--hps-root",
        str(args.hps_root),
        "--hps-checkpoint",
        str(args.hps_checkpoint),
        "--pyiqa-root",
        str(args.pyiqa_root),
        "--python-deps",
        str(args.python_deps),
    ]
    if args.fid_reference_dir is not None:
        command.extend(["--fid-reference-dir", str(args.fid_reference_dir)])
    command.extend(["--fid-reference-mode", args.fid_reference_mode])
    if args.fid_reference_expected_count is not None:
        command.extend(
            ["--fid-reference-expected-count", str(args.fid_reference_expected_count)]
        )
    return command


def run_suite(args: argparse.Namespace) -> int:
    if args.expected_count <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("expected-count/batch-size must be positive; num-workers non-negative")
    records, manifest_sha256 = _load_stable_records(args.manifest, args.expected_count)
    generation_provenance = _load_generation_provenance(
        args.manifest,
        records,
        args.expected_count,
        args.evaluation_protocol,
        manifest_sha256,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "extended_metric_status.json"
    requested = list(dict.fromkeys(args.metrics))
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "evaluation_protocol": args.evaluation_protocol,
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_sha256": manifest_sha256,
        "generation_provenance": generation_provenance,
        "expected_count": args.expected_count,
        "requested_metrics": requested,
        "metrics": {},
    }
    _atomic_json(status_path, status)
    for metric in requested:
        started = time.monotonic()
        result_path = args.output / f"metrics_{metric}.json"
        stdout_path = args.output / f"metrics_{metric}.stdout.log"
        stderr_path = args.output / f"metrics_{metric}.stderr.log"
        command = _worker_command(args, metric, result_path)
        entry = {
            "status": "running",
            "started_utc": _utc_now(),
            "command": command,
            "result": str(result_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        status["metrics"][metric] = entry
        status["updated_utc"] = _utc_now()
        _atomic_json(status_path, status)
        completed: subprocess.CompletedProcess | None = None
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"worker exited with code {completed.returncode}")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("status") != "completed" or payload.get("metric_id") != metric:
                raise ValueError("worker result did not satisfy the metric artifact contract")
            if payload.get("generation_provenance") != generation_provenance:
                raise ValueError("worker generation provenance differs from suite contract")
            expected_worker_identity = {
                "evaluation_protocol": args.evaluation_protocol,
                "manifest": str(args.manifest.expanduser().resolve()),
                "manifest_sha256": status["manifest_sha256"],
                "n": args.expected_count,
            }
            mismatches = {
                key: {"actual": payload.get(key), "expected": expected}
                for key, expected in expected_worker_identity.items()
                if payload.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"worker input identity differs from suite contract: {mismatches}")
            entry.update(
                {
                    "status": "completed",
                    "result_sha256": _sha256(result_path),
                    "returncode": completed.returncode,
                }
            )
        except Exception as error:  # isolation boundary: continue to the next metric
            entry.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "returncode": completed.returncode if completed is not None else None,
                }
            )
        entry["finished_utc"] = _utc_now()
        entry["elapsed_seconds"] = time.monotonic() - started
        status["updated_utc"] = _utc_now()
        _atomic_json(status_path, status)
        print(f"[{metric}] {entry['status']}: {entry.get('error', entry['result'])}", flush=True)
    completed_count = sum(
        entry["status"] == "completed" for entry in status["metrics"].values()
    )
    failed_count = len(requested) - completed_count
    status["summary"] = {"completed": completed_count, "failed": failed_count}
    status["status"] = (
        "completed" if failed_count == 0 else "partial" if completed_count else "failed"
    )
    status["updated_utc"] = _utc_now()
    _atomic_json(status_path, status)
    print(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True))
    if args.require_all:
        return 0 if failed_count == 0 else 1
    return 0 if completed_count > 0 else 1


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--evaluation-protocol", default="coco2017_val5k_paired_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--hps-root", type=Path, default=DEFAULT_HPS_ROOT)
    parser.add_argument("--hps-checkpoint", type=Path, default=DEFAULT_HPS_CHECKPOINT)
    parser.add_argument("--pyiqa-root", type=Path, default=DEFAULT_PYIQA_ROOT)
    parser.add_argument("--python-deps", type=Path, default=DEFAULT_PYTHON_DEPS)
    parser.add_argument(
        "--fid-reference-dir",
        type=Path,
        help="Directory containing official COCO2017 val images. Required only by FID.",
    )
    parser.add_argument(
        "--fid-reference-mode",
        choices=("matched-records", "full-directory"),
        default="matched-records",
        help=(
            "Use manifest-matched COCO images (default), or compare generated images "
            "against every supported image in the reference directory."
        ),
    )
    parser.add_argument(
        "--fid-reference-expected-count",
        type=int,
        help="Fail unless full-directory FID finds exactly this many reference images.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    suite = subparsers.add_parser("suite", help="Run metrics in isolated child processes")
    _common(suite)
    suite.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    suite.add_argument("--python", default=sys.executable)
    suite.add_argument(
        "--require-all",
        action="store_true",
        help="Return non-zero when any requested metric fails (default: only if all fail).",
    )
    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    _common(worker)
    worker.add_argument("--metric", choices=METRICS, required=True)
    worker.add_argument("--result", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "worker":
            return _worker(args)
        return run_suite(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
