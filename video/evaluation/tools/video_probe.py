# SPDX-License-Identifier: Apache-2.0
"""Read a video's geometry through whichever backend this machine has.

Shared by the smoke test and the generation pipeline so both judge a file the
same way. ffprobe is not guaranteed on a compute node and imageio may lack the
pyav plugin, so try in order and report every failure if none works.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _probe_ffprobe(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
        "-of", "json", str(path),
    ]
    out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    st = out["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "frames": int(st["nb_read_frames"]),
        "fps": float(num) / float(den),
        "source": "ffprobe",
    }


def _probe_imageio(path: Path, plugin: Optional[str]) -> Dict[str, Any]:
    import imageio.v3 as iio

    kwargs = {"plugin": plugin} if plugin else {}
    frames = iio.imread(path, **kwargs)
    if frames.ndim != 4:
        raise ValueError(f"expected [T,H,W,C], got {frames.shape}")
    try:
        fps = float(iio.immeta(path, **kwargs).get("fps", 0.0))
    except Exception:  # noqa: BLE001 -- metadata optional, geometry is not
        fps = 0.0
    return {
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "frames": int(frames.shape[0]),
        "fps": fps,
        "source": f"imageio:{plugin or 'default'}",
    }


def probe_video(path: Path) -> Dict[str, Any]:
    errors = []
    if shutil.which("ffprobe"):
        try:
            return _probe_ffprobe(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ffprobe: {exc}")
    for plugin in ("pyav", "FFMPEG", None):
        try:
            return _probe_imageio(path, plugin)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"imageio[{plugin or 'default'}]: {type(exc).__name__}: {exc}")
    raise RuntimeError("no usable video backend -- " + " | ".join(errors))


def validate_video(path: Path, width: int, height: int, frames: int, fps: float,
                   fps_tol: float = 0.05) -> tuple[bool, str]:
    """Return (ok, reason). A file that cannot be probed is invalid, not skipped."""
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty file"
    try:
        info = probe_video(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {exc}"
    problems = []
    if (info["width"], info["height"]) != (width, height):
        problems.append(f"geometry {info['width']}x{info['height']} != {width}x{height}")
    if info["frames"] != frames:
        problems.append(f"frames {info['frames']} != {frames}")
    if info["fps"] > 0 and abs(info["fps"] - fps) > fps_tol:
        problems.append(f"fps {info['fps']:.3f} != {fps}")
    return (not problems), ("ok" if not problems else "; ".join(problems))
