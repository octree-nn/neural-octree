"""Offline image and metadata output.

Everything is written directly to disk; there is no screenshot mechanism.
PNG output applies tone mapping and the exact sRGB encode.  A linear ``.npy``
buffer is always available.  Metadata is canonical JSON (sorted keys, compact
separators) and carries a SHA-256 config hash so a render can be tied back to
the exact configuration.  EXR is optional: if OpenEXR is missing, the writer
raises an actionable error rather than silently degrading.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .color import linear_to_display

__all__ = [
    "canonical_json",
    "config_hash",
    "write_metadata",
    "save_linear_npy",
    "linear_to_uint8",
    "save_png",
    "save_exr",
    "save_image",
]


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def canonical_json(data: Any) -> str:
    """Deterministic JSON text: sorted keys, compact separators, newline."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(config: Any) -> str:
    """SHA-256 of the canonical JSON encoding of ``config``."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def write_metadata(path: str | Path, metadata: dict[str, Any], *, config: Any | None = None) -> Path:
    """Write canonical JSON metadata, embedding a config hash when given."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    if config is not None:
        payload["config_hash"] = config_hash(config)
        payload.setdefault("config", config)
    output.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return output


def save_linear_npy(path: str | Path, image: Tensor | np.ndarray) -> Path:
    """Save a linear float buffer as ``.npy``."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, _numpy(image).astype(np.float32))
    return output


def linear_to_uint8(
    image: Tensor | np.ndarray,
    *,
    mode: str = "aces",
    exposure: float = 1.0,
    already_srgb: bool = False,
) -> np.ndarray:
    """Convert a linear image buffer to 8-bit sRGB/display values."""
    array = _numpy(image).astype(np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {array.shape}")
    if already_srgb:
        display = array
    else:
        display = _numpy(
            linear_to_display(torch.as_tensor(array), mode=mode, exposure=exposure, encoded=True)
        )
    return (np.clip(display, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_png(
    path: str | Path,
    image: Tensor | np.ndarray,
    *,
    mode: str = "aces",
    exposure: float = 1.0,
    already_srgb: bool = False,
) -> Path:
    """Tone-map and sRGB-encode a linear image, then write a PNG."""
    from PIL import Image

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pixels = linear_to_uint8(image, mode=mode, exposure=exposure, already_srgb=already_srgb)
    Image.fromarray(pixels, mode="RGB").save(output)
    return output


def _openexr_modules():
    try:
        import OpenEXR  # type: ignore

        try:
            import Imath  # type: ignore

            return OpenEXR, Imath
        except Exception:
            return OpenEXR, getattr(OpenEXR, "Imath", None)
    except Exception as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "EXR output requires the optional 'OpenEXR' package (and Imath). "
            "Install it with `pip install OpenEXR` or write a .npy/.png instead."
        ) from error


def save_exr(path: str | Path, image: Tensor | np.ndarray) -> Path:
    """Write a linear float32 EXR, or fail with an actionable message."""
    OpenEXR, Imath = _openexr_modules()
    if Imath is None:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "EXR output requires the Imath bindings shipped with OpenEXR; "
            "write a .npy/.png instead or install a complete OpenEXR build."
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = _numpy(image).astype(np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {array.shape}")
    height, width = array.shape[:2]
    header = OpenEXR.Header(width, height)
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    header["channels"] = {
        channel: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for channel in "RGB"
    }
    try:  # pragma: no cover - exercises the installed OpenEXR bindings
        output_file = OpenEXR.OutputFile(str(output), header)
        output_file.writePixels(
            {
                "R": array[:, :, 0].tobytes(),
                "G": array[:, :, 1].tobytes(),
                "B": array[:, :, 2].tobytes(),
            }
        )
        output_file.close()
    except Exception as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"OpenEXR was importable but writing {output} failed: {error}. "
            "Write a .npy/.png instead or check the OpenEXR build."
        ) from error
    return output


def save_image(path: str | Path, image: Tensor | np.ndarray, **kwargs) -> Path:
    """Dispatch on the output suffix (``.png``, ``.npy``/``.npz``, ``.exr``)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        return save_png(path, image, **kwargs)
    if suffix in (".npy", ".npz"):
        return save_linear_npy(path, image)
    if suffix == ".exr":
        return save_exr(path, image)
    raise ValueError(f"Unsupported output suffix {suffix!r}; use .png, .npy or .exr")
