from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_image_tensor(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        array = np.array(image)
    return torch.from_numpy(array)


def load_exr_tensor(path: str | Path) -> Tensor:
    try:
        import OpenEXR
    except ImportError as error:
        raise RuntimeError(
            "OpenEXR is required when DATA.normal_source is 'exr'"
        ) from error
    channels = OpenEXR.File(str(path)).channels()
    if "RGB" not in channels:
        raise ValueError(f"EXR image does not contain an RGB channel: {path}")
    array = np.asarray(channels["RGB"].pixels, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"EXR RGB shape must be HxWx3, got {array.shape} for {path}")
    return torch.from_numpy(array.copy())


class PointCloudCache:
    """Small per-process LRU cache for object-level point clouds."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._items: OrderedDict[Path, tuple[Tensor, Tensor]] = OrderedDict()

    def load(self, path: str | Path) -> tuple[Tensor, Tensor]:
        resolved = Path(path).resolve()
        if resolved in self._items:
            self._items.move_to_end(resolved)
            points, colors = self._items[resolved]
            return points.clone(), colors.clone()
        points, colors = torch.load(resolved, map_location="cpu", weights_only=True)
        points = torch.as_tensor(points, dtype=torch.float32).contiguous()
        colors = torch.as_tensor(colors, dtype=torch.float32).contiguous()
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Point positions must have shape [N, 3], got {tuple(points.shape)}")
        if colors.shape != points.shape:
            raise ValueError(
                f"Point colors must match positions, got {tuple(colors.shape)} vs {tuple(points.shape)}"
            )
        if self.capacity > 0:
            self._items[resolved] = (points, colors)
            self._items.move_to_end(resolved)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return points.clone(), colors.clone()
