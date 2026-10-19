from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


@dataclass(frozen=True)
class RenderOutput:
    position: Tensor
    normal: Tensor
    color: Tensor
    hit: Tensor
    scale: int

    def float_tiles(self) -> dict[str, Tensor]:
        hit = self.hit.unsqueeze(-1).expand_as(self.color).float()
        return {
            "position": (self.position / self.scale).clamp(0.0, 1.0),
            "normal": ((self.normal + 1.0) * 0.5).clamp(0.0, 1.0) * hit,
            "color": self.color.clamp(0.0, 1.0),
            "hit": hit,
        }

    def uint8_tiles(self) -> dict[str, Tensor]:
        return {
            name: (value * 255.0).round().to(torch.uint8)
            for name, value in self.float_tiles().items()
        }

    def composite(self, *, include_color: bool = True) -> Tensor:
        tiles = self.uint8_tiles()
        names = ["position", "normal"]
        if include_color:
            names.append("color")
        names.append("hit")
        return torch.cat([tiles[name] for name in names], dim=1)


def _save_tensor_image(path: Path, image: Tensor) -> None:
    array = image.detach().cpu().numpy()
    Image.fromarray(np.asarray(array)).save(path)


def save_render_output(
    output: RenderOutput,
    directory: str | Path,
    stem: str,
    *,
    save_composite: bool = True,
    include_color: bool = True,
) -> dict[str, Path]:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, tile in output.uint8_tiles().items():
        if name == "color" and not include_color:
            continue
        path = output_dir / f"{stem}_{name}.png"
        _save_tensor_image(path, tile)
        paths[name] = path
    if save_composite:
        path = output_dir / f"{stem}_composite.png"
        _save_tensor_image(path, output.composite(include_color=include_color))
        paths["composite"] = path
    return paths
