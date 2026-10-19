"""Deterministic material resolution for mesh and NeuOct primitives.

The resolver never invents non-deterministic values: every missing field falls
back to a documented constant.  Mesh materials prefer glTF PBR fields exposed
by trimesh (``baseColorFactor``, ``metallicFactor``, ``roughnessFactor``) and
otherwise use the same defaults as a NeuOct primitive.  NeuOct defaults are the
audited ones: ``roughness=0.5``, ``metallic=0``, ``specular=0.04``,
``transmission=0``, ``ior=1.5``, ``thin_walled=True``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .color import srgb_to_linear
from .types import Material

NEUOCT_DEFAULTS: dict[str, Any] = {
    "roughness": 0.5,
    "metallic": 0.0,
    "specular": 0.04,
    "transmission": 0.0,
    "ior": 1.5,
    "thin_walled": True,
}

MESH_DEFAULTS: dict[str, Any] = dict(NEUOCT_DEFAULTS)


def _read(container: object, names: tuple[str, ...]) -> object | None:
    for name in names:
        if container is None:
            return None
        value = getattr(container, name, None)
        if value is not None:
            return value
    return None


def _as_color(value: object, default: tuple[float, float, float]) -> torch.Tensor:
    if value is None:
        return torch.tensor(default, dtype=torch.float32)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 4:
        array = array[:3]
    if array.size != 3:
        return torch.tensor(default, dtype=torch.float32)
    if float(np.nanmax(array)) > 1.0 + 1.0e-6:
        array = array / 255.0
    return torch.tensor(array, dtype=torch.float32).clamp(0.0, 1.0)


class MaterialResolver:
    """Build :class:`Material` instances from scene descriptions.

    ``mesh_defaults`` and ``neuoct_defaults`` let a config override the audited
    defaults globally; per-primitive ``overrides`` take precedence in turn.
    """

    def __init__(
        self,
        *,
        mesh_defaults: dict[str, Any] | None = None,
        neuoct_defaults: dict[str, Any] | None = None,
    ) -> None:
        self.mesh_defaults = {**MESH_DEFAULTS, **(mesh_defaults or {})}
        self.neuoct_defaults = {**NEUOCT_DEFAULTS, **(neuoct_defaults or {})}

    def resolve_neuoct(self, overrides: dict[str, Any] | None = None) -> Material:
        values = {**self.neuoct_defaults, **(overrides or {})}
        base_color = _as_color(values.pop("base_color", None), (1.0, 1.0, 1.0))
        emission = _as_color(values.pop("emission", None), (0.0, 0.0, 0.0))
        texture = values.pop("texture", None)
        allowed = {
            "roughness",
            "metallic",
            "specular",
            "transmission",
            "ior",
            "thin_walled",
            "base_color_srgb",
        }
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown NeuOct material overrides: {sorted(unknown)}")
        return Material(
            base_color=base_color,
            emission=emission,
            texture=texture,
            **{key: values[key] for key in values},
        )

    def resolve_mesh(
        self,
        mesh: object,
        overrides: dict[str, Any] | None = None,
    ) -> Material:
        """Resolve a trimesh geometry's material deterministically.

        ``baseColorFactor`` from glTF is linear and is used as-is; named
        fall-backs such as ``diffuse`` are treated as sRGB and decoded.  The
        returned base colour is always linear.
        """
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None)
        factor = _read(material, ("baseColorFactor", "base_color_factor"))
        named = None if factor is not None else _read(material, ("baseColor", "diffuse", "color"))
        roughness = _read(material, ("roughnessFactor", "roughness"))
        metallic = _read(material, ("metallicFactor", "metallic"))
        transmission = _read(material, ("transmissionFactor", "transmission"))
        ior = _read(material, ("indexOfRefraction", "ior"))
        texture = _read(material, ("baseColorTexture", "image", "base_color_texture"))

        values = {**self.mesh_defaults, **(overrides or {})}
        # An explicit caller override always wins over the mesh's own glTF
        # factor; a value that merely came from ``mesh_defaults`` does not.
        explicit = set(overrides or {})
        explicit_base = "base_color" in explicit
        base_color = _as_color(values.pop("base_color", None), (1.0, 1.0, 1.0))
        base_is_srgb = bool(values.pop("base_color_srgb", False))
        if explicit_base:
            pass
        elif factor is not None:
            base_color = _as_color(factor, (1.0, 1.0, 1.0))
            base_is_srgb = False
        elif named is not None:
            base_color = _as_color(named, (1.0, 1.0, 1.0))
            base_is_srgb = True
        if roughness is not None and "roughness" not in explicit:
            values["roughness"] = float(np.clip(float(roughness), 0.0, 1.0))
        if metallic is not None and "metallic" not in explicit:
            values["metallic"] = float(np.clip(float(metallic), 0.0, 1.0))
        if transmission is not None and "transmission" not in explicit:
            # KHR_materials_transmission is a *thin-walled* model unless a
            # volume extension says otherwise, which is exactly the default
            # ``thin_walled=True`` this resolver already carries.
            values["transmission"] = float(np.clip(float(transmission), 0.0, 1.0))
        if ior is not None and "ior" not in explicit:
            value = float(ior)
            if value > 0.0:
                values["ior"] = value
        emission = _as_color(values.pop("emission", None), (0.0, 0.0, 0.0))
        resolved_texture = texture if texture is not None else values.pop("texture", None)
        allowed = {"roughness", "metallic", "specular", "transmission", "ior", "thin_walled"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown mesh material overrides: {sorted(unknown)}")
        linear = srgb_to_linear(base_color) if base_is_srgb else base_color
        return Material(
            base_color=linear,
            emission=emission,
            base_color_srgb=base_is_srgb,
            texture=resolved_texture,
            **{key: values[key] for key in values},
        )
