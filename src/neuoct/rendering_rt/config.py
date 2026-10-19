"""Isolated YAML validation for the offline ray tracer.

This module deliberately does **not** touch the existing NeuOct solver schema
(:mod:`neuoct.config.schema`).  It defines its own small, documented schema
under ``configs/raytracing`` so a ray-tracing config can never change training
behaviour.  ``validate_rt_config`` returns a fully-populated, normalised dict
with defaults applied and rejects unknown keys, wrong types and out-of-range
values with a precise message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RT_CONFIG_VERSION = 2
# Phase-one configs declare version 1; they stay valid because every version-2
# field is optional and defaults to the phase-one behaviour.
_SUPPORTED_VERSIONS = (1, 2)

_RENDER_MODES = ("direct", "one_bounce", "path")
_TONEMAPPERS = ("none", "linear", "clamp", "reinhard", "aces")
_CAMERA_TYPES = ("perspective", "equidistant")
_CAMERA_CONVENTIONS = ("neuoct", "legacy_blender")
_CAMERA_LAYOUTS = ("look_at", "orbit", "matrix")
_ENVIRONMENT_TYPES = ("constant", "hdri")


def _fail(path: str, message: str) -> None:
    raise ValueError(f"Invalid ray-tracing config at {path}: {message}")


def _check_keys(value: dict, allowed: set[str], path: str) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        _fail(path, f"unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected a number, got {type(value).__name__}")
    return float(value)


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"expected an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    return int(value)


def _vector3(value: Any, path: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        _fail(path, "expected a list of three numbers")
    return [_number(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _rotation(value: Any, path: str) -> list[list[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(row, (list, tuple)) and len(row) == 3 for row in value
    ):
        return [_vector3(list(row), f"{path}[{index}]") for index, row in enumerate(value)]
    if isinstance(value, (list, tuple)) and len(value) == 9:
        flat = [_number(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return [flat[0:3], flat[3:6], flat[6:9]]
    _fail(path, "expected a 3x3 matrix or a flat list of nine numbers")


def _section(config: dict, name: str, path: str = "root") -> dict:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        _fail(f"{path}.{name}", "expected a mapping")
    return value


def _validate_render(section: dict) -> dict:
    _check_keys(
        section,
        {
            "width",
            "height",
            "samples",
            "mode",
            "max_depth",
            "seed",
            "tonemap",
            "exposure",
            "background",
            "self_hit_eps",
            "pixel_mask",
            "batch_pixels",
            "batch_samples",
        },
        "render",
    )
    width = _integer(section.get("width", 512), "render.width", minimum=1)
    height = _integer(section.get("height", 512), "render.height", minimum=1)
    samples = _integer(section.get("samples", 1), "render.samples", minimum=1)
    mode = section.get("mode", "path")
    if mode not in _RENDER_MODES:
        _fail("render.mode", f"must be one of {list(_RENDER_MODES)}")
    max_depth = _integer(section.get("max_depth", 4), "render.max_depth", minimum=0)
    seed = _integer(section.get("seed", 0), "render.seed")
    tonemap = section.get("tonemap", "aces")
    if tonemap not in _TONEMAPPERS:
        _fail("render.tonemap", f"must be one of {list(_TONEMAPPERS)}")
    exposure = _number(section.get("exposure", 1.0), "render.exposure")
    background = _vector3(section.get("background", [0.0, 0.0, 0.0]), "render.background")
    self_hit_eps = _number(section.get("self_hit_eps", 1.0e-4), "render.self_hit_eps")
    if self_hit_eps < 0.0:
        _fail("render.self_hit_eps", "must be non-negative")
    pixel_mask = bool(section.get("pixel_mask", True))
    batch_pixels = _integer(section.get("batch_pixels", 0), "render.batch_pixels", minimum=0)
    batch_samples = _integer(section.get("batch_samples", 0), "render.batch_samples", minimum=0)
    return {
        "width": width,
        "height": height,
        "samples": samples,
        "mode": mode,
        "max_depth": max_depth,
        "seed": seed,
        "tonemap": tonemap,
        "exposure": exposure,
        "background": background,
        "self_hit_eps": self_hit_eps,
        "pixel_mask": pixel_mask,
        # 0 means "one batch"; the renderer bounds peak tensors by these.
        "batch_pixels": batch_pixels,
        "batch_samples": batch_samples,
    }


def _validate_camera(section: dict) -> dict:
    _check_keys(
        section,
        {
            "type",
            "layout",
            "camera_angle_x",
            "yfov_deg",
            "fisheye_fov_deg",
            "matrix",
            "position",
            "target",
            "up",
            "razimuth_deg",
            "elevation_deg",
            "azimuth_deg",
            "radius",
            "convention",
        },
        "camera",
    )
    projection = section.get("type", "perspective")
    if projection not in _CAMERA_TYPES:
        _fail("camera.type", f"must be one of {list(_CAMERA_TYPES)}")
    layout = section.get("layout", "look_at")
    if layout not in _CAMERA_LAYOUTS:
        _fail("camera.layout", f"must be one of {list(_CAMERA_LAYOUTS)}")
    yfov_deg = _number(section.get("yfov_deg", 45.0), "camera.yfov_deg")
    if not 0.0 < yfov_deg < 180.0:
        _fail("camera.yfov_deg", "must be in (0, 180)")
    matrix = section.get("matrix")
    if layout == "matrix":
        if matrix is None:
            _fail("camera.matrix", "is required when camera.layout is 'matrix'")
        if not isinstance(matrix, (list, tuple)) or len(matrix) != 16:
            _fail("camera.matrix", "expected 16 numbers (GLM column-major world-to-camera)")
        matrix = [_number(item, f"camera.matrix[{index}]") for index, item in enumerate(matrix)]
    elif matrix is not None:
        _fail("camera.matrix", "is only valid when camera.layout is 'matrix'")
    convention = section.get("convention", "legacy_blender" if layout == "matrix" else "neuoct")
    if convention not in _CAMERA_CONVENTIONS:
        _fail("camera.convention", f"must be one of {list(_CAMERA_CONVENTIONS)}")
    camera_angle_x = _number(section.get("camera_angle_x", 0.6911112070083618), "camera.camera_angle_x")
    if not 0.0 < camera_angle_x < 3.141592653589793:
        _fail("camera.camera_angle_x", "must be in (0, pi)")
    fisheye_fov_deg = _number(section.get("fisheye_fov_deg", 180.0), "camera.fisheye_fov_deg")
    if not 0.0 < fisheye_fov_deg <= 360.0:
        _fail("camera.fisheye_fov_deg", "must be in (0, 360]")
    position = _vector3(section.get("position", [3.0, 1.5, 3.0]), "camera.position")
    target = _vector3(section.get("target", [0.0, 0.0, 0.0]), "camera.target")
    up = _vector3(section.get("up", [0.0, 1.0, 0.0]), "camera.up")
    radius = _number(section.get("radius", 3.0), "camera.radius")
    if radius <= 0.0:
        _fail("camera.radius", "must be positive")
    azimuth_deg = _number(
        section.get("azimuth_deg", section.get("razimuth_deg", 45.0)), "camera.azimuth_deg"
    )
    elevation_deg = _number(section.get("elevation_deg", 20.0), "camera.elevation_deg")
    return {
        "type": projection,
        "layout": layout,
        "camera_angle_x": camera_angle_x,
        "yfov_deg": yfov_deg,
        "fisheye_fov_deg": fisheye_fov_deg,
        "matrix": matrix,
        "position": position,
        "target": target,
        "up": up,
        "radius": radius,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "convention": convention,
    }


def _validate_environment(section: dict) -> dict:
    _check_keys(
        section,
        {"type", "color", "intensity", "path", "rotation_deg", "strength", "channel_cap"},
        "environment",
    )
    environment_type = section.get("type", "constant")
    if environment_type not in _ENVIRONMENT_TYPES:
        _fail("environment.type", f"must be one of {list(_ENVIRONMENT_TYPES)}")
    color = _vector3(section.get("color", [1.0, 1.0, 1.0]), "environment.color")
    intensity = _number(section.get("intensity", 1.0), "environment.intensity")
    if environment_type == "constant":
        return {
            "type": "constant",
            "color": [channel * intensity for channel in color],
            "intensity": intensity,
        }
    path = section.get("path")
    if not isinstance(path, str) or not path:
        _fail("environment.path", "is required when environment.type is 'hdri'")
    rotation_deg = _number(section.get("rotation_deg", 0.0), "environment.rotation_deg")
    strength = _number(section.get("strength", 1.0), "environment.strength")
    if strength < 0.0:
        _fail("environment.strength", "must be non-negative")
    channel_cap = _number(section.get("channel_cap", 100.0), "environment.channel_cap")
    if channel_cap <= 0.0:
        _fail("environment.channel_cap", "must be positive")
    return {
        "type": "hdri",
        "path": path,
        "rotation_deg": rotation_deg,
        "strength": strength,
        # N-BVH clamps sampled envmap channels to 100 to kill sun fireflies.
        "channel_cap": channel_cap,
    }


def _validate_similarity(section: Any, path: str) -> dict:
    if section is None:
        section = {}
    if not isinstance(section, dict):
        _fail(path, "expected a mapping")
    _check_keys(section, {"scale", "rotation", "translation"}, path)
    return {
        "scale": _number(section.get("scale", 1.0), f"{path}.scale"),
        "rotation": _rotation(
            section.get("rotation", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), f"{path}.rotation"
        ),
        "translation": _vector3(section.get("translation", [0.0, 0.0, 0.0]), f"{path}.translation"),
    }


_MATERIAL_KEYS = (
    "base_color",
    "emission",
    "roughness",
    "metallic",
    "specular",
    "transmission",
    "ior",
    "thin_walled",
)


def _validate_material_block(value: Any, path: str) -> dict:
    """Validate one material override block shared by scene and NeuOct sections."""
    if not isinstance(value, dict) or not value:
        _fail(path, "expected a non-empty mapping")
    unknown = set(value).difference(_MATERIAL_KEYS)
    if unknown:
        _fail(path, f"unknown keys {sorted(unknown)}; allowed: {sorted(_MATERIAL_KEYS)}")
    resolved = dict(value)
    if "base_color" in resolved:
        resolved["base_color"] = _vector3(resolved["base_color"], f"{path}.base_color")
    if "emission" in resolved:
        resolved["emission"] = _vector3(resolved["emission"], f"{path}.emission")
    for key in ("roughness", "metallic", "specular", "transmission", "ior"):
        if key in resolved:
            resolved[key] = _number(resolved[key], f"{path}.{key}")
    if "thin_walled" in resolved:
        resolved["thin_walled"] = bool(resolved["thin_walled"])
    return resolved


def _validate_neuoct_material(value: Any) -> dict:
    """Material block applied to the NeuOct primitive itself.

    An empty mapping is valid and means "keep the audited NeuOct defaults".
    """
    if value is None:
        return {}
    if isinstance(value, dict) and not value:
        return {}
    return _validate_material_block(value, "scene.neuoct_material")


def _validate_material_overrides(value: Any) -> list[dict]:
    """Deterministic, name/pattern-keyed material overrides for ordinary meshes.

    Each entry is ``{match: <str>, by: node|material|pattern, material: {...}}``.
    ``by: pattern`` treats ``match`` as a case-insensitive substring test; the
    other two compare against the glTF node name and the glTF material name.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        _fail("scene.material_overrides", "expected a list")
    overrides: list[dict] = []
    for index, item in enumerate(value):
        path = f"scene.material_overrides[{index}]"
        if not isinstance(item, dict):
            _fail(path, "expected a mapping")
        _check_keys(item, {"match", "by", "material"}, path)
        match = item.get("match")
        if not isinstance(match, str) or not match:
            _fail(f"{path}.match", "expected a non-empty string")
        by = item.get("by", "pattern")
        if by not in ("node", "material", "pattern"):
            _fail(f"{path}.by", "must be 'node', 'material' or 'pattern'")
        material = _validate_material_block(item.get("material"), f"{path}.material")
        overrides.append({"match": match, "by": by, "material": material})
    return overrides


def _validate_scene(section: dict) -> dict:
    _check_keys(
        section,
        {
            "checkpoint",
            "object",
            "asset",
            "material_overrides",
            "neuoct_material",
            "rest_on_plane",
            "offset_xz",
            "query_depth",
            "point_scale",
            "ray_scale",
            "similarity",
            "hit_threshold",
            "ray_chunk_size",
            "candidate_chunk_size",
            "self_hit_eps",
            "name",
        },
        "scene",
    )
    checkpoint = section.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, str):
        _fail("scene.checkpoint", "expected a path string")
    object_key = section.get("object")
    if object_key is not None and not isinstance(object_key, str):
        _fail("scene.object", "expected a string")
    asset = section.get("asset")
    if asset is not None and not isinstance(asset, str):
        _fail("scene.asset", "expected a path string")
    query_depth_value = section.get("query_depth")
    query_depth = (
        None
        if query_depth_value is None
        else _integer(query_depth_value, "scene.query_depth", minimum=1)
    )
    point_scale = _number(section.get("point_scale", 0.99), "scene.point_scale")
    ray_scale = _number(section.get("ray_scale", 0.5), "scene.ray_scale")
    if point_scale <= 0.0 or ray_scale <= 0.0:
        _fail("scene.point_scale/ray_scale", "must be positive")
    hit_threshold = _number(section.get("hit_threshold", 0.5), "scene.hit_threshold")
    if not 0.0 <= hit_threshold < 1.0:
        _fail("scene.hit_threshold", "must be in [0, 1)")
    ray_chunk_size = _integer(section.get("ray_chunk_size", 16384), "scene.ray_chunk_size", minimum=1)
    candidate_chunk_size = _integer(
        section.get("candidate_chunk_size", 65536), "scene.candidate_chunk_size", minimum=1
    )
    rest_on_plane_value = section.get("rest_on_plane")
    rest_on_plane = (
        None
        if rest_on_plane_value is None
        else _number(rest_on_plane_value, "scene.rest_on_plane")
    )
    offset_xz = section.get("offset_xz")
    if offset_xz is not None:
        if not isinstance(offset_xz, (list, tuple)) or len(offset_xz) != 2:
            _fail("scene.offset_xz", "expected a list of two numbers [x, z]")
        offset_xz = [
            _number(offset_xz[0], "scene.offset_xz[0]"),
            _number(offset_xz[1], "scene.offset_xz[1]"),
        ]
    return {
        "checkpoint": checkpoint,
        "object": object_key,
        "asset": asset,
        "material_overrides": _validate_material_overrides(section.get("material_overrides")),
        "neuoct_material": _validate_neuoct_material(section.get("neuoct_material")),
        "rest_on_plane": rest_on_plane,
        "offset_xz": offset_xz,
        "query_depth": query_depth,
        "point_scale": point_scale,
        "ray_scale": ray_scale,
        "similarity": _validate_similarity(section.get("similarity"), "scene.similarity"),
        "hit_threshold": hit_threshold,
        "ray_chunk_size": ray_chunk_size,
        "candidate_chunk_size": candidate_chunk_size,
        "self_hit_eps": _number(section.get("self_hit_eps", 1.0e-4), "scene.self_hit_eps"),
        "name": str(section.get("name", "neuoct")),
    }


def _validate_primitives(value: Any) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        _fail("primitives", "expected a list")
    primitives: list[dict] = []
    for index, item in enumerate(value):
        path = f"primitives[{index}]"
        if not isinstance(item, dict):
            _fail(path, "expected a mapping")
        kind = item.get("type")
        if kind == "mesh":
            _check_keys(
                item,
                {"type", "path", "vertices", "faces", "transform", "material", "uv_scale", "name"},
                path,
            )
            has_path = item.get("path") is not None
            has_inline = item.get("vertices") is not None or item.get("faces") is not None
            if not has_path and not (item.get("vertices") is not None and item.get("faces") is not None):
                _fail(path, "mesh primitive needs either 'path' or both 'vertices' and 'faces'")
            if has_path and has_inline:
                _fail(path, "mesh primitive cannot combine 'path' with inline 'vertices'/'faces'")
            primitives.append(
                {
                    "type": "mesh",
                    "path": None if item.get("path") is None else str(item["path"]),
                    "vertices": item.get("vertices"),
                    "faces": item.get("faces"),
                    "transform": item.get("transform"),
                    "material": item.get("material", {}) or {},
                    "uv_scale": item.get("uv_scale"),
                    "name": str(item.get("name", f"mesh_{index}")),
                }
            )
        elif kind == "neuoct":
            _check_keys(item, {"type", "material", "name", "similarity"}, path)
            primitives.append(
                {
                    "type": "neuoct",
                    "material": item.get("material", {}) or {},
                    "name": str(item.get("name", f"neuoct_{index}")),
                    "similarity": item.get("similarity"),
                }
            )
        else:
            _fail(path + ".type", "must be 'mesh' or 'neuoct'")
    return primitives


def _validate_output(section: dict) -> dict:
    _check_keys(section, {"path", "metadata", "format", "already_srgb"}, "output")
    path = section.get("path")
    if path is not None and not isinstance(path, str):
        _fail("output.path", "expected a path string")
    metadata = section.get("metadata")
    if metadata is not None and not isinstance(metadata, str):
        _fail("output.metadata", "expected a path string")
    output_format = section.get("format", "png")
    if output_format not in ("png", "npy", "exr"):
        _fail("output.format", "must be 'png', 'npy' or 'exr'")
    return {
        "path": path,
        "metadata": metadata,
        "format": output_format,
        "already_srgb": bool(section.get("already_srgb", False)),
    }


def validate_rt_config(config: Any) -> dict:
    """Validate and normalise a ray-tracing config mapping."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("Ray-tracing config must be a mapping")
    _check_keys(
        config,
        {"version", "placement", "render", "camera", "scene", "environment", "primitives", "output"},
        "root",
    )
    version = _integer(config.get("version", RT_CONFIG_VERSION), "version", minimum=1)
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported ray-tracing config version {version}; "
            f"expected one of {list(_SUPPORTED_VERSIONS)}"
        )
    if not any(key in config for key in ("scene", "primitives")):
        raise ValueError("Ray-tracing config must define a 'scene' or 'primitives'")
    return {
        "version": version,
        "render": _validate_render(_section(config, "render")),
        "camera": _validate_camera(_section(config, "camera")),
        "scene": _validate_scene(_section(config, "scene")),
        "environment": _validate_environment(_section(config, "environment")),
        "primitives": _validate_primitives(config.get("primitives")),
        "output": _validate_output(_section(config, "output")),
    }


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_rt_config(path: str | Path) -> dict:
    """Load and validate a ray-tracing YAML file.

    If the root contains a ``placement`` path, that YAML is loaded first
    (relative to the config file) and the main file is merged on top of it.
    This lets a shared placement file supply the similarity transform and
    camera while a scene config supplies the checkpoint and render settings.
    """
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("Ray-tracing config must be a mapping")
    placement = config.pop("placement", None)
    if placement is not None:
        placement_path = Path(str(placement))
        if not placement_path.is_absolute():
            placement_path = config_path.parent / placement_path
        placement_config = yaml.safe_load(placement_path.read_text(encoding="utf-8"))
        if not isinstance(placement_config, dict):
            raise ValueError(f"Placement file {placement_path} must be a mapping")
        config = _deep_merge(placement_config, config)
    return validate_rt_config(config)
