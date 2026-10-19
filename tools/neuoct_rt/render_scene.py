#!/usr/bin/env python3
"""Render a ray-tracing YAML config offline, directly to image + metadata.

The script never uses a screenshot: it builds a
:class:`~neuoct.rendering_rt.scene.Scene` from the config (a graph-preserving
glTF asset, inline/file meshes, and/or a packed NeuOct checkpoint), renders with
the batched integrator, and writes the image, the linear buffers and canonical
JSON metadata to disk.  ``--dry-run`` builds and validates everything without
tracing a pixel, which is useful on CPU-only checkouts.

Graph-preserving glTF loading
-----------------------------
:func:`load_gltf_scene` loads **only the glTF default scene** and walks the full
node graph itself, composing each node's TRS/matrix exactly the way N-BVH's
``read_node_transform`` does.  Every geometry instance is kept: the chess asset
has 15 shared meshes and 49 instances, and the second ``scene_neural``
hierarchy is never touched.  Instances are flattened into one world-space
accelerator with parallel per-face arrays for node, source-geometry and material
provenance, so UVs, vertex normals, base-colour textures and per-instance
materials all survive.

Legacy matrix camera
--------------------
:func:`camera_from_legacy_matrix` adapts the N-BVH flat ``camera.matrix`` field:
it is GLM column-major **world-to-camera**, so it is reshaped with
``order='F'``, inverted, and used with local ``-Z`` forward
(``legacy_blender``) and N-BVH's vertical FOV.  The raw translation column is
not the camera origin.

Examples::

    python tools/neuoct_rt/render_scene.py \
        --config configs/raytracing/smoke_triangle.yaml \
        --output outputs/raytracing/smoke_triangle.png
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


def _bootstrap() -> Path:
    repo = Path(__file__).resolve().parents[2]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import ocnn  # noqa: F401

        return src
    except Exception:
        pass
    for name in (
        "neuoct",
        "neuoct.geometry",
        "neuoct.models",
        "neuoct.engine",
        "neuoct.data",
        "neuoct.rendering",
        "neuoct.config",
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(src / name.replace(".", "/"))]
        sys.modules[name] = module
    sys.modules["neuoct"].__version__ = "0.1.0"
    return src


_BOOTSTRAP_SRC = _bootstrap()

from neuoct.rendering_rt import (  # noqa: E402
    Camera,
    CombinedMeshPrimitive,
    Integrator,
    MaterialResolver,
    MeshPrimitive,
    NeuOctPrimitive,
    Scene,
    SimilarityTransform,
    embree_status,
    load_rt_config,
    resolve_device,
    save_image,
    save_linear_npy,
    write_metadata,
)
from neuoct.rendering_rt.hdri import environment_from_config  # noqa: E402


class _InlineMesh:
    """Minimal mesh stand-in when trimesh is unavailable."""

    def __init__(self, vertices, faces) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        self.visual = None


# --------------------------------------------------------------------------- glTF
@dataclass
class GltfInstance:
    """One glTF node that references a mesh, in default-scene depth-first order."""

    name: str
    mesh_index: int
    mesh_name: str
    material_index: int | None
    material_name: str | None
    world_matrix: np.ndarray
    children: list[int] = field(default_factory=list)


@dataclass
class GltfSceneData:
    """A graph-preserving, world-space flattening of a glTF default scene."""

    source: str
    scene_index: int
    scene_name: str
    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray | None
    vertex_uv: np.ndarray | None
    face_material: np.ndarray
    face_instance: np.ndarray
    face_geometry: np.ndarray
    materials: list
    material_names: list[str]
    instance_names: list[str]
    geometry_names: list[str]
    instances: list[GltfInstance]
    bounds: np.ndarray
    skipped_multi_primitive: list[str] = field(default_factory=list)

    @property
    def num_triangles(self) -> int:
        return int(self.faces.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])


def _node_matrix(node: dict) -> np.ndarray:
    """Compose one glTF node's TRS/matrix exactly like N-BVH's reader."""
    if node.get("matrix"):
        return np.asarray(node["matrix"], dtype=np.float64).reshape((4, 4), order="F")
    matrix = np.eye(4, dtype=np.float64)
    scale = node.get("scale")
    if scale:
        matrix = np.diag([float(scale[0]), float(scale[1]), float(scale[2]), 1.0]) @ matrix
    rotation = node.get("rotation")
    if rotation:
        x, y, z, w = (float(value) for value in rotation)
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        if norm > 0.0:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
        matrix = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ) @ matrix
    translation = node.get("translation")
    if translation:
        translate = np.eye(4, dtype=np.float64)
        translate[:3, 3] = [float(value) for value in translation]
        matrix = translate @ matrix
    return matrix


def _gltf_instances(gltf: dict) -> tuple[int, str, list[GltfInstance]]:
    """Walk the **default scene** graph and return its mesh-bearing instances."""
    scenes = gltf.get("scenes") or []
    if not scenes:
        raise ValueError("glTF has no scenes; cannot pick a default scene")
    index = int(gltf.get("scene", 0))
    if not 0 <= index < len(scenes):
        raise ValueError(f"glTF default scene index {index} is out of range")
    scene = scenes[index]
    nodes = gltf.get("nodes") or []
    meshes = gltf.get("meshes") or []
    materials = gltf.get("materials") or []
    instances: list[GltfInstance] = []

    def visit(node_index: int, parent: np.ndarray) -> None:
        node = nodes[node_index]
        world = parent @ _node_matrix(node)
        if node.get("mesh") is not None:
            mesh_index = int(node["mesh"])
            mesh = meshes[mesh_index] if 0 <= mesh_index < len(meshes) else {}
            primitives = mesh.get("primitives") or []
            material_index = primitives[0].get("material") if primitives else None
            material_name = None
            if material_index is not None and 0 <= int(material_index) < len(materials):
                material_name = materials[int(material_index)].get("name")
                material_index = int(material_index)
            instances.append(
                GltfInstance(
                    name=str(node.get("name", f"node_{node_index}")),
                    mesh_index=mesh_index,
                    mesh_name=str(mesh.get("name", f"mesh_{mesh_index}")),
                    material_index=material_index,
                    material_name=material_name,
                    world_matrix=world,
                    children=[int(child) for child in node.get("children", [])],
                )
            )
        for child in node.get("children", []):
            visit(int(child), world)

    for root in scene.get("nodes", []):
        visit(int(root), np.eye(4, dtype=np.float64))
    return index, str(scene.get("name", f"scene_{index}")), instances


def _override_matches(entry: dict, node_name: str, material_name: str | None) -> bool:
    match = str(entry["match"])
    kind = entry.get("by", "pattern")
    if kind == "node":
        return node_name == match
    if kind == "material":
        return material_name is not None and material_name == match
    lowered = match.lower()
    return lowered in node_name.lower() or (
        material_name is not None and lowered in material_name.lower()
    )


def _read_accessor(gltf: dict, base_dir: Path, accessor_index: int) -> np.ndarray | None:
    """Read a float accessor straight out of the glTF buffers.

    Vertex attributes are read from the file rather than from trimesh so the
    loader cannot inherit a library's convention change (trimesh, for example,
    flips the V coordinate and quantises ``baseColorFactor`` to 8 bits).  glTF
    UVs are top-left origin, which is exactly how the texture is sampled and how
    N-BVH's own ``tex2D`` lookup behaves, so no flip is applied here.
    """
    accessors = gltf.get("accessors") or []
    views = gltf.get("bufferViews") or []
    buffers = gltf.get("buffers") or []
    if not 0 <= accessor_index < len(accessors):
        return None
    accessor = accessors[accessor_index]
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}.get(accessor.get("type"))
    if components is None or int(accessor.get("componentType", 0)) != 5126:
        return None
    view = views[int(accessor["bufferView"])]
    buffer = buffers[int(view.get("buffer", 0))]
    uri = buffer.get("uri")
    if uri is None:
        return None
    if uri.startswith("data:"):
        payload = base64.b64decode(uri.split(",", 1)[1])
    else:
        payload = (base_dir / uri).read_bytes()
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"])
    stride = int(view.get("byteStride") or (components * 4))
    raw = np.frombuffer(payload, dtype=np.uint8, count=stride * (count - 1) + components * 4, offset=start)
    rows = np.lib.stride_tricks.as_strided(
        raw[: (count - 1) * stride + components * 4],
        shape=(count, components * 4),
        strides=(stride, 1),
    )
    contiguous = np.ascontiguousarray(rows)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float32).reshape(count, components).astype(
        np.float64
    )


def _texture_image(geometry: object) -> object | None:
    """The glTF base-colour image of a trimesh geometry, or ``None``.

    ``PBRMaterial.image`` is not always an image (trimesh may hand back another
    material object), so the base-colour texture attribute is preferred and the
    generic ``image`` attribute is only accepted when it is *not* a material.
    """
    material = getattr(getattr(geometry, "visual", None), "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is not None:
        return image
    candidate = getattr(material, "image", None)
    if candidate is not None and not hasattr(candidate, "pbrMetallicRoughness"):
        return candidate
    return None


def _material_from_gltf(
    gltf_material: dict | None,
    texture: object | None,
    resolver: MaterialResolver,
    overrides: dict,
):
    """Build a :class:`Material` from raw glTF PBR factors, then apply overrides.

    ``baseColorFactor`` is linear in glTF and is used verbatim (never routed
    through an 8-bit colour); ``metallicFactor``/``roughnessFactor`` default to
    glTF's own 1.0; ``emissiveFactor`` becomes emission.  ``KHR_materials_ior``
    and ``KHR_materials_transmission`` are honoured, and ``KHR_materials_volume``
    marks the material as a *closed* medium (``thin_walled=False``) because that
    extension is precisely the statement that the surface bounds a volume.
    """
    from neuoct.rendering_rt import Material

    defaults = dict(resolver.mesh_defaults)
    base_color = (1.0, 1.0, 1.0)
    emission = (0.0, 0.0, 0.0)
    if gltf_material:
        pbr = gltf_material.get("pbrMetallicRoughness") or {}
        factor = pbr.get("baseColorFactor")
        if factor:
            base_color = tuple(float(value) for value in factor[:3])
        if "metallicFactor" in pbr:
            defaults["metallic"] = float(pbr["metallicFactor"])
        if "roughnessFactor" in pbr:
            defaults["roughness"] = float(pbr["roughnessFactor"])
        emissive = gltf_material.get("emissiveFactor")
        if emissive:
            emission = tuple(float(value) for value in emissive[:3])
        extensions = gltf_material.get("extensions") or {}
        transmission = extensions.get("KHR_materials_transmission")
        if transmission and "transmissionFactor" in transmission:
            defaults["transmission"] = float(transmission["transmissionFactor"])
        ior = extensions.get("KHR_materials_ior")
        if ior and "ior" in ior:
            defaults["ior"] = float(ior["ior"])
        if "KHR_materials_volume" in extensions:
            defaults["thin_walled"] = False
    defaults.update(overrides)
    return Material(
        base_color=torch_as_tensor3(defaults.pop("base_color", base_color)),
        emission=torch_as_tensor3(defaults.pop("emission", emission)),
        texture=texture,
        base_color_srgb=False,
        **{
            key: value
            for key, value in defaults.items()
            if key in ("roughness", "metallic", "specular", "transmission", "ior", "thin_walled")
        },
    )


def torch_as_tensor3(value):
    import torch

    return torch.as_tensor(value, dtype=torch.float32).reshape(3)


def load_gltf_scene(
    path: str | Path,
    *,
    material_resolver: MaterialResolver | None = None,
    material_overrides: list[dict] | None = None,
) -> GltfSceneData:
    """Load a glTF **default scene** preserving every instance and material.

    Duplicate instances of a shared mesh are flattened into one world-space
    triangle soup (cheapest correct accelerator) but never deduplicated away:
    ``face_instance`` / ``face_geometry`` / ``face_material`` keep the full
    provenance.  A second hierarchy such as the chess asset's ``scene_neural``
    is never visited because only ``gltf['scene']`` is walked.
    """
    asset_path = Path(path)
    if not asset_path.exists():
        raise FileNotFoundError(f"glTF asset not found: {asset_path}")
    gltf = json.loads(asset_path.read_text(encoding="utf-8"))
    scene_index, scene_name, instances = _gltf_instances(gltf)
    if not instances:
        raise ValueError(f"{asset_path} default scene {scene_name!r} has no mesh nodes")
    try:
        import trimesh
    except Exception as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"Loading glTF {asset_path} requires trimesh; install it or use an inline mesh"
        ) from error
    scene = trimesh.load(str(asset_path), process=False)
    if not hasattr(scene, "geometry"):  # a bare mesh file, not a scene
        scene = trimesh.Scene({"mesh": scene})
    graph = getattr(scene, "graph", None)
    resolver = material_resolver or MaterialResolver()
    overrides = list(material_overrides or [])

    vertices: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    face_material: list[np.ndarray] = []
    face_instance: list[np.ndarray] = []
    face_geometry: list[np.ndarray] = []
    materials: list = []
    material_names: list[str] = []
    instance_names: list[str] = []
    geometry_names: list[str] = []
    skipped: list[str] = []
    for instance_index, instance in enumerate(instances):
        geometry_name = None
        if graph is not None:
            try:
                _, geometry_name = graph[instance.name]
            except Exception:
                geometry_name = None
        if geometry_name is None or geometry_name not in scene.geometry:
            geometry_name = instance.mesh_name
        if geometry_name not in scene.geometry:
            raise KeyError(
                f"glTF node {instance.name!r} references mesh {instance.mesh_name!r} "
                f"which trimesh did not expose"
            )
        geometry = scene.geometry[geometry_name]
        matrix = np.asarray(instance.world_matrix, dtype=np.float64)
        linear = matrix[:3, :3]
        local = np.asarray(geometry.vertices, dtype=np.float64)
        world_vertices = local @ linear.T + matrix[:3, 3]
        vertices.append(world_vertices)
        local_faces = np.asarray(geometry.faces, dtype=np.int64)
        faces.append(local_faces)
        # Vertex normals must be transformed by the inverse transpose.
        try:
            normal_matrix = np.linalg.inv(linear).T
        except np.linalg.LinAlgError:
            normal_matrix = None
        local_normals = getattr(geometry, "vertex_normals", None)
        if local_normals is not None and normal_matrix is not None:
            array = np.asarray(local_normals, dtype=np.float64)
            if array.shape == local.shape:
                array = array @ normal_matrix.T
                lengths = np.linalg.norm(array, axis=1, keepdims=True)
                normals.append(array / np.where(lengths < 1.0e-15, 1.0, lengths))
            else:
                normals.append(np.zeros_like(world_vertices))
        else:
            normals.append(np.zeros_like(world_vertices))
        # UVs come from the file's own TEXCOORD_0 accessor (top-left origin, no
        # flip), falling back to trimesh only when the raw accessor is unusable.
        primitives = (gltf.get("meshes") or [{}])[instance.mesh_index].get("primitives", [])
        attributes = primitives[0].get("attributes", {}) if primitives else {}
        raw_uv = None
        if "TEXCOORD_0" in attributes:
            raw_uv = _read_accessor(gltf, asset_path.parent, int(attributes["TEXCOORD_0"]))
        if raw_uv is None or raw_uv.shape != (local.shape[0], 2):
            raw_uv = getattr(getattr(geometry, "visual", None), "uv", None)
            raw_uv = (
                np.asarray(raw_uv, dtype=np.float64)
                if raw_uv is not None and np.asarray(raw_uv).shape == (local.shape[0], 2)
                else np.zeros((local.shape[0], 2), dtype=np.float64)
            )
        uvs.append(raw_uv)
        face_count = int(local_faces.shape[0])
        face_material.append(np.full(face_count, instance_index, dtype=np.int64))
        face_instance.append(np.full(face_count, instance_index, dtype=np.int64))
        face_geometry.append(np.full(face_count, instance.mesh_index, dtype=np.int64))
        # Per-instance material: raw glTF PBR factors plus any override that
        # matches this node/material.  Identity is preserved, not merged.
        resolved_overrides: dict = {}
        for entry in overrides:
            if _override_matches(entry, instance.name, instance.material_name):
                resolved_overrides.update(entry["material"])
        gltf_materials = gltf.get("materials") or []
        gltf_material = (
            gltf_materials[instance.material_index]
            if instance.material_index is not None
            and 0 <= instance.material_index < len(gltf_materials)
            else None
        )
        material = _material_from_gltf(
            gltf_material,
            _texture_image(geometry),
            resolver,
            resolved_overrides,
        )
        materials.append(material)
        material_names.append(
            f"{instance.material_name or geometry_name}@{instance.name}"
        )
        instance_names.append(instance.name)
        geometry_names.append(instance.mesh_name)
        if len(primitives) > 1:
            skipped.append(instance.name)

    concatenated_vertices = np.concatenate(vertices, axis=0)
    # Re-offset faces now that every instance's vertex count is known.
    offsets = np.cumsum([0] + [array.shape[0] for array in vertices[:-1]])
    faces = [array + offset for array, offset in zip(faces, offsets)]
    data = GltfSceneData(
        source=str(asset_path),
        scene_index=scene_index,
        scene_name=scene_name,
        vertices=concatenated_vertices,
        faces=np.concatenate(faces, axis=0),
        vertex_normals=np.concatenate(normals, axis=0),
        vertex_uv=np.concatenate(uvs, axis=0),
        face_material=np.concatenate(face_material),
        face_instance=np.concatenate(face_instance),
        face_geometry=np.concatenate(face_geometry),
        materials=materials,
        material_names=material_names,
        instance_names=instance_names,
        geometry_names=geometry_names,
        instances=instances,
        bounds=np.stack([concatenated_vertices.min(axis=0), concatenated_vertices.max(axis=0)]),
        skipped_multi_primitive=skipped,
    )
    return data


def camera_from_legacy_matrix(
    matrix,
    *,
    width: int,
    height: int,
    yfov_deg: float,
    projection: str = "perspective",
    fisheye_fov_deg: float = 180.0,
) -> Camera:
    """Exact adapter for N-BVH's flat ``camera.matrix`` field.

    The 16 numbers are GLM column-major and encode the **world-to-camera** view
    matrix, so the adapter reshapes with ``order='F'``, inverts it and uses the
    ``legacy_blender`` convention (local ``-Z`` forward).  The vertical FOV is
    N-BVH's ``yfov``.
    """
    return Camera.from_legacy_matrix(
        matrix,
        width=width,
        height=height,
        yfov=float(np.deg2rad(yfov_deg)),
        projection=projection,
        fisheye_fov=float(np.deg2rad(fisheye_fov_deg)),
    )


# --------------------------------------------------------------------------- scene
def _load_mesh(entry: dict) -> object:
    if entry.get("vertices") is not None and entry.get("faces") is not None:
        vertices = entry["vertices"]
        faces = entry["faces"]
        try:
            import trimesh

            return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        except Exception:
            return _InlineMesh(vertices, faces)
    try:
        import trimesh
    except Exception as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"Loading mesh {entry['path']!r} requires trimesh; install it or use an inline mesh"
        ) from error
    mesh = trimesh.load_mesh(entry["path"], process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _similarity_from_config(section: dict) -> SimilarityTransform:
    return SimilarityTransform(
        scale=float(section["scale"]),
        rotation=np.asarray(section["rotation"], dtype=np.float32),
        translation=np.asarray(section["translation"], dtype=np.float32),
    )


def occupancy_aabb(octree, query_depth: int, *, point_scale: float = 0.99, ray_scale: float = 0.5):
    """Canonical occupied AABB of a packed checkpoint, in the object frame.

    The occupied voxel set *is* the checkpoint's canonical extent, so it is read
    from the topology rather than assumed to be a unit cube.  A voxel with
    integer coordinate ``i`` spans ``[i, i + 1]``, and the audited mapping is
    ``p_octree = p_object * point_scale * ray_scale * 2**depth + 2**(depth-1)``.
    """
    import torch

    coordinates = octree.voxel_coordinates
    scale = 1 << int(query_depth)
    linear = float(point_scale) * float(ray_scale) * scale
    offset = 0.5 * scale
    lower = coordinates.min(dim=0).values.to(torch.float64).cpu().numpy()
    upper = coordinates.max(dim=0).values.to(torch.float64).cpu().numpy() + 1.0
    return (lower - offset) / linear, (upper - offset) / linear


def _placement_from_checkpoint(
    checkpoint: str,
    scene_config: dict,
    *,
    device: str,
):
    """Build the NeuOct similarity transform, resting the object on the board.

    ``p_scene = scale * (p_object @ R.T) + translation`` with the audited proper
    rotation ``R = [[1,0,0],[0,0,1],[0,-1,0]]`` (det +1, Z-up -> Y-up).  The
    vertical translation solves ``board_top - scale * bbox_min.z`` using the
    checkpoint's *measured* occupied AABB rather than an assumed unit extent.
    """
    from neuoct.engine.local_checkpoint import load_local_inference_checkpoint

    _, octree, payload = load_local_inference_checkpoint(
        checkpoint, device="cpu", expected_object=scene_config.get("object")
    )
    depth = int(payload["query_depth"])
    lower, upper = occupancy_aabb(
        octree,
        depth,
        point_scale=float(scene_config["point_scale"]),
        ray_scale=float(scene_config["ray_scale"]),
    )
    similarity_config = dict(scene_config["similarity"])
    rotation = np.asarray(similarity_config["rotation"], dtype=np.float64)
    scale = float(similarity_config["scale"])
    translation = np.asarray(similarity_config["translation"], dtype=np.float64).copy()
    plane = scene_config.get("rest_on_plane")
    # ``R`` maps object +Z onto scene +Y, so resting the object on the board is
    # ``board_top - scale * min_over_object(rotated up axis)``.  The rotated
    # extents must come from all eight corners: a rotation about a negative axis
    # swaps which corner is the minimum, so transforming ``lower``/``upper``
    # alone would report (and rest on) the wrong face.
    corners = np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    rotated_corners = corners @ rotation.T
    rotated_lower = rotated_corners.min(axis=0)
    rotated_upper = rotated_corners.max(axis=0)
    if plane is not None:
        translation[1] = float(plane) - scale * float(rotated_lower[1])
    offset = scene_config.get("offset_xz")
    if offset is not None:
        translation[0] = float(offset[0])
        translation[2] = float(offset[1])
    similarity = SimilarityTransform(
        scale=scale,
        rotation=rotation.astype(np.float32),
        translation=translation.astype(np.float32),
    )
    placed_lower = rotated_lower * scale + translation
    placed_upper = rotated_upper * scale + translation
    report = {
        "checkpoint": str(checkpoint),
        "query_depth": depth,
        "point_scale": float(scene_config["point_scale"]),
        "ray_scale": float(scene_config["ray_scale"]),
        "object_bbox_min": [float(value) for value in lower],
        "object_bbox_max": [float(value) for value in upper],
        "rotation": rotation.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "scale": scale,
        "translation": [float(value) for value in translation],
        "transformed_bbox_min": [float(value) for value in placed_lower],
        "transformed_bbox_max": [float(value) for value in placed_upper],
        "rotated_bbox_min": [float(value) for value in rotated_lower],
        "rotated_bbox_max": [float(value) for value in rotated_upper],
        "bbox_corners_transformed": 8,
        "rest_on_plane": None if plane is None else float(plane),
        "offset_xz": None if offset is None else [float(offset[0]), float(offset[1])],
        "translation_y_rule": (
            "translation_y = rest_on_plane - scale * min_over_8_rotated_corners(up axis)"
        ),
        "aabb_source": "occupied voxel set of the packed checkpoint topology",
    }
    return similarity, report


def build_scene(config: dict, *, device: str = "cpu", dry_run: bool = False) -> Scene:
    scene_config = config["scene"]
    default_similarity = _similarity_from_config(scene_config["similarity"])
    resolver = MaterialResolver()
    primitives = []
    entries = config["primitives"]
    if not entries and (scene_config["checkpoint"] or scene_config.get("asset")):
        if scene_config.get("asset"):
            entries.append({"type": "gltf", "path": scene_config["asset"], "name": "asset"})
        if scene_config["checkpoint"]:
            entries.append(
                {
                    "type": "neuoct",
                    "material": dict(scene_config.get("neuoct_material") or {}),
                    "name": "neuoct",
                    "similarity": None,
                }
            )
    if not entries:
        raise ValueError("Config defines no primitives, no scene.asset and no scene.checkpoint")
    reports: dict[str, object] = {}
    entries = list(entries)
    for index, entry in enumerate(entries):
        kind = entry.get("type")
        if kind == "mesh":
            geometry = _load_mesh(entry)
            primitive = MeshPrimitive(
                geometry,
                transform=entry.get("transform"),
                material_overrides=entry.get("material") or None,
                material_resolver=resolver,
                primitive_id=index,
                name=entry.get("name"),
                uv_scale=entry.get("uv_scale"),
            )
        elif kind == "gltf":
            data = load_gltf_scene(
                entry["path"],
                material_resolver=resolver,
                material_overrides=scene_config.get("material_overrides"),
            )
            primitive = CombinedMeshPrimitive(
                data.vertices,
                data.faces,
                face_material=data.face_material,
                materials=data.materials,
                face_instance=data.face_instance,
                face_geometry=data.face_geometry,
                instance_names=data.instance_names,
                geometry_names=data.geometry_names,
                vertex_normals=data.vertex_normals,
                vertex_uv=data.vertex_uv,
                primitive_id=index,
                name=entry.get("name") or f"gltf_{index}",
            )
            reports["asset"] = {
                "source": data.source,
                "default_scene_index": data.scene_index,
                "default_scene_name": data.scene_name,
                "instances": len(data.instances),
                "unique_geometries": len(set(data.geometry_names)),
                "triangles": data.num_triangles,
                "vertices": data.num_vertices,
                "materials": data.material_names,
                "instance_names": data.instance_names,
                "bounds": data.bounds.tolist(),
                "multi_primitive_meshes": sorted(set(data.skipped_multi_primitive)),
                "accelerator": primitive.accelerator,
            }
        else:
            checkpoint = scene_config["checkpoint"]
            if checkpoint is None:
                raise ValueError("NeuOct primitive requires scene.checkpoint")
            similarity = (
                _similarity_from_config(entry["similarity"])
                if entry.get("similarity")
                else default_similarity
            )
            if dry_run or not Path(checkpoint).exists():
                if not dry_run:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
                primitive = _PlaceholderNeuOct(entry, similarity, index)
            else:
                similarity, placement_report = _placement_from_checkpoint(
                    checkpoint, scene_config, device=device
                )
                reports["placement"] = placement_report
                checkpoint_kwargs = dict(
                    hit_threshold=scene_config["hit_threshold"],
                    ray_chunk_size=scene_config["ray_chunk_size"],
                    candidate_chunk_size=scene_config["candidate_chunk_size"],
                    self_hit_eps=scene_config["self_hit_eps"],
                    material_overrides=entry.get("material") or None,
                    material_resolver=resolver,
                    primitive_id=index,
                    name=entry.get("name"),
                    point_scale=scene_config["point_scale"],
                    ray_scale=scene_config["ray_scale"],
                )
                if scene_config["query_depth"] is not None:
                    checkpoint_kwargs["depth"] = scene_config["query_depth"]
                primitive = NeuOctPrimitive.from_checkpoint(
                    checkpoint,
                    object_key=scene_config["object"],
                    device=device,
                    object_to_scene=similarity,
                    **checkpoint_kwargs,
                )
        primitives.append(primitive)
    scene = Scene(primitives, name=scene_config.get("name", "scene"))
    scene.build_report = reports  # type: ignore[attr-defined]
    return scene


class _PlaceholderNeuOct:
    """Dry-run stand-in that records the requested setup without a checkpoint."""

    primitive_id: int
    name: str

    def __init__(self, entry: dict, similarity: SimilarityTransform, primitive_id: int) -> None:
        self.entry = entry
        self.similarity = similarity
        self.primitive_id = int(primitive_id)
        self.name = entry.get("name", f"neuoct_{primitive_id}")

    def intersect(self, rays):  # pragma: no cover - only used in dry runs
        from neuoct.rendering_rt import RayHit

        return RayHit.miss(rays.num_rays, device=rays.device)


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, or ``"missing"`` when it does not exist."""
    import hashlib

    target = Path(path)
    if not target.exists() or not target.is_file():
        return "missing"
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gltf_input_hashes(gltf_path: str | Path) -> dict[str, str]:
    """SHA-256 of the glTF and of every buffer/texture it references."""
    asset_path = Path(gltf_path)
    hashes = {str(asset_path): sha256_file(asset_path)}
    try:
        document = json.loads(asset_path.read_text(encoding="utf-8"))
    except Exception:
        return hashes
    for entry in document.get("buffers", []) or []:
        uri = entry.get("uri")
        if isinstance(uri, str) and not uri.startswith("data:"):
            hashes[str(asset_path.parent / uri)] = sha256_file(asset_path.parent / uri)
    for entry in document.get("images", []) or []:
        uri = entry.get("uri")
        if isinstance(uri, str) and not uri.startswith("data:"):
            hashes[str(asset_path.parent / uri)] = sha256_file(asset_path.parent / uri)
    return hashes


def build_camera(config: dict, *, device: str = "cpu") -> Camera:
    camera_config = config["camera"]
    render = config["render"]
    if camera_config["layout"] == "matrix":
        camera = camera_from_legacy_matrix(
            camera_config["matrix"],
            width=render["width"],
            height=render["height"],
            yfov_deg=camera_config["yfov_deg"],
            projection=camera_config["type"],
            fisheye_fov_deg=camera_config["fisheye_fov_deg"],
        )
        return camera.to(device)
    if camera_config["layout"] == "orbit":
        target = np.asarray(camera_config["target"], dtype=np.float32)
        azimuth = np.deg2rad(camera_config["azimuth_deg"])
        elevation = np.deg2rad(camera_config["elevation_deg"])
        radius = camera_config["radius"]
        eye = target + radius * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.sin(elevation),
                np.cos(elevation) * np.sin(azimuth),
            ],
            dtype=np.float32,
        )
    else:
        eye = np.asarray(camera_config["position"], dtype=np.float32)
        target = np.asarray(camera_config["target"], dtype=np.float32)
    camera = Camera.look_at(
        eye,
        target,
        camera_config["up"],
        width=render["width"],
        height=render["height"],
        camera_angle_x=camera_config["camera_angle_x"],
        projection=camera_config["type"],
        fisheye_fov=np.deg2rad(camera_config["fisheye_fov_deg"]),
        convention=camera_config["convention"],
    )
    # Primary rays are generated from the camera's own device, so the camera must
    # live on the requested render device or a CUDA scene gets CPU rays.
    return camera.to(device)


def build_environment(config: dict, *, device: str = "cpu"):
    return environment_from_config(config["environment"], device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="image path override")
    parser.add_argument("--metadata", type=Path, default=None, help="metadata path override")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:1")
    parser.add_argument("--dry-run", action="store_true", help="validate/build only")
    args = parser.parse_args()

    config = load_rt_config(args.config)
    render = config["render"]
    # Resolve once so `--device cuda:N` is validated before anything is built.
    device = resolve_device(args.device)
    scene = build_scene(config, device=str(device), dry_run=args.dry_run)
    camera = build_camera(config, device=str(device))
    environment = build_environment(config, device=str(device))
    integrator = Integrator(
        scene,
        environment=environment,
        mode=render["mode"],
        max_depth=render["max_depth"],
        samples=render["samples"],
        seed=render["seed"],
        self_hit_eps=render["self_hit_eps"],
        background=config["environment"].get("color", [0.0, 0.0, 0.0]),
        sample_batch=render["batch_samples"],
    )

    output_path = args.output or config["output"]["path"]
    if output_path is None:
        output_path = Path("outputs") / "raytracing" / f"{args.config.stem}.{config['output']['format']}"
    output_path = Path(output_path)
    metadata_path = args.metadata or config["output"]["metadata"]
    if metadata_path is None:
        metadata_path = output_path.with_suffix(".json")

    environment_report = (
        environment.as_dict() if hasattr(environment, "as_dict") else config["environment"]
    )
    meta = {
        "config_path": str(args.config),
        "scene": {
            "name": scene.name,
            "primitives": [
                {"type": type(primitive).__name__, "name": primitive.name, "id": primitive.primitive_id}
                for primitive in scene
            ],
            "report": getattr(scene, "build_report", {}),
            "checkpoint": config["scene"]["checkpoint"],
            "object": config["scene"]["object"],
            "asset": config["scene"].get("asset"),
            "query_depth": config["scene"]["query_depth"],
            "similarity": config["scene"]["similarity"],
            "material_overrides": config["scene"]["material_overrides"],
            "neuoct_material": config["scene"]["neuoct_material"],
            "neuoct_material_applied": bool(config["scene"]["neuoct_material"]),
        },
        "camera": {
            **config["camera"],
            "camera_to_world": camera.camera_to_world.tolist(),
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
        },
        "render": render,
        "environment": environment_report,
        "device": str(device),
        "output": {"path": str(output_path), "format": config["output"]["format"]},
        "dry_run": bool(args.dry_run),
        "mode": render["mode"],
        "samples": render["samples"],
        "max_depth": render["max_depth"],
        "sample_batch": render["batch_samples"],
    }
    if args.dry_run:
        write_metadata(metadata_path, meta, config=config)
        print(f"dry run ok: {len(scene)} primitive(s); metadata -> {metadata_path}")
        return 0

    buffers = integrator.render(camera, device=device)
    save_image(output_path, buffers.color, mode=render["tonemap"], exposure=render["exposure"])
    save_linear_npy(output_path.with_suffix(".npy"), buffers.color)
    # Every result carries the full set: tonemapped PNG, linear EXR, linear NPY
    # and JSON.  EXR is written through the same helper the CLI uses, so a host
    # without OpenEXR fails loudly here rather than silently omitting a file.
    exr_path = output_path.with_suffix(".exr")
    try:
        save_image(exr_path, buffers.color)
    except RuntimeError as error:
        raise RuntimeError(
            f"linear EXR output is required but {exr_path} could not be written: {error}"
        ) from error
    save_linear_npy(output_path.with_suffix(".depth.npy"), buffers.depth)
    save_linear_npy(output_path.with_suffix(".normal.npy"), buffers.normal)
    save_linear_npy(output_path.with_suffix(".albedo.npy"), buffers.albedo)
    if buffers.primitive_id is not None:
        save_linear_npy(
            output_path.with_suffix(".primid.npy"),
            np.asarray(buffers.primitive_id.detach().cpu()).astype(np.float32),
        )
    meta["hit_pixels"] = int(buffers.hit.sum())
    meta["tonemap"] = render["tonemap"]
    meta["coverage"] = buffers.coverage
    meta["secondary_hits_by_primitive"] = buffers.secondary_hits_by_primitive
    meta["declared_primitives"] = buffers.declared_primitives
    meta["accelerators"] = {
        str(primitive.primitive_id): getattr(primitive, "accelerator", "neural")
        for primitive in scene
    }
    meta["embree"] = embree_status()
    meta["transmission"] = {
        "thin_walled_paths": buffers.thin_walled_transmission_paths,
        "closed_interface_paths": buffers.closed_transmission_paths,
        "total_internal_reflection_paths": buffers.total_internal_reflection_paths,
        "neuoct_model": (
            "normal_based_two_sided_snell_single_interface"
            "_approximation_transparent_no_learned_albedo_tint"
        ),
        "closed_mesh_model": "watertight_snell_with_tir",
    }
    finite = torch.isfinite(buffers.color)
    meta["statistics"] = {
        "finite": bool(finite.all()),
        "min": float(buffers.color.min()),
        "max": float(buffers.color.max()),
        "mean": float(buffers.color.mean()),
        "nonzero_pixels": int((buffers.color.abs().sum(dim=-1) > 0).sum()),
    }
    meta["outputs"] = {
        "png": str(output_path),
        "exr": str(exr_path),
        "npy": str(output_path.with_suffix(".npy")),
        "depth_npy": str(output_path.with_suffix(".depth.npy")),
        "normal_npy": str(output_path.with_suffix(".normal.npy")),
        "albedo_npy": str(output_path.with_suffix(".albedo.npy")),
        "primid_npy": str(output_path.with_suffix(".primid.npy")),
        "metadata": str(metadata_path),
    }
    # Provenance: SHA-256 for the config, the checkpoint, the glTF plus every
    # buffer/texture it references, and the HDRI.
    input_hashes = {str(args.config): sha256_file(args.config)}
    if config["scene"]["checkpoint"]:
        input_hashes[str(config["scene"]["checkpoint"])] = sha256_file(config["scene"]["checkpoint"])
    if config["scene"].get("asset"):
        input_hashes.update(gltf_input_hashes(config["scene"]["asset"]))
    if config["environment"].get("path"):
        input_hashes[str(config["environment"]["path"])] = sha256_file(config["environment"]["path"])
    meta["input_sha256"] = input_hashes
    meta["output_sha256"] = {
        "png": sha256_file(output_path),
        "exr": sha256_file(exr_path),
        "npy": sha256_file(output_path.with_suffix(".npy")),
        "depth_npy": sha256_file(output_path.with_suffix(".depth.npy")),
        "normal_npy": sha256_file(output_path.with_suffix(".normal.npy")),
        "albedo_npy": sha256_file(output_path.with_suffix(".albedo.npy")),
        "primid_npy": sha256_file(output_path.with_suffix(".primid.npy")),
    }
    write_metadata(metadata_path, meta, config=config)
    print(
        f"rendered {tuple(buffers.color.shape)} mode={render['mode']} "
        f"samples={render['samples']} hit={int(buffers.hit.sum())} -> {output_path}"
    )
    print(f"metadata -> {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
