"""Scene primitives that share the ``intersect(Rays) -> RayHit`` contract.

``MeshPrimitive`` builds its own deterministic CPU BVH from transformed trimesh
geometry, interpolates vertex normals and UVs when present, and resolves a glTF
base colour factor or texture.  ``NeuOctPrimitive`` wraps the *actual*
``LocalSceneModel`` and ``LocalSparseOctree`` from a packed checkpoint and never
extracts a mesh.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .color import linear_to_srgb, srgb_to_linear
from .materials import MaterialResolver
from .mesh_bvh import BVHIntersection, TriangleBVH, accelerator_name, make_accelerator
from .neuoct_tracer import NeuOctRayTracer
from .transforms import SimilarityTransform
from .types import Material, MaterialBatch, RayHit, Rays


class Primitive(ABC):
    """Common interface for everything the :class:`~.scene.Scene` can trace."""

    name: str
    primitive_id: int

    @abstractmethod
    def intersect(self, rays: Rays) -> RayHit:  # pragma: no cover - interface
        raise NotImplementedError


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# Decoded base-colour textures, keyed by ``id(image)``.  A 2k x 2k JPEG is
# ~12 MB as float64 and decoding it once per shading call would dominate a
# render, so the decoded array is memoised.  The entry keeps a strong reference
# to the source object so its ``id`` cannot be recycled by a later allocation.
_TEXTURE_CACHE: dict[int, tuple[object, np.ndarray]] = {}


def _decode_texture(image: object) -> np.ndarray | None:
    key = id(image)
    cached = _TEXTURE_CACHE.get(key)
    if cached is not None and cached[0] is image:
        return cached[1]
    try:  # PIL image / numpy array
        array = np.asarray(image)
    except Exception:
        return None
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float64) / 255.0
    else:
        array = array.astype(np.float64)
    if array.shape[0] == 0 or array.shape[1] == 0:
        return None
    _TEXTURE_CACHE[key] = (image, array)
    return array


def _sample_texture(texture: object, uv: np.ndarray) -> np.ndarray | None:
    """Deterministically sample a base-colour texture, returning linear RGB.

    Returns ``None`` when the texture cannot be interpreted, so the caller can
    fall back to the material base colour.  UVs wrap; sampling is bilinear --
    the glTF default sampler address mode is REPEAT, which is what the chess
    asset's samplers use.
    """
    image = texture
    if image is None:
        return None
    # trimesh TextureVisuals stores the image on the material.
    for attribute in ("image", "baseColorTexture", "source"):
        candidate = getattr(image, attribute, None)
        if candidate is not None:
            image = candidate
            break
    array = _decode_texture(image)
    if array is None:
        return None
    height, width = array.shape[:2]
    u = np.mod(uv[:, 0], 1.0) * width - 0.5
    v = np.mod(uv[:, 1], 1.0) * height - 0.5
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    tx = u - x0
    ty = v - y0
    x0m = np.mod(x0, width)
    x1m = np.mod(x0 + 1, width)
    y0m = np.mod(y0, height)
    y1m = np.mod(y0 + 1, height)
    c00 = array[y0m, x0m]
    c10 = array[y0m, x1m]
    c01 = array[y1m, x0m]
    c11 = array[y1m, x1m]
    tx = tx[:, None]
    ty = ty[:, None]
    sampled = (
        c00 * (1 - tx) * (1 - ty)
        + c10 * tx * (1 - ty)
        + c01 * (1 - tx) * ty
        + c11 * tx * ty
    )
    return sampled


class MeshPrimitive(Primitive):
    def __init__(
        self,
        geometry: object,
        *,
        transform: Tensor | np.ndarray | None = None,
        material: Material | None = None,
        material_overrides: dict[str, Any] | None = None,
        material_resolver: MaterialResolver | None = None,
        primitive_id: int = 0,
        name: str | None = None,
        leaf_size: int = 4,
        orient_normals_to_ray: bool = True,
        uv_scale: tuple[float, float] | None = None,
        prefer_embree: bool = True,
    ) -> None:
        vertices = np.asarray(_numpy(geometry.vertices), dtype=np.float64)
        faces = np.asarray(_numpy(geometry.faces), dtype=np.int64)
        if transform is not None:
            matrix = np.asarray(_numpy(transform), dtype=np.float64).reshape(4, 4)
            linear = matrix[:3, :3]
            vertices = vertices @ linear.T + matrix[:3, 3]
            normal_matrix = np.linalg.inv(linear).T
        else:
            matrix = None
            normal_matrix = None
        self.geometry = geometry
        self.transform = matrix
        self.faces = faces
        self.primitive_id = int(primitive_id)
        self.name = name or f"mesh_{self.primitive_id}"
        self.orient_normals_to_ray = bool(orient_normals_to_ray)
        self.uv_scale = uv_scale
        self.vertex_normals = self._transform_normals(
            self._vertex_normals(geometry), normal_matrix
        )
        self.face_normals = self._face_normals(vertices, faces)
        self.uv = self._vertex_uv(geometry, uv_scale)
        self.material = material or (material_resolver or MaterialResolver()).resolve_mesh(
            geometry, material_overrides
        )
        self.bvh: TriangleBVH | None = None
        if faces.shape[0] > 0:
            self.bvh = make_accelerator(
                vertices, faces, prefer_embree=prefer_embree, leaf_size=leaf_size
            )

    @property
    def accelerator(self) -> str:
        """Name of the accelerator actually backing this primitive."""
        return "none" if self.bvh is None else accelerator_name(self.bvh)

    @staticmethod
    def _vertex_normals(geometry: object) -> np.ndarray | None:
        normals = getattr(geometry, "vertex_normals", None)
        if normals is None:
            return None
        normals = np.asarray(_numpy(normals), dtype=np.float64)
        if normals.shape != (len(geometry.vertices), 3):
            return None
        return normals

    @staticmethod
    def _transform_normals(normals: np.ndarray | None, normal_matrix: np.ndarray | None) -> np.ndarray | None:
        if normals is None or normal_matrix is None:
            return normals
        return normals @ normal_matrix.T

    @staticmethod
    def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        if faces.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        triangles = vertices[faces]
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(edge1, edge2)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(lengths < 1.0e-15, 1.0, lengths)

    @staticmethod
    def _vertex_uv(geometry: object, uv_scale: tuple[float, float] | None) -> np.ndarray | None:
        visual = getattr(geometry, "visual", None)
        uv = getattr(visual, "uv", None)
        if uv is None:
            return None
        uv = np.asarray(_numpy(uv), dtype=np.float64)
        if uv.shape != (len(geometry.vertices), 2):
            return None
        if uv_scale is not None:
            uv = uv * np.asarray(uv_scale, dtype=np.float64)
        return uv

    @property
    def num_triangles(self) -> int:
        return 0 if self.bvh is None else self.bvh.num_triangles

    def _interpolate_normals(
        self, triangles: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        w = 1.0 - u - v
        if self.vertex_normals is None:
            normals = self.face_normals[triangles]
        else:
            n0 = self.vertex_normals[self.faces[triangles, 0]]
            n1 = self.vertex_normals[self.faces[triangles, 1]]
            n2 = self.vertex_normals[self.faces[triangles, 2]]
            normals = w[:, None] * n0 + u[:, None] * n1 + v[:, None] * n2
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(lengths < 1.0e-15, 1.0, lengths)

    def _albedo(self, triangles: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Per-hit diffuse reflectance in linear space, without the material tint.

        The integrator multiplies this by ``Material.base_color`` so the texture
        and the base-colour factor compose exactly once, mirroring how a NeuOct
        prediction is tinted.  With no texture this is white, which reduces to
        the plain base colour after tinting.
        """
        color = np.ones((triangles.shape[0], 3), dtype=np.float64)
        if self.material.texture is not None and self.uv is not None:
            w = 1.0 - u - v
            uv0 = self.uv[self.faces[triangles, 0]]
            uv1 = self.uv[self.faces[triangles, 1]]
            uv2 = self.uv[self.faces[triangles, 2]]
            uv = w[:, None] * uv0 + u[:, None] * uv1 + v[:, None] * uv2
            sampled = _sample_texture(self.material.texture, uv)
            if sampled is not None:
                color = srgb_to_linear(torch.as_tensor(sampled)).numpy()
        return color

    def intersect(self, rays: Rays) -> RayHit:
        if self.bvh is None or rays.num_rays == 0:
            result = RayHit.miss(rays.num_rays, device=rays.device)
            return result
        origins = _numpy(rays.origins).astype(np.float64)
        directions = _numpy(rays.directions).astype(np.float64)
        t_min = _numpy(rays.t_min).astype(np.float64)
        t_max = _numpy(rays.t_max).astype(np.float64)
        intersection: BVHIntersection = self.bvh.intersect(
            origins, directions, t_min=t_min, t_max=t_max
        )
        count = rays.num_rays
        device = rays.device
        hit = torch.as_tensor(intersection.hit, device=device)
        if not bool(hit.any()):
            return RayHit.miss(count, device=device)
        indices = np.nonzero(intersection.hit)[0]
        triangles = np.asarray(intersection.triangle_index[indices], dtype=np.int64)
        u = np.asarray(intersection.u[indices], dtype=np.float64)
        v = np.asarray(intersection.v[indices], dtype=np.float64)
        w = 1.0 - u - v
        t = np.asarray(intersection.t[indices], dtype=np.float64)
        positions = origins[indices] + t[:, None] * directions[indices]
        normals = self._interpolate_normals(triangles, u, v)
        if self.orient_normals_to_ray:
            flip = np.einsum("ij,ij->i", normals, directions[indices]) > 0.0
            normals = np.where(flip[:, None], -normals, normals)
        albedo = self._albedo(triangles, u, v)
        linear_albedo = torch.as_tensor(albedo, dtype=torch.float32, device=device)
        uv_hit = None
        if self.uv is not None:
            uv_hit = (
                w[:, None] * self.uv[self.faces[triangles, 0]]
                + u[:, None] * self.uv[self.faces[triangles, 1]]
                + v[:, None] * self.uv[self.faces[triangles, 2]]
            )
        # Two-sided orientation of the closed surface, taken from the winding
        # *before* any ray-facing flip.  ``front_face`` lets the integrator pick
        # eta_i / eta_t for real Snell refraction with total internal reflection.
        geometric = self.face_normals[triangles]
        front_face = np.einsum("ij,ij->i", geometric, directions[indices]) < 0.0

        miss = RayHit.miss(count, device=device)
        hit_tensor = miss.hit.clone()
        t_tensor = miss.t.clone()
        position_tensor = miss.position.clone()
        normal_tensor = miss.normal.clone()
        bary_tensor = miss.barycentric.clone()
        tri_tensor = miss.triangle_index.clone()
        albedo_tensor = miss.albedo.clone()
        albedo_srgb_tensor = miss.albedo_srgb.clone()
        prob_tensor = miss.hit_probability.clone()
        uv_tensor = None if uv_hit is None else torch.zeros(count, 2, device=device)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        hit_tensor[index_tensor] = True
        t_tensor[index_tensor] = torch.as_tensor(t, dtype=torch.float32, device=device)
        position_tensor[index_tensor] = torch.as_tensor(positions, dtype=torch.float32, device=device)
        normal_tensor[index_tensor] = torch.as_tensor(normals, dtype=torch.float32, device=device)
        bary_tensor[index_tensor] = torch.as_tensor(
            np.stack([w, u, v], axis=1), dtype=torch.float32, device=device
        )
        tri_tensor[index_tensor] = torch.as_tensor(triangles, dtype=torch.long, device=device)
        albedo_tensor[index_tensor] = linear_albedo
        albedo_srgb_tensor[index_tensor] = linear_to_srgb(linear_albedo)
        prob_tensor[index_tensor] = 1.0
        if uv_tensor is not None and uv_hit is not None:
            uv_tensor[index_tensor] = torch.as_tensor(uv_hit, dtype=torch.float32, device=device)
        geometric_tensor = miss.position.clone()
        geometric_tensor[index_tensor] = torch.as_tensor(
            geometric, dtype=torch.float32, device=device
        )
        front_tensor = torch.zeros(count, dtype=torch.bool, device=device)
        front_tensor[index_tensor] = torch.as_tensor(front_face, device=device)
        return RayHit(
            hit=hit_tensor,
            t=t_tensor,
            position=position_tensor,
            normal=normal_tensor,
            barycentric=bary_tensor,
            triangle_index=tri_tensor,
            primitive_id=torch.full((count,), self.primitive_id, dtype=torch.long, device=device),
            albedo=albedo_tensor,
            albedo_srgb=albedo_srgb_tensor,
            hit_probability=prob_tensor,
            uv=uv_tensor,
            material=self.material.to(device),
            materials=MaterialBatch.from_material(self.material, count, device=device),
            geometric_normal=geometric_tensor,
            front_face=front_tensor,
        )


def _scatter_hit(count: int, index: Tensor, hit: RayHit, device) -> RayHit:
    """Expand a sub-batch hit back onto ``count`` rays; the rest are misses."""
    if index.numel() == count:
        return hit
    miss = RayHit.miss(count, device=device)
    out = miss.hit.clone()
    out[index] = hit.hit
    t = miss.t.clone()
    t[index] = hit.t
    position = miss.position.clone()
    position[index] = hit.position
    normal = miss.normal.clone()
    normal[index] = hit.normal
    bary = miss.barycentric.clone()
    bary[index] = hit.barycentric
    tri = miss.triangle_index.clone()
    tri[index] = hit.triangle_index
    pid = miss.primitive_id.clone()
    pid[index] = hit.primitive_id
    albedo = miss.albedo.clone()
    albedo[index] = hit.albedo
    albedo_srgb = miss.albedo_srgb.clone()
    albedo_srgb[index] = hit.albedo_srgb
    prob = miss.hit_probability.clone()
    prob[index] = hit.hit_probability
    uv = None
    if hit.uv is not None:
        uv = torch.zeros(count, 2, device=device)
        uv[index] = hit.uv
    geometric = None
    if hit.geometric_normal is not None:
        geometric = miss.position.clone()
        geometric[index] = hit.geometric_normal
    front = None
    if hit.front_face is not None:
        front = torch.zeros(count, dtype=torch.bool, device=device)
        front[index] = hit.front_face
    materials = None
    if hit.materials is not None:
        materials = hit.materials  # already one row per *traced* ray
        padded = MaterialBatch(
            base_color=torch.zeros(count, 3, device=device),
            roughness=torch.ones(count, device=device),
            metallic=torch.zeros(count, device=device),
            specular=torch.zeros(count, device=device),
            transmission=torch.zeros(count, device=device),
            ior=torch.ones(count, device=device),
            thin_walled=torch.ones(count, dtype=torch.bool, device=device),
            emission=torch.zeros(count, 3, device=device),
            base_color_srgb=torch.zeros(count, dtype=torch.bool, device=device),
        )
        for name in (
            "base_color",
            "roughness",
            "metallic",
            "specular",
            "transmission",
            "ior",
            "thin_walled",
            "emission",
        ):
            getattr(padded, name)[index] = getattr(materials, name)
        srgb = materials.base_color_srgb
        if srgb is not None:
            padded.base_color_srgb[index] = srgb
        materials = padded
    return RayHit(
        hit=out,
        t=t,
        position=position,
        normal=normal,
        barycentric=bary,
        triangle_index=tri,
        primitive_id=pid,
        albedo=albedo,
        albedo_srgb=albedo_srgb,
        hit_probability=prob,
        uv=uv,
        material=hit.material,
        materials=materials,
        geometric_normal=geometric,
        front_face=front,
    )


def material_table_batch(materials: list[Material], *, device) -> MaterialBatch:
    """Stack per-material parameters into one indexable :class:`MaterialBatch`."""
    if not materials:
        raise ValueError("material_table_batch requires at least one material")
    return MaterialBatch(
        base_color=torch.stack([material.base_color for material in materials]).to(device),
        roughness=torch.tensor([material.roughness for material in materials], dtype=torch.float32, device=device),
        metallic=torch.tensor([material.metallic for material in materials], dtype=torch.float32, device=device),
        specular=torch.tensor([material.specular for material in materials], dtype=torch.float32, device=device),
        transmission=torch.tensor(
            [material.transmission for material in materials], dtype=torch.float32, device=device
        ),
        ior=torch.tensor([material.ior for material in materials], dtype=torch.float32, device=device),
        thin_walled=torch.tensor(
            [material.thin_walled for material in materials], dtype=torch.bool, device=device
        ),
        emission=torch.stack([material.emission for material in materials]).to(device),
        base_color_srgb=torch.tensor(
            [material.base_color_srgb for material in materials], dtype=torch.bool, device=device
        ),
    )


class CombinedMeshPrimitive(Primitive):
    """One world-space mesh accelerator built from many glTF instances.

    A faithful glTF load must keep every *instance* of a shared geometry while
    preserving per-face provenance.  Flattening all instances into a single
    world-space triangle soup is the cheapest correct accelerator, so this
    primitive carries one BVH over the combined vertices/faces plus three
    parallel per-face arrays:

    ``face_material``
        index into ``materials``/``textures`` (glTF ``material`` identity),
    ``face_instance``
        index into ``instance_names`` (glTF node provenance),
    ``face_geometry``
        index into ``geometry_names`` (source glTF mesh provenance).

    The combined mesh keeps vertex normals, UVs and per-instance materials, so
    nothing that matters for shading is lost by flattening duplicate instances.
    """

    def __init__(
        self,
        vertices,
        faces,
        *,
        face_material,
        materials: list[Material],
        face_instance=None,
        face_geometry=None,
        instance_names: list[str] | None = None,
        geometry_names: list[str] | None = None,
        vertex_normals=None,
        vertex_uv=None,
        primitive_id: int = 0,
        name: str = "combined_mesh",
        leaf_size: int = 32,
        orient_normals_to_ray: bool = True,
        prefer_embree: bool = True,
    ) -> None:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        face_material = np.asarray(face_material, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if face_material.shape != (faces.shape[0],):
            raise ValueError("face_material must have one entry per face")
        if not materials:
            raise ValueError("CombinedMeshPrimitive requires at least one material")
        if int(face_material.max(initial=0)) >= len(materials):
            raise ValueError("face_material references a material index out of range")
        if faces.shape[0] == 0:
            raise ValueError("CombinedMeshPrimitive requires at least one face")
        self.vertices = vertices
        self.faces = faces
        self.face_material = face_material
        self.materials = list(materials)
        self.face_instance = (
            np.zeros(faces.shape[0], dtype=np.int64)
            if face_instance is None
            else np.asarray(face_instance, dtype=np.int64)
        )
        self.face_geometry = (
            np.zeros(faces.shape[0], dtype=np.int64)
            if face_geometry is None
            else np.asarray(face_geometry, dtype=np.int64)
        )
        self.instance_names = list(instance_names or [name])
        self.geometry_names = list(geometry_names or [name])
        self.primitive_id = int(primitive_id)
        self.name = name
        self.orient_normals_to_ray = bool(orient_normals_to_ray)
        normals = None if vertex_normals is None else np.asarray(vertex_normals, dtype=np.float64)
        if normals is not None and normals.shape != vertices.shape:
            normals = None
        self.vertex_normals = normals
        uv = None if vertex_uv is None else np.asarray(vertex_uv, dtype=np.float64)
        if uv is not None and uv.shape != (vertices.shape[0], 2):
            uv = None
        self.vertex_uv = uv
        self._material_tables = [material.texture for material in self.materials]
        self.face_normals = MeshPrimitive._face_normals(vertices, faces)
        self.bvh = make_accelerator(
            vertices, faces, prefer_embree=prefer_embree, leaf_size=leaf_size
        )

    @property
    def accelerator(self) -> str:
        """Name of the accelerator actually backing this primitive."""
        return accelerator_name(self.bvh)

    @property
    def num_triangles(self) -> int:
        return self.bvh.num_triangles

    @property
    def num_materials(self) -> int:
        return len(self.materials)

    @property
    def num_instances(self) -> int:
        return len(self.instance_names)

    def coverage(self, primitive_face: np.ndarray) -> dict[str, int]:
        """Face-hit histogram keyed by source geometry name (diagnostics only)."""
        counts: dict[str, int] = {}
        for index in np.unique(primitive_face):
            name = self.geometry_names[int(index)] if int(index) < len(self.geometry_names) else str(index)
            counts[name] = int((primitive_face == index).sum())
        return counts

    def intersect(self, rays: Rays) -> RayHit:
        count = rays.num_rays
        device = rays.device
        if count == 0:
            return RayHit.miss(0, device=device)
        origins = _numpy(rays.origins).astype(np.float64)
        directions = _numpy(rays.directions).astype(np.float64)
        t_min = _numpy(rays.t_min).astype(np.float64)
        t_max = _numpy(rays.t_max).astype(np.float64)
        intersection: BVHIntersection = self.bvh.intersect(
            origins, directions, t_min=t_min, t_max=t_max
        )
        if not bool(intersection.hit.any()):
            return RayHit.miss(count, device=device)
        indices = np.nonzero(intersection.hit)[0]
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        triangles = np.asarray(intersection.triangle_index[indices], dtype=np.int64)
        u = np.asarray(intersection.u[indices], dtype=np.float64)
        v = np.asarray(intersection.v[indices], dtype=np.float64)
        w = 1.0 - u - v
        t = np.asarray(intersection.t[indices], dtype=np.float64)
        positions = origins[indices] + t[:, None] * directions[indices]
        ray_directions = directions[indices]
        geometric = self.face_normals[triangles]
        front_face = np.einsum("ij,ij->i", geometric, ray_directions) < 0.0
        if self.vertex_normals is None:
            normals = geometric.copy()
        else:
            n0 = self.vertex_normals[self.faces[triangles, 0]]
            n1 = self.vertex_normals[self.faces[triangles, 1]]
            n2 = self.vertex_normals[self.faces[triangles, 2]]
            normals = w[:, None] * n0 + u[:, None] * n1 + v[:, None] * n2
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(lengths < 1.0e-15, 1.0, lengths)
        if self.orient_normals_to_ray:
            flip = np.einsum("ij,ij->i", normals, ray_directions) > 0.0
            normals = np.where(flip[:, None], -normals, normals)
        # Per-hit albedo: texture (linear) or white.  The material tint is
        # applied exactly once by the integrator, exactly like MeshPrimitive.
        albedo = np.ones((indices.size, 3), dtype=np.float64)
        uv_hit = None
        if self.vertex_uv is not None:
            uv_hit = (
                w[:, None] * self.vertex_uv[self.faces[triangles, 0]]
                + u[:, None] * self.vertex_uv[self.faces[triangles, 1]]
                + v[:, None] * self.vertex_uv[self.faces[triangles, 2]]
            )
            hit_material = self.face_material[triangles]
            for material_index in np.unique(hit_material):
                texture = self._material_tables[int(material_index)]
                if texture is None:
                    continue
                group = hit_material == material_index
                sampled = _sample_texture(texture, uv_hit[group])
                if sampled is not None:
                    albedo[group] = srgb_to_linear(torch.as_tensor(sampled)).numpy()
        hit_material = self.face_material[triangles]
        # ``RayHit.materials`` must have one row per *ray*, not one row per hit:
        # the scene gathers with row indices into the full ray batch.  Misses
        # keep material 0, which is never read because ``hit`` is False there.
        table = material_table_batch(self.materials, device=device)
        material_index = torch.zeros(count, dtype=torch.long, device=device)
        material_index[index_tensor] = torch.as_tensor(
            hit_material, dtype=torch.long, device=device
        )
        materials = table.index(material_index)

        miss = RayHit.miss(count, device=device)
        hit_tensor = miss.hit.clone()
        hit_tensor[index_tensor] = True
        t_tensor = miss.t.clone()
        t_tensor[index_tensor] = torch.as_tensor(t, dtype=torch.float32, device=device)
        position_tensor = miss.position.clone()
        position_tensor[index_tensor] = torch.as_tensor(positions, dtype=torch.float32, device=device)
        normal_tensor = miss.normal.clone()
        normal_tensor[index_tensor] = torch.as_tensor(normals, dtype=torch.float32, device=device)
        geometric_tensor = miss.position.clone()
        geometric_tensor[index_tensor] = torch.as_tensor(geometric, dtype=torch.float32, device=device)
        front_tensor = torch.zeros(count, dtype=torch.bool, device=device)
        front_tensor[index_tensor] = torch.as_tensor(front_face, device=device)
        bary_tensor = miss.barycentric.clone()
        bary_tensor[index_tensor] = torch.as_tensor(
            np.stack([w, u, v], axis=1), dtype=torch.float32, device=device
        )
        tri_tensor = miss.triangle_index.clone()
        tri_tensor[index_tensor] = torch.as_tensor(triangles, dtype=torch.long, device=device)
        albedo_tensor = miss.albedo.clone()
        linear_albedo = torch.as_tensor(albedo, dtype=torch.float32, device=device)
        albedo_tensor[index_tensor] = linear_albedo
        albedo_srgb_tensor = miss.albedo_srgb.clone()
        albedo_srgb_tensor[index_tensor] = linear_to_srgb(linear_albedo)
        prob_tensor = miss.hit_probability.clone()
        prob_tensor[index_tensor] = 1.0
        uv_tensor = None
        if uv_hit is not None:
            uv_tensor = torch.zeros(count, 2, device=device)
            uv_tensor[index_tensor] = torch.as_tensor(uv_hit, dtype=torch.float32, device=device)
        return RayHit(
            hit=hit_tensor,
            t=t_tensor,
            position=position_tensor,
            normal=normal_tensor,
            barycentric=bary_tensor,
            triangle_index=tri_tensor,
            primitive_id=torch.full((count,), self.primitive_id, dtype=torch.long, device=device),
            albedo=albedo_tensor,
            albedo_srgb=albedo_srgb_tensor,
            hit_probability=prob_tensor,
            uv=uv_tensor,
            material=None,
            materials=materials,
            geometric_normal=geometric_tensor,
            front_face=front_tensor,
        )


class NeuOctPrimitive(Primitive):
    """A checkpoint-backed NeuOct scene.  No mesh extraction is performed."""

    def __init__(
        self,
        model,
        octree,
        *,
        depth: int,
        point_scale: float = 0.99,
        ray_scale: float = 0.5,
        object_to_scene: SimilarityTransform | None = None,
        hit_threshold: float = 0.5,
        ray_chunk_size: int = 16384,
        candidate_chunk_size: int = 65536,
        self_hit_eps: float = 1.0e-4,
        material: Material | None = None,
        material_overrides: dict[str, Any] | None = None,
        material_resolver: MaterialResolver | None = None,
        primitive_id: int = 0,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.octree = octree
        self.depth = int(depth)
        self.primitive_id = int(primitive_id)
        self.name = name or f"neuoct_{self.primitive_id}"
        self.material = material or (material_resolver or MaterialResolver()).resolve_neuoct(
            material_overrides
        )
        self.tracer = NeuOctRayTracer(
            model,
            octree,
            depth=self.depth,
            point_scale=point_scale,
            ray_scale=ray_scale,
            object_to_scene=object_to_scene,
            hit_threshold=hit_threshold,
            ray_chunk_size=ray_chunk_size,
            candidate_chunk_size=candidate_chunk_size,
            self_hit_eps=self_hit_eps,
            material=self.material,
            primitive_id=self.primitive_id,
        )
        self.cull_bounds = self._scene_bounds()
        self.culled_rays = 0
        self.traced_rays = 0

    def _scene_bounds(self) -> tuple[Tensor, Tensor] | None:
        """World-space AABB of the octree grid, for ray rejection.

        A NeuOct hit is always a position *inside* one of the occupied voxels,
        hence inside the ``[0, 2**depth]`` integer grid.  A ray that misses the
        grid's world-space box therefore cannot produce an accepted hit, so the
        whole neural traversal can be skipped for it.  This is a pure
        acceleration: it removes no candidate that could ever be accepted.
        """
        transform = self.tracer.octree_transform
        similarity = self.tracer.object_to_scene
        extent = float(transform.scale)
        corners = torch.tensor(
            [
                [x, y, z]
                for x in (0.0, extent)
                for y in (0.0, extent)
                for z in (0.0, extent)
            ],
            dtype=torch.float32,
        )
        object_corners = transform.octree_to_object(corners)
        scene_corners = similarity.transform_points(object_corners)
        return scene_corners.min(dim=0).values, scene_corners.max(dim=0).values

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        object_key: str | None = None,
        device: torch.device | str = "cpu",
        object_to_scene: SimilarityTransform | None = None,
        **kwargs: Any,
    ) -> "NeuOctPrimitive":
        # Lazy import so the module does not require the OCNN/triton stack.
        from neuoct.engine.local_checkpoint import load_local_inference_checkpoint

        model, octree, payload = load_local_inference_checkpoint(
            checkpoint, device=device, expected_object=object_key
        )
        model.eval()
        payload_depth = int(payload["query_depth"])
        requested_depth = kwargs.pop("depth", None)
        if requested_depth is not None and int(requested_depth) != payload_depth:
            raise ValueError(
                f"Configured query depth {requested_depth} does not match checkpoint "
                f"query depth {payload_depth} for {checkpoint}"
            )
        return cls(model, octree, depth=payload_depth, object_to_scene=object_to_scene, **kwargs)

    def intersect(self, rays: Rays) -> RayHit:
        if self.cull_bounds is None or rays.num_rays == 0:
            return self.tracer.trace(rays)
        lower, upper = self.cull_bounds
        lower = lower.to(rays.device)
        upper = upper.to(rays.device)
        origins = rays.origins
        directions = rays.directions
        t_min = rays.t_min.unsqueeze(-1)
        t_max = rays.t_max.unsqueeze(-1)
        # Slab test in scene space, with the ray's own [t_min, t_max] window.
        safe = torch.where(
            directions.abs() < 1.0e-12,
            torch.full_like(directions, 1.0e-12) * torch.sign(directions + 1.0e-30),
            directions,
        )
        inverse = 1.0 / safe
        near = (lower - origins) * inverse
        far = (upper - origins) * inverse
        enter = torch.minimum(near, far).max(dim=-1).values
        leave = torch.maximum(near, far).min(dim=-1).values
        keep = (leave >= enter) & (leave >= t_min.squeeze(-1)) & (enter <= t_max.squeeze(-1))
        index = torch.nonzero(keep, as_tuple=False).squeeze(1)
        self.culled_rays += int(rays.num_rays - index.numel())
        self.traced_rays += int(index.numel())
        if index.numel() == 0:
            return RayHit.miss(rays.num_rays, device=rays.device)
        subset = rays.index(index)
        hit = self.tracer.trace(subset)
        return _scatter_hit(rays.num_rays, index, hit, rays.device)
