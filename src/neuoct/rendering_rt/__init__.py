"""NeuOct phase-one offline ray tracer.

A self-contained, deterministic, CPU-friendly ray tracer that renders both
triangle meshes and packed NeuOct local checkpoints.  The package never
extracts a mesh from a NeuOct checkpoint: ``NeuOctPrimitive`` drives the actual
``LocalSceneModel`` on traversed voxel segments.

See :mod:`neuoct.rendering_rt.config` for the isolated YAML schema and
``tools/neuoct_rt`` for the offline command-line entry points.
"""

from __future__ import annotations

from .camera import Camera, look_at_matrix
from .color import (
    aces,
    linear_to_display,
    linear_to_srgb,
    reinhard,
    srgb_to_linear,
    tonemap,
)
from .config import (
    RT_CONFIG_VERSION,
    load_rt_config,
    validate_rt_config,
)
from .hdri import (
    DEFAULT_CHANNEL_CAP,
    LatLongHDRI,
    direction_to_uv,
    environment_from_config,
    load_exr_rgb,
    rotation_y_matrix,
    uv_to_direction,
)
from .integrator import (
    ConstantEnvironment,
    Environment,
    Integrator,
    RenderBuffers,
    render_scene,
)
from .io import (
    canonical_json,
    config_hash,
    linear_to_uint8,
    save_exr,
    save_image,
    save_linear_npy,
    save_png,
    write_metadata,
)
from .materials import NEUOCT_DEFAULTS, MaterialResolver
from .mesh_bvh import (
    BVHIntersection,
    EmbreeBVH,
    TriangleBVH,
    accelerator_name,
    embree_status,
    make_accelerator,
)
from .neuoct_tracer import NeuOctRayTracer, OctreeTraceResult, trace_octree
from .primitives import (
    CombinedMeshPrimitive,
    MeshPrimitive,
    NeuOctPrimitive,
    Primitive,
    material_table_batch,
)
from .rng import CounterRNG, counter_seed_from
from .sampling import (
    RefractionResult,
    SnellRefraction,
    cosine_hemisphere_sample,
    dielectric_f0,
    ggx_half_vector,
    ggx_reflection,
    mirror_reflection,
    orient_normal_to_ray,
    orthonormal_basis,
    reflect,
    refract,
    schlick_fresnel,
    snell_refract,
    thin_walled_refract,
)
from .scene import Scene
from .transforms import (
    Z_UP_TO_Y_UP,
    OctreeTransform,
    SimilarityTransform,
)
from .types import (
    Material,
    MaterialBatch,
    RayHit,
    Rays,
    gather_ray_directions,
    resolve_device,
    stable_lexsort,
)

__all__ = [
    "BVHIntersection",
    "Camera",
    "CombinedMeshPrimitive",
    "ConstantEnvironment",
    "CounterRNG",
    "DEFAULT_CHANNEL_CAP",
    "EmbreeBVH",
    "Environment",
    "Integrator",
    "LatLongHDRI",
    "Material",
    "MaterialBatch",
    "MaterialResolver",
    "MeshPrimitive",
    "NEUOCT_DEFAULTS",
    "NeuOctPrimitive",
    "NeuOctRayTracer",
    "OctreeTraceResult",
    "OctreeTransform",
    "Primitive",
    "RT_CONFIG_VERSION",
    "RayHit",
    "Rays",
    "RefractionResult",
    "RenderBuffers",
    "Scene",
    "SimilarityTransform",
    "SnellRefraction",
    "TriangleBVH",
    "Z_UP_TO_Y_UP",
    "aces",
    "accelerator_name",
    "canonical_json",
    "config_hash",
    "cosine_hemisphere_sample",
    "counter_seed_from",
    "dielectric_f0",
    "direction_to_uv",
    "embree_status",
    "environment_from_config",
    "gather_ray_directions",
    "ggx_half_vector",
    "ggx_reflection",
    "linear_to_display",
    "linear_to_srgb",
    "linear_to_uint8",
    "load_exr_rgb",
    "load_rt_config",
    "look_at_matrix",
    "make_accelerator",
    "material_table_batch",
    "mirror_reflection",
    "orient_normal_to_ray",
    "orthonormal_basis",
    "reflect",
    "refract",
    "render_scene",
    "reinhard",
    "resolve_device",
    "rotation_y_matrix",
    "save_exr",
    "save_image",
    "save_linear_npy",
    "save_png",
    "schlick_fresnel",
    "snell_refract",
    "srgb_to_linear",
    "stable_lexsort",
    "thin_walled_refract",
    "tonemap",
    "trace_octree",
    "uv_to_direction",
    "validate_rt_config",
    "write_metadata",
]

__version__ = "0.2.0"
