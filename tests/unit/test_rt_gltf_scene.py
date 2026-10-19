"""Graph-preserving glTF loading: instances, provenance, UVs and materials.

The asset used here is a miniature instanced, multi-material glTF written to a
temporary directory: a textured plate instanced twice at different translations
and a metallic cube, plus a *second* ``scene_neural`` scene that duplicates the
plate far away.  The loader must keep both instances of the plate, keep the
cube's own material, and never touch the duplicate scene.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from render_scene import build_scene, load_gltf_scene

from neuoct.rendering_rt import Rays, Scene, srgb_to_linear

TEXTURE_SIZE = 4


def _gradient_png() -> bytes:
    """A 4x4 sRGB PNG: red ramps with u, green ramps with v."""
    from PIL import Image

    pixels = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)
    for row in range(TEXTURE_SIZE):
        for col in range(TEXTURE_SIZE):
            pixels[row, col] = (
                round(255 * col / (TEXTURE_SIZE - 1)),
                round(255 * row / (TEXTURE_SIZE - 1)),
                40,
            )
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _gltf_document() -> dict:
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add_view(payload: bytes, target: int, *, align: int = 4) -> int:
        while len(blob) % align:
            blob.append(0)
        offset = len(blob)
        blob.extend(payload)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": target})
        return len(views) - 1

    def add_accessor(view: int, component_type: int, count: int, kind: str, *, minmax=None) -> int:
        accessor = {"bufferView": view, "componentType": component_type, "count": count, "type": kind}
        if minmax is not None:
            accessor["min"], accessor["max"] = minmax
        accessors.append(accessor)
        return len(accessors) - 1

    # --- plate: a quad in the z = 0 plane -----------------------------------
    plate_positions = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]], dtype=np.float32
    )
    plate_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    plate_uvs = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    plate_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
    # --- cube ---------------------------------------------------------------
    cube_positions = np.array(
        [
            [-0.25, -0.25, -0.25], [0.25, -0.25, -0.25], [0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
            [-0.25, -0.25, 0.25], [0.25, -0.25, 0.25], [0.25, 0.25, 0.25], [-0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )
    cube_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (8, 1))
    cube_uvs = np.zeros((8, 2), dtype=np.float32)
    cube_indices = np.array(
        [
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
        ],
        dtype=np.uint16,
    )

    def mesh_accessors(positions, normals, uvs, indices):
        view = add_view(positions.tobytes(), 34962)
        position = add_accessor(
            view, 5126, len(positions), "VEC3",
            minmax=[positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        )
        normal = add_accessor(add_view(normals.tobytes(), 34962), 5126, len(normals), "VEC3")
        uv = add_accessor(add_view(uvs.tobytes(), 34962), 5126, len(uvs), "VEC2")
        index = add_accessor(add_view(indices.tobytes(), 34963), 5123, len(indices), "SCALAR")
        return {"POSITION": position, "NORMAL": normal, "TEXCOORD_0": uv}, index

    plate_attributes, plate_index = mesh_accessors(
        plate_positions, plate_normals, plate_uvs, plate_indices
    )
    cube_attributes, cube_index = mesh_accessors(cube_positions, cube_normals, cube_uvs, cube_indices)

    png = base64.b64encode(_gradient_png()).decode("ascii")
    return {
        "asset": {"version": "2.0", "generator": "neuoct-rt-test"},
        "scene": 0,
        "scenes": [
            {"name": "default_scene", "nodes": [0, 1, 2]},
            {"name": "scene_neural", "nodes": [3]},
        ],
        "nodes": [
            {"name": "InstA", "mesh": 0, "translation": [1.0, 0.0, 0.0]},
            {"name": "InstB", "mesh": 0, "translation": [-1.0, 0.0, 0.0]},
            {"name": "Cube", "mesh": 1},
            {"name": "NeuralDup", "mesh": 0, "translation": [5.0, 5.0, 5.0]},
        ],
        "meshes": [
            {"name": "Plate", "primitives": [{"attributes": plate_attributes, "indices": plate_index, "material": 0}]},
            {"name": "Cube", "primitives": [{"attributes": cube_attributes, "indices": cube_index, "material": 1}]},
        ],
        "materials": [
            {
                "name": "Textured",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.4,
                },
            },
            {
                "name": "RedMetal",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.9, 0.1, 0.1, 1.0],
                    "metallicFactor": 1.0,
                    "roughnessFactor": 0.1,
                },
            },
        ],
        "textures": [{"source": 0, "sampler": 0}],
        "images": [{"uri": f"data:image/png;base64,{png}"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [
            {
                "byteLength": len(blob),
                "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode("ascii"),
            }
        ],
    }


@pytest.fixture(scope="module")
def gltf_path(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("gltf_scene")
    path = directory / "miniature.gltf"
    path.write_text(json.dumps(_gltf_document()), encoding="utf-8")
    return path


def test_default_scene_only_and_every_instance_kept(gltf_path: Path) -> None:
    data = load_gltf_scene(gltf_path)
    assert data.scene_index == 0
    assert data.scene_name == "default_scene"
    # Both plate instances and the cube, and never the duplicate scene.
    assert data.instance_names == ["InstA", "InstB", "Cube"]
    assert "NeuralDup" not in data.instance_names
    # 2 plate triangles twice plus 12 cube triangles.
    assert data.num_triangles == 16
    assert set(data.geometry_names) == {"Plate", "Cube"}
    assert data.face_instance.shape == (16,)
    assert data.face_geometry.shape == (16,)
    assert data.face_material.shape == (16,)
    assert sorted(set(data.face_instance.tolist())) == [0, 1, 2]
    assert len(set(data.face_geometry.tolist())) == 2
    assert np.array_equal(np.bincount(data.face_instance), np.array([2, 2, 12]))


def test_world_transforms_are_applied_to_vertices(gltf_path: Path) -> None:
    data = load_gltf_scene(gltf_path)
    bounds = data.bounds
    assert np.allclose(bounds[0], [-1.5, -0.5, -0.25], atol=1e-6)
    assert np.allclose(bounds[1], [1.5, 0.5, 0.25], atol=1e-6)
    # Instance provenance must follow the translated geometry, not the source.
    plate_faces = data.face_instance == 0
    plate_vertices = data.vertices[np.unique(data.faces[plate_faces])]
    assert plate_vertices[:, 0].min() == pytest.approx(0.5, abs=1e-6)
    assert plate_vertices[:, 0].max() == pytest.approx(1.5, abs=1e-6)


def test_nearest_hit_selects_the_right_instance(gltf_path: Path) -> None:
    data = load_gltf_scene(gltf_path)
    scene = Scene([build_gltf_primitive(data)])
    # Straight down onto the translated plate at x = +1.
    hit = scene.intersect(Rays(torch.tensor([[1.25, 0.0, 2.0]]), torch.tensor([[0.0, 0.0, -1.0]])))
    assert bool(hit.hit[0])
    assert data.instance_names[int(data.face_instance[hit.triangle_index[0]])] == "InstA"
    assert float(hit.position[0, 2]) == pytest.approx(0.0, abs=1e-6)
    # The other plate instance.
    hit = scene.intersect(Rays(torch.tensor([[-1.25, 0.0, 2.0]]), torch.tensor([[0.0, 0.0, -1.0]])))
    assert data.instance_names[int(data.face_instance[hit.triangle_index[0]])] == "InstB"
    # Between them the cube's front face is nearest.
    hit = scene.intersect(Rays(torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[0.0, 0.0, -1.0]])))
    assert data.instance_names[int(data.face_instance[hit.triangle_index[0]])] == "Cube"
    assert float(hit.position[0, 2]) == pytest.approx(0.25, abs=1e-6)
    # And the duplicate scene is unreachable.
    hit = scene.intersect(Rays(torch.tensor([[5.0, 5.0, 6.0]]), torch.tensor([[0.0, 0.0, -1.0]])))
    assert not bool(hit.hit[0])


def build_gltf_primitive(data):
    from neuoct.rendering_rt import CombinedMeshPrimitive

    return CombinedMeshPrimitive(
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
        primitive_id=0,
        name="miniature",
    )


def test_uv_and_texture_alignment(gltf_path: Path) -> None:
    data = load_gltf_scene(gltf_path)
    primitive = build_gltf_primitive(data)
    # The plate maps local (-0.5,-0.5) to uv (0,1) and local (0.5,0.5) to (1,0).
    # Probe a point whose interpolated uv has a known bilinear texture value.
    world_x = 1.25
    world_y = -0.25
    ray = Rays(torch.tensor([[world_x, world_y, 2.0]]), torch.tensor([[0.0, 0.0, -1.0]]))
    hit = primitive.intersect(ray)
    assert bool(hit.hit[0])
    assert hit.uv is not None
    uv = hit.uv[0].numpy()
    assert uv[0] == pytest.approx(0.75, abs=1e-6)
    assert uv[1] == pytest.approx(0.75, abs=1e-6)

    # Independent clamped-bilinear reference on the same 4x4 sRGB texture.
    from PIL import Image

    image = np.asarray(Image.open(io.BytesIO(_gradient_png())), dtype=np.float64)
    x = uv[0] * image.shape[1] - 0.5
    y = uv[1] * image.shape[0] - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    tx, ty = x - x0, y - y0
    c00 = image[max(y0, 0), max(x0, 0)]
    c10 = image[max(y0, 0), min(x0 + 1, image.shape[1] - 1)]
    c01 = image[min(y0 + 1, image.shape[0] - 1), max(x0, 0)]
    c11 = image[min(y0 + 1, image.shape[0] - 1), min(x0 + 1, image.shape[1] - 1)]
    reference = c00 * (1 - tx) * (1 - ty) + c10 * tx * (1 - ty) + c01 * (1 - tx) * ty + c11 * tx * ty
    reference = (reference / 255.0).astype(np.float32)
    expected = srgb_to_linear(torch.as_tensor(reference)).numpy()
    # ``hit.albedo`` stays tint-free (the texture decodes to linear exactly
    # once); ``Scene`` then multiplies it by the material base colour.
    assert np.allclose(hit.albedo[0].numpy(), expected, atol=2e-3)
    # The texture really is a 2-D gradient: moving up in the image (smaller v)
    # must raise the green channel while leaving the red channel alone.
    upper = primitive.intersect(
        Rays(torch.tensor([[world_x, 0.25, 2.0]]), torch.tensor([[0.0, 0.0, -1.0]]))
    )
    upper_uv = upper.uv[0].numpy()
    assert upper_uv[1] == pytest.approx(0.25, abs=1e-6)
    assert float(upper.albedo[0, 1]) < float(hit.albedo[0, 1])
    assert float(upper.albedo[0, 0]) == pytest.approx(float(hit.albedo[0, 0]), rel=1e-6)


def test_material_identity_is_per_instance(gltf_path: Path) -> None:
    data = load_gltf_scene(gltf_path)
    plate = data.materials[data.face_material[0]]
    cube = data.materials[data.face_material[-1]]
    assert plate.metallic == pytest.approx(0.0)
    assert plate.roughness == pytest.approx(0.4)
    assert plate.texture is not None
    assert cube.metallic == pytest.approx(1.0)
    assert cube.roughness == pytest.approx(0.1)
    assert torch.allclose(cube.base_color, torch.tensor([0.9, 0.1, 0.1]), atol=1e-6)
    assert cube.texture is None
    assert "Textured@InstA" in data.material_names
    assert "RedMetal@Cube" in data.material_names


def test_material_override_by_node_material_and_pattern(gltf_path: Path) -> None:
    def overrides(entry):
        return load_gltf_scene(gltf_path, material_overrides=[entry])

    by_node = overrides({"match": "InstA", "by": "node", "material": {"transmission": 0.75, "ior": 1.7}})
    assert by_node.materials[0].transmission == pytest.approx(0.75)
    assert by_node.materials[0].ior == pytest.approx(1.7)
    assert by_node.materials[1].transmission == pytest.approx(0.0)

    by_material = overrides({"match": "RedMetal", "by": "material", "material": {"roughness": 0.02}})
    assert by_material.materials[2].roughness == pytest.approx(0.02)
    assert by_material.materials[0].roughness == pytest.approx(0.4)

    by_pattern = overrides({"match": "inst", "by": "pattern", "material": {"metallic": 0.5}})
    assert by_pattern.materials[0].metallic == pytest.approx(0.5)
    assert by_pattern.materials[1].metallic == pytest.approx(0.5)
    assert by_pattern.materials[2].metallic == pytest.approx(1.0)

    # A pattern also matches the *source material* name (case-insensitively).
    by_texture_pattern = overrides({"match": "textured", "by": "pattern", "material": {"specular": 0.2}})
    assert by_texture_pattern.materials[0].specular == pytest.approx(0.2)
    assert by_texture_pattern.materials[2].specular == pytest.approx(0.04)


def test_build_scene_uses_the_asset_entry(tmp_path, gltf_path: Path) -> None:
    from neuoct.rendering_rt import load_rt_config

    config_path = tmp_path / "asset_scene.yaml"
    config_path.write_text(
        "version: 2\n"
        "scene:\n"
        f"  asset: {gltf_path}\n"
        "  material_overrides:\n"
        "    - match: Cube\n"
        "      by: node\n"
        "      material: {roughness: 0.05, transmission: 0.9, thin_walled: false}\n"
        "  similarity:\n"
        "    scale: 1.0\n"
        "    rotation: [[1,0,0],[0,1,0],[0,0,1]]\n"
        "    translation: [0,0,0]\n"
        "camera: {type: perspective, layout: look_at, position: [0,0,3], target: [0,0,0]}\n"
        "render: {width: 16, height: 16, samples: 1}\n"
        "environment: {type: constant, color: [1,1,1], intensity: 1.0}\n"
        "output: {path: out.png, format: png}\n",
        encoding="utf-8",
    )
    config = load_rt_config(config_path)
    assert config["scene"]["asset"] == str(gltf_path)
    scene = build_scene(config, device="cpu")
    assert len(scene) == 1
    primitive = scene.primitives[0]
    assert primitive.num_triangles == 16
    assert primitive.instance_names == ["InstA", "InstB", "Cube"]
    assert primitive.materials[2].roughness == pytest.approx(0.05)
    assert primitive.materials[2].transmission == pytest.approx(0.9)
    assert primitive.materials[2].thin_walled is False
