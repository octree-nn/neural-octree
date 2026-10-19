from __future__ import annotations

import numpy as np
import torch

from neuoct.rendering_rt import (
    Material,
    MeshPrimitive,
    NeuOctPrimitive,
    OctreeTransform,
    Rays,
    Scene,
    srgb_to_linear,
)


def _triangle(z: float) -> object:
    import trimesh

    vertices = np.array([[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]], dtype=np.float64)
    return trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)


def _ray(origin=(0.25, 0.25, 1.0), direction=(0.0, 0.0, -1.0)) -> Rays:
    return Rays(torch.tensor([origin], dtype=torch.float32), torch.tensor([direction], dtype=torch.float32))


def test_scene_selects_closest_primitive() -> None:
    near = MeshPrimitive(_triangle(0.0), primitive_id=0, material=Material(base_color=torch.tensor([1.0, 0.0, 0.0])))
    far = MeshPrimitive(_triangle(-1.0), primitive_id=1, material=Material(base_color=torch.tensor([0.0, 1.0, 0.0])))
    scene = Scene([far, near])
    hit = scene.intersect(_ray())
    assert bool(hit.hit[0])
    assert hit.primitive_id[0].item() == 0
    assert hit.position[0, 2].item() == 0.0
    assert torch.allclose(hit.materials.base_color[0], torch.tensor([1.0, 0.0, 0.0]))


def test_scene_tie_break_prefers_lower_primitive_id() -> None:
    first = MeshPrimitive(_triangle(0.0), primitive_id=0)
    second = MeshPrimitive(_triangle(0.0), primitive_id=1)
    scene = Scene([second, first])
    hit = scene.intersect(_ray())
    assert hit.primitive_id[0].item() == 0


def test_scene_empty_and_miss() -> None:
    empty = Scene([])
    hit = empty.intersect(_ray())
    assert not bool(hit.hit[0])
    assert torch.isinf(hit.t[0])
    assert hit.primitive_id[0].item() == -1
    single_miss = Scene([MeshPrimitive(_triangle(0.0), primitive_id=0)]).intersect(
        _ray(origin=(3.0, 3.0, 1.0))
    )
    assert not bool(single_miss.hit[0])


def test_scene_single_primitive_passthrough() -> None:
    primitive = MeshPrimitive(_triangle(0.0), primitive_id=7)
    rays = _ray()
    direct = primitive.intersect(rays)
    through = Scene([primitive]).intersect(rays)
    assert torch.equal(direct.hit, through.hit)
    assert torch.allclose(direct.t[direct.hit], through.t[through.hit])
    assert torch.equal(direct.primitive_id, through.primitive_id)


def _synthetic_neuoct(primitive_id: int = 0) -> NeuOctPrimitive:
    from neuoct.geometry.local_octree import LocalSparseOctree
    from neuoct.geometry.vertex_index import compute_vertex_index
    from neuoct.models.local_scene import LocalSceneModel

    depth = 4
    coordinates = torch.tensor([[4, 4, 4], [4, 4, 5], [5, 4, 4]], dtype=torch.long)
    octree = LocalSparseOctree(coordinates, depth, "cpu")
    index = compute_vertex_index(octree, depth, include_batch=True)
    torch.manual_seed(0)
    model = LocalSceneModel(index, feature_dim=4, hidden_dim=8, n_iters=2).eval()
    return NeuOctPrimitive(
        model,
        octree,
        depth=depth,
        hit_threshold=0.0,
        primitive_id=primitive_id,
        material_overrides={"base_color": [0.1, 0.2, 0.9]},
    )


def test_scene_mixes_mesh_and_neuoct_and_keeps_nearest() -> None:
    mesh = MeshPrimitive(
        _triangle(-100.0),
        primitive_id=0,
        material=Material(base_color=torch.tensor([1.0, 1.0, 1.0])),
    )
    neuoct = _synthetic_neuoct(primitive_id=1)
    transform = OctreeTransform(depth=4)
    object_origin = transform.octree_to_object(torch.tensor([[4.5, 4.5, 4.5]]))
    rays = Rays(object_origin, torch.tensor([[0.0, 0.0, 1.0]]))
    scene = Scene([mesh, neuoct])
    hit = scene.intersect(rays)
    assert bool(hit.hit[0])
    assert hit.primitive_id[0].item() == 1
    assert torch.allclose(hit.materials.base_color[0], torch.tensor([0.1, 0.2, 0.9]), atol=1.0e-6)
    # The NeuOct model predicts its own sRGB albedo; the primitive material
    # override controls the shading parameters, not the network colour.
    assert torch.allclose(
        hit.albedo[0], srgb_to_linear(hit.albedo_srgb[0]), atol=1.0e-6
    )


def test_scene_result_matches_per_primitive_nearest() -> None:
    mesh = MeshPrimitive(_triangle(0.05), primitive_id=0)
    neuoct = _synthetic_neuoct(primitive_id=1)
    transform = OctreeTransform(depth=4)
    object_origin = transform.octree_to_object(torch.tensor([[4.5, 4.5, 4.5]]))
    rays = Rays(object_origin, torch.tensor([[0.0, 0.0, 1.0]]))
    scene = Scene([mesh, neuoct])
    combined = scene.intersect(rays)
    mesh_hit = mesh.intersect(rays)
    neu_hit = neuoct.intersect(rays)
    candidates = [(mesh_hit, 0), (neu_hit, 1)]
    hits = [(hit.t[0].item(), pid) for hit, pid in candidates if bool(hit.hit[0])]
    if not hits:
        assert not bool(combined.hit[0])
    else:
        expected_t, expected_pid = min(hits)
        assert bool(combined.hit[0])
        assert combined.primitive_id[0].item() == expected_pid
        assert combined.t[0].item() == expected_t
