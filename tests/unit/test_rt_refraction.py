"""Snell refraction, total internal reflection and the documented thin wall.

The closed-interface tests drive :func:`~neuoct.rendering_rt.snell_refract` from
explicit front-face state, then check the same physics end to end through the
integrator on a closed mesh and through a NeuOct-shaped thin-walled surface.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from neuoct.rendering_rt import (
    Camera,
    ConstantEnvironment,
    Integrator,
    Material,
    MeshPrimitive,
    Rays,
    Scene,
    dielectric_f0,
    orient_normal_to_ray,
    schlick_fresnel,
    snell_refract,
    thin_walled_refract,
)
from neuoct.rendering_rt.integrator import Environment
from neuoct.rendering_rt.rng import CounterRNG
from neuoct.rendering_rt.types import RayHit


def _unit(vector) -> torch.Tensor:
    tensor = torch.tensor([vector], dtype=torch.float64)
    return tensor / tensor.norm(dim=-1, keepdim=True)


def test_normal_incidence_refraction_is_straight() -> None:
    # A ray entering glass along the interface normal must not bend, and the
    # entering ratio must be 1 / ior.  ``front_face`` and the winding normal are
    # consistent: the ray arrives against the outward normal (+Z).
    result = snell_refract(_unit([0.0, 0.0, -1.0]), _unit([0.0, 0.0, 1.0]), torch.tensor([True]), ior=1.5)
    assert torch.allclose(result.direction, _unit([0.0, 0.0, -1.0]), atol=1e-9)
    assert float(result.eta[0]) == pytest.approx(1.0 / 1.5, rel=1e-9)
    assert float(result.cos_theta_i[0]) == pytest.approx(1.0, abs=1e-9)
    assert not bool(result.tir[0])
    # Normal-incidence Fresnel of an n=1.5 interface is ((n-1)/(n+1))^2.
    assert float(dielectric_f0(1.5)) == pytest.approx((0.5 / 2.5) ** 2, rel=1e-6)
    assert float(result.fresnel[0]) == pytest.approx(float(dielectric_f0(1.5)), rel=1e-6)


def test_oblique_refraction_obeys_snells_law() -> None:
    ior = 1.5
    incidence = math.radians(50.0)
    direction = _unit([math.sin(incidence), 0.0, -math.cos(incidence)])
    normal = _unit([0.0, 0.0, 1.0])
    result = snell_refract(direction, normal, torch.tensor([True]), ior=ior)
    sin_t = math.sin(incidence) / ior
    assert float(result.cos_theta_t[0]) == pytest.approx(math.sqrt(1.0 - sin_t * sin_t), rel=1e-6)
    # Snell: eta_i sin(theta_i) == eta_t sin(theta_t).
    assert math.sin(incidence) == pytest.approx(ior * sin_t, rel=1e-9)
    # And the refracted direction must satisfy the same relation geometrically.
    refracted = result.direction[0]
    transmitted_angle = math.asin(min(1.0, float(refracted[0]) ** 2 + float(refracted[1]) ** 2) ** 0.5)
    assert math.sin(transmitted_angle) == pytest.approx(sin_t, rel=1e-6)
    # Angle-dependent Schlick reflectance must exceed the normal-incidence value.
    assert float(result.fresnel[0]) > float(dielectric_f0(ior))


def test_exiting_ray_uses_the_inverted_index_ratio() -> None:
    # Travelling outwards against an outward-facing normal: front_face False.
    inside_direction = _unit([math.sin(0.3), 0.0, math.cos(0.3)])
    outward_normal = _unit([0.0, 0.0, 1.0])
    result = snell_refract(inside_direction, outward_normal, torch.tensor([False]), ior=1.5)
    assert float(result.eta[0]) == pytest.approx(1.5, rel=1e-9)
    assert not bool(result.tir[0])
    # The exit ray must bend *away* from the normal relative to the interior ray.
    assert abs(float(result.direction[0, 0])) > abs(float(inside_direction[0, 0]))


def test_total_internal_reflection_above_the_critical_angle() -> None:
    ior = 1.5
    critical = math.asin(1.0 / ior)
    inside = _unit([math.sin(critical + 0.05), 0.0, math.cos(critical + 0.05)])
    outward = _unit([0.0, 0.0, 1.0])
    result = snell_refract(inside, outward, torch.tensor([False]), ior=ior)
    assert bool(result.tir[0])
    # The direction is the mirror reflection about the interface normal.
    assert torch.allclose(result.direction, _unit([inside[0, 0], 0.0, -inside[0, 2]]), atol=1e-6)
    assert float(result.reflectance[0]) == pytest.approx(1.0, abs=1e-12)
    # Just below the critical angle refraction still happens.
    below = _unit([math.sin(critical - 0.05), 0.0, math.cos(critical - 0.05)])
    assert not bool(snell_refract(below, outward, torch.tensor([False]), ior=ior).tir[0])


def test_orient_normal_reports_entry_and_exit() -> None:
    normal = _unit([0.0, 0.0, -1.0])
    entering, front_in = orient_normal_to_ray(normal, _unit([0.0, 0.0, 1.0]))
    assert bool(front_in[0])
    assert torch.allclose(entering, normal, atol=1e-12)
    _, front_out = orient_normal_to_ray(normal, _unit([0.0, 0.0, -1.0]))
    assert not bool(front_out[0])


def test_thin_walled_transmission_keeps_the_incident_direction() -> None:
    direction = _unit([0.5, 0.0, -0.8660254037844386])
    transmitted, weight = thin_walled_refract(direction, _unit([0.0, 0.0, -1.0]), f0=0.04)
    assert transmitted.shape == direction.shape
    assert torch.allclose(transmitted, direction, atol=1e-12)
    expected = 1.0 - float(schlick_fresnel(torch.tensor(0.8660254037844386), 0.04))
    assert float(weight[0]) == pytest.approx(expected, rel=1e-6)


def test_neuoct_transmission_uses_normal_snell_and_ignores_learned_albedo() -> None:
    """NeuOct's learned normal bends a transparent path without diffuse tint."""
    material = Material(
        base_color=torch.ones(3),
        roughness=0.0,
        metallic=0.0,
        specular=0.04,
        transmission=1.0,
        ior=1.5,
        thin_walled=True,
    )
    hit = RayHit(
        hit=torch.tensor([True]),
        t=torch.tensor([1.0]),
        position=torch.zeros(1, 3),
        normal=torch.tensor([[0.0, 0.0, 1.0]]),
        barycentric=torch.tensor([[1.0, 0.0, 0.0]]),
        triangle_index=torch.tensor([-1]),
        primitive_id=torch.tensor([7]),
        # Deliberately dark learned diffuse colour: transmission must not use it.
        albedo=torch.full((1, 3), 0.01),
        albedo_srgb=torch.full((1, 3), 0.1),
        hit_probability=torch.ones(1),
        material=material,
    )
    integrator = Integrator(Scene([]), environment=ConstantEnvironment([1.0, 1.0, 1.0]))
    materials = integrator._material_batch(hit, 1, torch.device("cpu"))
    incident_angle = math.radians(45.0)
    incident = torch.tensor([[math.sin(incident_angle), 0.0, -math.cos(incident_angle)]])
    direction, weight, inside, stats = integrator._sample_branch(
        hit,
        materials,
        incident,
        bounce=0,
        pixel_index=torch.tensor([0]),
        sample_index=torch.tensor([0]),
        rng=CounterRNG(seed=3),
        inside=torch.tensor([False]),
    )
    expected_sin = math.sin(math.asin(math.sin(incident_angle) / 1.5))
    assert float(direction[0, 0]) == pytest.approx(expected_sin, rel=1e-5)
    assert float(direction[0, 2]) < 0.0
    assert float(weight[0, 0]) > 0.9
    assert bool(inside[0])
    assert stats["thin_walled"] == 1


def _slab_scene(*, thick: bool, base_color=None) -> Scene:
    """A closed glass slab spanning z in [0, 0.2] with real thickness."""
    import trimesh

    # Axis-aligned box built from 12 triangles, wound outwards.
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.2], [1.0, -1.0, 0.2], [1.0, 1.0, 0.2], [-1.0, 1.0, 0.2],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],   # front (-z)
            [4, 5, 6], [4, 6, 7],   # back (+z)
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    geometry = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material = Material(
        base_color=torch.ones(3) if base_color is None else torch.as_tensor(base_color),
        roughness=0.0,
        metallic=0.0,
        specular=0.04,
        transmission=1.0,
        ior=1.5,
        thin_walled=not thick,
    )
    # ``orient_normals_to_ray=False`` keeps the winding so front_face is real.
    return Scene(
        [
            MeshPrimitive(
                geometry, material=material, primitive_id=0, orient_normals_to_ray=False
            )
        ]
    )


def test_mesh_primitive_reports_front_face_and_geometric_normal() -> None:
    scene = _slab_scene(thick=True)
    # The slab spans z in [0, 0.2] and the ray starts at z = +1, so the nearest
    # interface is the +z face, whose outward winding normal is +Z.
    rays = Rays(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[0.0, 0.0, -1.0]]))
    hit = scene.intersect(rays)
    assert bool(hit.hit[0])
    assert float(hit.position[0, 2]) == pytest.approx(0.2, abs=1e-6)
    assert hit.front_face is not None and bool(hit.front_face[0])
    assert hit.geometric_normal is not None
    assert float(hit.geometric_normal[0, 2]) == pytest.approx(1.0, abs=1e-6)
    # From inside, the exit face at z = 0 is a back face (ray leaves along the
    # winding normal), which is exactly the state Snell needs for eta_t = 1.
    inside = Rays(torch.tensor([[0.0, 0.0, 0.1]]), torch.tensor([[0.0, 0.0, -1.0]]))
    back = scene.intersect(inside)
    assert bool(back.hit[0])
    assert not bool(back.front_face[0])
    assert float(back.geometric_normal[0, 2]) == pytest.approx(-1.0, abs=1e-6)


def test_closed_glass_refracts_while_thin_wall_does_not() -> None:
    # A ray at 45 degrees: the closed slab must displace the background sample,
    # the thin-walled sheet must not change its direction at all.
    direction = (math.sin(math.radians(45.0)), 0.0, -math.cos(math.radians(45.0)))
    thick = _slab_scene(thick=True)
    thin = _slab_scene(thick=False)
    rays = Rays(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([direction]),
        t_min=1.0e-6,
    )
    thick_hit = thick.intersect(rays)
    thin_hit = thin.intersect(rays)
    assert bool(thick_hit.hit[0]) and bool(thin_hit.hit[0])
    assert bool(thick_hit.front_face[0]) and bool(thin_hit.front_face[0])
    # Same geometry, so the first interface is identical; only the closing
    # interface differs.  Trace the exit explicitly from just inside the slab.
    exit_ray = Rays(
        thick_hit.position + torch.tensor([0.0, 0.0, -0.05]),
        torch.tensor([direction]),
        t_min=1.0e-6,
    )
    exit_hit = thick.intersect(exit_ray)
    assert float(exit_hit.position[0, 2]) == pytest.approx(0.0, abs=1e-6)
    assert not bool(exit_hit.front_face[0])
    snell = snell_refract(
        torch.tensor([direction], dtype=torch.float64),
        thick_hit.geometric_normal.double(),
        thick_hit.front_face,
        ior=1.5,
    )
    assert not bool(snell.tir[0])
    assert not torch.allclose(snell.direction, torch.tensor([direction], dtype=torch.float64), atol=1e-3)
    thin_direction, _ = thin_walled_refract(
        torch.tensor([direction], dtype=torch.float64),
        thin_hit.geometric_normal.double(),
    )
    assert torch.allclose(thin_direction, torch.tensor([direction], dtype=torch.float64), atol=1e-12)


def test_integrator_counts_transmission_models_and_stays_finite() -> None:
    scene = _slab_scene(thick=True)
    camera = Camera(
        width=4,
        height=4,
        fx=2.0,
        fy=2.0,
        cx=2.0,
        cy=2.0,
        camera_to_world=torch.eye(4),
    )
    # Camera at +z looking towards -z through the slab.
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor([0.0, 0.0, 1.0])
    camera = Camera(
        width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0, camera_to_world=matrix,
        camera_convention="legacy_blender",
    )
    integrator = Integrator(
        scene,
        environment=ConstantEnvironment([0.5, 0.5, 0.5]),
        mode="path",
        max_depth=3,
        samples=4,
        seed=3,
        self_hit_eps=1.0e-5,
    )
    buffers = integrator.render(camera)
    assert torch.isfinite(buffers.color).all()
    assert float(buffers.color.min()) >= 0.0
    assert buffers.closed_transmission_paths > 0
    assert buffers.thin_walled_transmission_paths == 0


def test_thin_walled_primitive_is_counted_separately() -> None:
    scene = _slab_scene(thick=False)
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor([0.0, 0.0, 1.0])
    camera = Camera(
        width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0, camera_to_world=matrix,
        camera_convention="legacy_blender",
    )
    integrator = Integrator(
        scene,
        environment=ConstantEnvironment([0.5, 0.5, 0.5]),
        mode="path",
        max_depth=3,
        samples=4,
        seed=3,
        self_hit_eps=1.0e-5,
    )
    buffers = integrator.render(camera)
    assert torch.isfinite(buffers.color).all()
    assert buffers.thin_walled_transmission_paths > 0
    assert buffers.closed_transmission_paths == 0


def test_terminal_transmission_does_not_restore_diffuse_albedo() -> None:
    # At max_depth=0 a hit is terminal. A dark base colour must not make a
    # fully transmissive surface dark: the terminal transparent contribution is
    # controlled by Fresnel only, not by diffuse albedo.
    scene = _slab_scene(thick=False, base_color=[0.01, 0.02, 0.03])
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor([0.0, 0.0, 1.0])
    camera = Camera(
        width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0,
        camera_to_world=matrix, camera_convention="legacy_blender",
    )
    buffers = Integrator(
        scene,
        environment=ConstantEnvironment([1.0, 1.0, 1.0]),
        mode="path",
        max_depth=0,
        samples=1,
        seed=3,
        self_hit_eps=1.0e-5,
    ).render(camera)
    hit = buffers.hit
    assert bool(hit.any())
    # The perspective rays include grazing angles, so Fresnel varies across the
    # image; it must nevertheless remain far above the dark diffuse albedo.
    assert float(buffers.color[hit].mean()) > 0.2


def test_terminal_transmission_samples_directional_environment() -> None:
    class DirectionalEnvironment(Environment):
        @property
        def ambient(self) -> torch.Tensor:
            # Deliberately different from every directional sample.  The old
            # terminal fallback returned this value and made transparent
            # objects collapse to a flat environment-average colour.
            return torch.zeros(3)

        def radiance(self, directions: torch.Tensor) -> torch.Tensor:
            return directions.abs()

    scene = _slab_scene(thick=False, base_color=[0.01, 0.02, 0.03])
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor([0.0, 0.0, 1.0])
    camera = Camera(
        width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0,
        camera_to_world=matrix, camera_convention="legacy_blender",
    )
    buffers = Integrator(
        scene,
        environment=DirectionalEnvironment(),
        mode="path",
        max_depth=0,
        samples=1,
        seed=3,
        self_hit_eps=1.0e-5,
    ).render(camera)
    hit = buffers.hit
    assert bool(hit.any())
    # Ambient is exactly zero, so any non-zero transparent contribution proves
    # that the terminal path queried radiance using its current ray direction.
    assert float(buffers.color[hit].mean()) > 0.1
    assert float(buffers.color[hit].std()) > 0.05


def test_render_buffers_expose_primitive_coverage() -> None:
    scene = _slab_scene(thick=True)
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor([0.0, 0.0, 1.0])
    camera = Camera(
        width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0, camera_to_world=matrix,
        camera_convention="legacy_blender",
    )
    buffers = Integrator(
        scene, environment=ConstantEnvironment([1.0, 1.0, 1.0]), max_depth=1, samples=1
    ).render(camera)
    assert buffers.primitive_id is not None
    assert buffers.coverage.get("0", 0) > 0
