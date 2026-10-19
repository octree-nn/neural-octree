"""Scene composition: closest-hit selection across heterogeneous primitives.

A scene holds any mix of mesh and NeuOct primitives.  ``intersect`` traces all
of them and keeps, per ray, the candidate with the smallest ``t``.  Exact ``t``
ties are broken deterministically by ``(primitive_id, triangle_index)`` and
finally by the order in which primitives were added, so the result never depends
on iteration order or floating-point associations.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .materials import MaterialResolver
from .primitives import Primitive
from .types import MaterialBatch, RayHit, Rays, stable_lexsort

_DEFAULT_MATERIAL = MaterialResolver().resolve_neuoct()


class Scene:
    def __init__(self, primitives: list[Primitive] | None = None, *, name: str = "scene") -> None:
        self.name = name
        self.primitives: list[Primitive] = []
        for primitive in primitives or []:
            self.add(primitive)

    def add(self, primitive: Primitive) -> Primitive:
        if not hasattr(primitive, "intersect"):
            raise TypeError("Scene primitives must implement intersect(Rays) -> RayHit")
        self.primitives.append(primitive)
        return primitive

    def __len__(self) -> int:
        return len(self.primitives)

    def __iter__(self):
        return iter(self.primitives)

    @property
    def primitive_ids(self) -> list[int]:
        return [int(primitive.primitive_id) for primitive in self.primitives]

    def intersect(self, rays: Rays) -> RayHit:
        if not self.primitives:
            return RayHit.miss(rays.num_rays, device=rays.device)
        if len(self.primitives) == 1:
            return self.primitives[0].intersect(rays)
        count = rays.num_rays
        device = rays.device
        flat_ray: list[Tensor] = []
        flat_t: list[Tensor] = []
        flat_position: list[Tensor] = []
        flat_normal: list[Tensor] = []
        flat_bary: list[Tensor] = []
        flat_triangle: list[Tensor] = []
        flat_primitive: list[Tensor] = []
        flat_albedo: list[Tensor] = []
        flat_albedo_srgb: list[Tensor] = []
        flat_prob: list[Tensor] = []
        flat_uv: list[Tensor | None] = []
        flat_geometric: list[Tensor | None] = []
        flat_front_face: list[Tensor | None] = []
        flat_material: list[MaterialBatch] = []
        has_uv = False
        has_geometric = False
        has_front_face = False
        for primitive in self.primitives:
            hit = primitive.intersect(rays)
            index = torch.nonzero(hit.hit, as_tuple=False).squeeze(1)
            if index.numel() == 0:
                continue
            flat_ray.append(index)
            flat_t.append(hit.t[index])
            flat_position.append(hit.position[index])
            flat_normal.append(hit.normal[index])
            flat_bary.append(hit.barycentric[index])
            flat_triangle.append(hit.triangle_index[index])
            flat_primitive.append(hit.primitive_id[index])
            flat_albedo.append(hit.albedo[index])
            flat_albedo_srgb.append(hit.albedo_srgb[index])
            flat_prob.append(hit.hit_probability[index])
            if hit.materials is not None:
                material = hit.materials.index(index)
            elif hit.material is not None:
                material = MaterialBatch.from_material(hit.material, count, device=device).index(index)
            else:
                material = MaterialBatch.from_material(_DEFAULT_MATERIAL, count, device=device).index(index)
            flat_material.append(material)
            if hit.uv is not None:
                has_uv = True
                flat_uv.append(hit.uv[index])
            else:
                flat_uv.append(None)
            if hit.geometric_normal is not None:
                has_geometric = True
                flat_geometric.append(hit.geometric_normal[index])
            else:
                flat_geometric.append(None)
            if hit.front_face is not None:
                has_front_face = True
                flat_front_face.append(hit.front_face[index])
            else:
                flat_front_face.append(None)
        if not flat_ray:
            return RayHit.miss(count, device=device)
        ray_index = torch.cat(flat_ray)
        t = torch.cat(flat_t)
        position = torch.cat(flat_position)
        normal = torch.cat(flat_normal)
        bary = torch.cat(flat_bary)
        triangle = torch.cat(flat_triangle)
        primitive_id = torch.cat(flat_primitive)
        albedo = torch.cat(flat_albedo)
        albedo_srgb = torch.cat(flat_albedo_srgb)
        prob = torch.cat(flat_prob)
        base_color = torch.cat([m.base_color for m in flat_material])
        roughness = torch.cat([m.roughness for m in flat_material])
        metallic = torch.cat([m.metallic for m in flat_material])
        specular = torch.cat([m.specular for m in flat_material])
        transmission = torch.cat([m.transmission for m in flat_material])
        ior = torch.cat([m.ior for m in flat_material])
        thin_walled = torch.cat([m.thin_walled for m in flat_material])
        emission = torch.cat([m.emission for m in flat_material])
        srgb_flags = torch.cat(
            [
                m.base_color_srgb
                if m.base_color_srgb is not None
                else torch.zeros(m.num_rays, dtype=torch.bool, device=device)
                for m in flat_material
            ]
        )
        uv = None
        if has_uv:
            uv = torch.zeros(ray_index.numel(), 2, device=device)
            # Advance by the number of flattened *hit rows* contributed by each
            # primitive, not by the UV length: a primitive with hits but no UV
            # contributes rows that must still consume an offset slot, otherwise
            # every later primitive's UVs are written at the wrong rows.
            offset = 0
            for hit_uv, hit_rows in zip(flat_uv, flat_ray):
                length = int(hit_rows.numel())
                if hit_uv is not None:
                    uv[offset : offset + length] = hit_uv
                offset += length
        geometric = None
        front_face = None
        if has_geometric:
            geometric = torch.zeros(ray_index.numel(), 3, device=device)
            offset = 0
            for hit_normal, hit_rows in zip(flat_geometric, flat_ray):
                length = int(hit_rows.numel())
                if hit_normal is not None:
                    geometric[offset : offset + length] = hit_normal
                offset += length
        if has_front_face:
            front_face = torch.zeros(ray_index.numel(), dtype=torch.bool, device=device)
            offset = 0
            for hit_flag, hit_rows in zip(flat_front_face, flat_ray):
                length = int(hit_rows.numel())
                if hit_flag is not None:
                    front_face[offset : offset + length] = hit_flag
                offset += length

        # Primary key ray index, then t, then primitive id, then triangle index.
        arity = torch.arange(ray_index.numel(), device=device)
        order = stable_lexsort(ray_index, t, primitive_id, triangle, arity)
        sorted_rays = ray_index[order]
        first = torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=device),
                sorted_rays[1:] != sorted_rays[:-1],
            ]
        )
        selected = order[first]
        selected_rays = ray_index[selected]

        miss = RayHit.miss(count, device=device)
        hit_tensor = miss.hit.clone()
        t_tensor = miss.t.clone()
        position_tensor = miss.position.clone()
        normal_tensor = miss.normal.clone()
        bary_tensor = miss.barycentric.clone()
        triangle_tensor = miss.triangle_index.clone()
        primitive_tensor = miss.primitive_id.clone()
        albedo_tensor = miss.albedo.clone()
        albedo_srgb_tensor = miss.albedo_srgb.clone()
        prob_tensor = miss.hit_probability.clone()
        hit_tensor[selected_rays] = True
        t_tensor[selected_rays] = t[selected]
        position_tensor[selected_rays] = position[selected]
        normal_tensor[selected_rays] = normal[selected]
        bary_tensor[selected_rays] = bary[selected]
        triangle_tensor[selected_rays] = triangle[selected]
        primitive_tensor[selected_rays] = primitive_id[selected]
        albedo_tensor[selected_rays] = albedo[selected]
        albedo_srgb_tensor[selected_rays] = albedo_srgb[selected]
        prob_tensor[selected_rays] = prob[selected]
        uv_tensor = None
        if uv is not None:
            uv_tensor = torch.zeros(count, 2, device=device)
            uv_tensor[selected_rays] = uv[selected]
        geometric_tensor = None
        if geometric is not None:
            geometric_tensor = torch.zeros(count, 3, device=device)
            geometric_tensor[selected_rays] = geometric[selected]
        front_tensor = None
        if front_face is not None:
            front_tensor = torch.zeros(count, dtype=torch.bool, device=device)
            front_tensor[selected_rays] = front_face[selected]
        # ``RayHit.materials`` is indexed with rows of the *full* ray batch, so it
        # must carry one row per ray -- not one row per winning hit.  A combined
        # scene is exactly where the two differ (the chess mesh plus the NeuOct
        # cannon: some rays miss everything), and a per-hit-sized batch would
        # make ``RayHit.index`` fail with a mask/length mismatch.  Misses keep
        # neutral values that are never read because ``hit`` is False there.
        def _scatter(values: Tensor, fill=0.0) -> Tensor:
            shape = (count,) + tuple(values.shape[1:])
            if values.dtype == torch.bool:
                out = torch.full(shape, bool(fill), dtype=torch.bool, device=device)
            else:
                out = torch.full(shape, float(fill), dtype=values.dtype, device=device)
            out[selected_rays] = values[selected]
            return out

        materials = MaterialBatch(
            base_color=_scatter(base_color),
            roughness=_scatter(roughness, 1.0),
            metallic=_scatter(metallic),
            specular=_scatter(specular),
            transmission=_scatter(transmission),
            ior=_scatter(ior, 1.0),
            thin_walled=_scatter(thin_walled, True),
            emission=_scatter(emission),
            base_color_srgb=_scatter(srgb_flags, False),
        )
        return RayHit(
            hit=hit_tensor,
            t=t_tensor,
            position=position_tensor,
            normal=normal_tensor,
            barycentric=bary_tensor,
            triangle_index=triangle_tensor,
            primitive_id=primitive_tensor,
            albedo=albedo_tensor,
            albedo_srgb=albedo_srgb_tensor,
            hit_probability=prob_tensor,
            uv=uv_tensor,
            material=None,
            materials=materials,
            geometric_normal=geometric_tensor,
            front_face=front_tensor,
        )
