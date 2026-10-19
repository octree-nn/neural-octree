"""Deterministic reflection / refraction / microfacet sampling helpers.

Conventions
-----------
``d`` is the unit *incident* direction (travelling from the eye into the
surface).  ``n`` is the unit surface normal.  Reflection and refraction return
unit directions pointing away from the surface.  All random inputs are explicit
uniforms in ``[0, 1)`` supplied by the caller (usually :class:`CounterRNG`), so
sampling is reproducible.

Transmission
------------
Two models are provided and clearly separated:

``refract``
    Snell's law at a single two-sided interface.  Total internal reflection is
    reported through the returned ``tir`` flag and the caller must fall back to
    the reflection direction.

``thin_walled_refract``
    The glTF-style zero-thickness sheet: the transmitted direction is unchanged
    because the entry and exit interfaces cancel, and only the Fresnel weight
    modulates transmission.  This is a thin sheet, *not* a closed solid medium.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

_EPS = 1.0e-8


def reflect(direction: Tensor, normal: Tensor) -> Tensor:
    """Mirror reflection of ``direction`` about ``normal``."""
    direction = torch.as_tensor(direction)
    normal = F.normalize(torch.as_tensor(normal), dim=-1, eps=_EPS)
    return F.normalize(direction - 2.0 * (direction * normal).sum(-1, keepdim=True) * normal, dim=-1, eps=_EPS)


def schlick_fresnel(cos_theta: Tensor, f0: Tensor | float) -> Tensor:
    """Schlick's approximation; ``cos_theta`` is the angle to the normal."""
    cos_theta = torch.as_tensor(cos_theta).clamp(0.0, 1.0)
    f0 = torch.as_tensor(f0)
    return f0 + (1.0 - f0) * torch.pow(1.0 - cos_theta, 5.0)


def orthonormal_basis(normal: Tensor) -> tuple[Tensor, Tensor]:
    """Deterministic, branch-free tangent frame around ``normal``.

    Uses the identity from Duff et al., "Building an Orthonormal Basis,
    Revisited", which is continuous over the whole sphere.
    """
    normal = F.normalize(torch.as_tensor(normal), dim=-1, eps=_EPS)
    sign = torch.where(normal[..., 2:3] >= 0.0, torch.ones_like(normal[..., 2:3]), -torch.ones_like(normal[..., 2:3]))
    a = -1.0 / (sign + normal[..., 2:3])
    b = normal[..., 0:1] * normal[..., 1:2] * a
    tangent = torch.cat(
        [
            1.0 + sign * normal[..., 0:1] * normal[..., 0:1] * a,
            sign * b,
            -sign * normal[..., 0:1],
        ],
        dim=-1,
    )
    bitangent = torch.cat(
        [
            b,
            sign + normal[..., 1:2] * normal[..., 1:2] * a,
            -normal[..., 1:2],
        ],
        dim=-1,
    )
    return F.normalize(tangent, dim=-1, eps=_EPS), F.normalize(bitangent, dim=-1, eps=_EPS)


def cosine_hemisphere_sample(normal: Tensor, u1: Tensor, u2: Tensor) -> Tensor:
    """Cosine-weighted upper-hemisphere sample around ``normal``."""
    normal = F.normalize(torch.as_tensor(normal), dim=-1, eps=_EPS)
    u1 = torch.as_tensor(u1).clamp(0.0, 1.0)
    u2 = torch.as_tensor(u2).clamp(0.0, 1.0)
    radius = torch.sqrt(u1)
    phi = 2.0 * torch.pi * u2
    tangent, bitangent = orthonormal_basis(normal)
    local = torch.stack([radius * torch.cos(phi), radius * torch.sin(phi), torch.sqrt((1.0 - u1).clamp_min(0.0))], dim=-1)
    sample = local[..., 0:1] * tangent + local[..., 1:2] * bitangent + local[..., 2:3] * normal
    return F.normalize(sample, dim=-1, eps=_EPS)


def ggx_half_vector(normal: Tensor, roughness: Tensor | float, u1: Tensor, u2: Tensor) -> Tensor:
    """Importance-sample a GGX half vector in the hemisphere around ``normal``."""
    normal = F.normalize(torch.as_tensor(normal), dim=-1, eps=_EPS)
    roughness = torch.as_tensor(roughness).clamp(1.0e-3, 1.0)
    u1 = torch.as_tensor(u1).clamp(0.0, 1.0)
    u2 = torch.as_tensor(u2).clamp(0.0, 1.0)
    alpha = roughness * roughness
    phi = 2.0 * torch.pi * u1
    cos_theta = torch.sqrt(((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2)).clamp(0.0, 1.0))
    sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp_min(0.0))
    cos_theta = cos_theta.unsqueeze(-1)
    sin_theta = sin_theta.unsqueeze(-1)
    tangent, bitangent = orthonormal_basis(normal)
    half_vector = (
        sin_theta * torch.cos(phi).unsqueeze(-1) * tangent
        + sin_theta * torch.sin(phi).unsqueeze(-1) * bitangent
        + cos_theta * normal
    )
    return F.normalize(half_vector, dim=-1, eps=_EPS)


def ggx_reflection(
    direction: Tensor,
    normal: Tensor,
    roughness: Tensor | float,
    u1: Tensor,
    u2: Tensor,
) -> Tensor:
    """GGX microfacet reflection direction."""
    half_vector = ggx_half_vector(normal, roughness, u1, u2)
    reflected = reflect(direction, half_vector)
    normal = F.normalize(torch.as_tensor(normal), dim=-1, eps=_EPS)
    same_side = (reflected * normal).sum(-1, keepdim=True) >= 0.0
    fallback = reflect(direction, normal)
    return torch.where(same_side, reflected, fallback)


def mirror_reflection(direction: Tensor, normal: Tensor) -> Tensor:
    """Perfect mirror reflection (roughness -> 0)."""
    return reflect(direction, normal)


@dataclass
class RefractionResult:
    direction: Tensor
    tir: Tensor
    cos_theta: Tensor


def refract(
    direction: Tensor,
    normal: Tensor,
    eta: Tensor | float,
    *,
    f0: Tensor | float = 0.04,
) -> RefractionResult:
    """Snell refraction at a two-sided interface.

    ``eta`` is ``n_incident / n_transmitted``.  On total internal reflection the
    returned direction is the mirror reflection and ``tir`` is ``True``.
    """
    direction = F.normalize(torch.as_tensor(direction), dim=-1, eps=_EPS)
    normal = F.normalize(torch.as_tensor(normal), dim=-1, eps=_EPS)
    # Orient the normal against the incoming direction.
    facing = (direction * normal).sum(-1, keepdim=True) < 0.0
    oriented = torch.where(facing, normal, -normal)
    cos_i = (-(direction * oriented).sum(-1, keepdim=True)).clamp(0.0, 1.0)
    eta = torch.as_tensor(eta)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    tir = sin2_t > 1.0
    cos_t = torch.sqrt((1.0 - sin2_t).clamp_min(0.0))
    refracted = eta * direction + (eta * cos_i - cos_t) * oriented
    refracted = F.normalize(refracted, dim=-1, eps=_EPS)
    reflected = reflect(direction, oriented)
    final = torch.where(tir, reflected, refracted)
    return RefractionResult(direction=final, tir=tir.squeeze(-1), cos_theta=cos_t.squeeze(-1))


def thin_walled_refract(
    direction: Tensor,
    normal: Tensor,
    *,
    f0: Tensor | float = 0.04,
    transmission: Tensor | float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Zero-thickness sheet transmission.

    Returns ``(transmitted_direction, weight)`` where the direction equals the
    incident one (the front and back interfaces cancel) and the scalar weight is
    ``transmission * (1 - fresnel)``.  This is explicitly a thin sheet, not a
    closed solid medium.
    """
    direction = F.normalize(torch.as_tensor(direction), dim=-1, eps=_EPS)
    normal = F.normalize(torch.as_tensor(normal), dim=-1, eps=_EPS)
    cos_i = (direction * normal).sum(-1).abs().clamp(0.0, 1.0)
    fresnel = schlick_fresnel(cos_i, f0)
    weight = torch.as_tensor(transmission) * (1.0 - fresnel)
    return direction, weight


def dielectric_f0(ior: Tensor | float) -> Tensor:
    """Normal-incidence Fresnel reflectance of a dielectric interface."""
    ior = torch.as_tensor(ior)
    ratio = (ior - 1.0) / (ior + 1.0)
    return (ratio * ratio).clamp(0.0, 1.0)


def orient_normal_to_ray(normal: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(oriented_normal, front_face)`` for a two-sided surface.

    ``front_face`` is ``True`` when the ray arrives against the geometric
    normal (it is entering the surface from the side the winding faces), which
    is what selects ``eta_i / eta_t`` for a closed dielectric.
    """
    normal = F.normalize(torch.as_tensor(normal), dim=-1, eps=_EPS)
    direction = F.normalize(torch.as_tensor(direction), dim=-1, eps=_EPS)
    front_face = (direction * normal).sum(-1, keepdim=True) < 0.0
    return torch.where(front_face, normal, -normal), front_face.squeeze(-1)


@dataclass
class SnellRefraction:
    """Result of a two-sided Snell refraction with explicit inside/outside state."""

    direction: Tensor
    tir: Tensor
    cos_theta_i: Tensor
    cos_theta_t: Tensor
    eta: Tensor
    fresnel: Tensor

    @property
    def reflectance(self) -> Tensor:
        """Fresnel reflectance actually realised (1.0 on total internal reflection)."""
        return torch.where(self.tir, torch.ones_like(self.fresnel), self.fresnel)


def snell_refract(
    direction: Tensor,
    geometric_normal: Tensor,
    front_face: Tensor,
    *,
    ior: Tensor | float,
    ior_outside: Tensor | float = 1.0,
) -> SnellRefraction:
    """Refract a ray at a closed dielectric interface using front-face state.

    ``front_face`` (from :func:`orient_normal_to_ray`) chooses the index ratio:
    entering uses ``eta = ior_outside / ior``, exiting uses ``ior / ior_outside``.
    Total internal reflection is reported through ``tir`` and the returned
    ``direction`` is then the mirror reflection about the true interface normal,
    so a caller can always offset its new origin along ``direction`` without a
    second self-hit epsilon.  ``fresnel`` is the angle-dependent Schlick
    reflectance evaluated at the interface's own dielectric ``F0`` -- not at the
    material's specular tint -- because it modulates energy across the boundary.
    """
    direction = F.normalize(torch.as_tensor(direction), dim=-1, eps=_EPS)
    geometric_normal = F.normalize(torch.as_tensor(geometric_normal), dim=-1, eps=_EPS)
    front = torch.as_tensor(front_face)
    if front.ndim == 1:
        front = front.unsqueeze(-1)
    if front.shape != direction.shape[:1] + (1,):
        raise ValueError(
            f"front_face must match the ray count; got {tuple(front.shape)} for "
            f"{tuple(direction.shape)} ray directions"
        )
    entering = front
    ior_tensor = torch.as_tensor(ior, dtype=direction.dtype, device=direction.device)
    outside = torch.as_tensor(ior_outside, dtype=direction.dtype, device=direction.device)
    # A per-ray ior arrives as (N,) and must become (N, 1) so it broadcasts
    # against the (N, 3) directions; a python scalar stays 0-dim.
    if ior_tensor.ndim == 1:
        ior_tensor = ior_tensor.unsqueeze(-1)
    if outside.ndim == 1:
        outside = outside.unsqueeze(-1)
    eta = torch.where(entering, outside / ior_tensor, ior_tensor / outside)

    # The interface normal must oppose the incoming ray on the side we came from.
    normal = torch.where(entering, geometric_normal, -geometric_normal)
    cos_i = (-(direction * normal).sum(-1, keepdim=True)).clamp(0.0, 1.0)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    tir = sin2_t > 1.0
    cos_t = torch.sqrt((1.0 - sin2_t).clamp_min(0.0))
    refracted = F.normalize(eta * direction + (eta * cos_i - cos_t) * normal, dim=-1, eps=_EPS)
    reflected = reflect(direction, normal)
    final = torch.where(tir, reflected, refracted)
    # Interface Fresnel at the realised ratio; Schlick's approximation is exact
    # at normal incidence and monotone to 1 at grazing incidence.
    f0 = dielectric_f0(torch.where(entering, ior_tensor / outside, outside / ior_tensor))
    fresnel = schlick_fresnel(cos_i, f0).clamp(0.0, 1.0)
    return SnellRefraction(
        direction=final,
        tir=tir.squeeze(-1),
        cos_theta_i=cos_i.squeeze(-1),
        cos_theta_t=cos_t.squeeze(-1),
        eta=eta.squeeze(-1),
        fresnel=fresnel.squeeze(-1),
    )
