from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def generate_orbit_views(
    num_views: int = 60,
    *,
    radius: float = 2.5,
    elevation_radians: float = 0.1 * torch.pi,
    target: Tensor | None = None,
    camera_convention: str = "neuoct",
    up_axis: str = "y",
    device: torch.device | str = "cuda",
) -> Tensor:
    """Construct camera-to-world poses for a circular orbit around ``target``.

    The project supports two on-disk camera conventions.  ``neuoct`` stores
    the camera basis as rows and uses ``+Z`` as the local forward direction;
    this is the convention used by the original ShapeNet data.  Blender
    transforms are standard camera-to-world matrices with basis vectors in
    columns and ``-Z`` forward.  The two representations are intentionally
    built separately here so an orbit pose has the same semantics as a pose
    loaded from ``transforms.json``.

    ``torch.linspace(..., 2*pi, num_views)`` would duplicate the first pose at
    the end of the video.  Using an index-based parameterization gives exactly
    one complete revolution with no repeated frame.
    """
    if num_views <= 0:
        raise ValueError(f"num_views must be positive, got {num_views}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if camera_convention not in ("neuoct", "legacy_blender"):
        raise ValueError(
            "camera_convention must be 'neuoct' or 'legacy_blender', got "
            f"{camera_convention!r}"
        )
    if up_axis not in ("x", "y", "z"):
        raise ValueError(f"up_axis must be 'x', 'y' or 'z', got {up_axis!r}")

    device = torch.device(device)
    # Keep the pose construction in float32 even when the model/checkpoint is
    # stored in half precision; ray generation can safely cast the result.
    dtype = torch.float32
    if target is None:
        target_tensor = torch.zeros(3, device=device, dtype=dtype)
    else:
        target_tensor = torch.as_tensor(target, device=device, dtype=dtype)
        if target_tensor.shape != (3,):
            raise ValueError(f"target must have shape (3,), got {tuple(target_tensor.shape)}")
    phi = torch.arange(num_views, device=device, dtype=dtype) * (2 * torch.pi / num_views)
    theta = torch.full_like(phi, float(elevation_radians))
    radial_cos = radius * torch.cos(theta) * torch.cos(phi)
    radial_sin = radius * torch.cos(theta) * torch.sin(phi)
    elevation = radius * torch.sin(theta)
    if up_axis == "x":
        offsets = torch.column_stack([elevation, radial_cos, radial_sin])
        world_up = [1.0, 0.0, 0.0]
    elif up_axis == "y":
        offsets = torch.column_stack([radial_cos, elevation, radial_sin])
        world_up = [0.0, 1.0, 0.0]
    else:
        offsets = torch.column_stack([radial_cos, radial_sin, elevation])
        world_up = [0.0, 0.0, 1.0]
    positions = offsets + target_tensor.unsqueeze(0)
    look = F.normalize(target_tensor.unsqueeze(0) - positions, dim=1, eps=1.0e-5)
    up = torch.tensor(world_up, device=device, dtype=dtype).expand_as(look)
    # Avoid a singular frame when the orbit is close to the world-up axis.
    reference_axis = look if camera_convention == "neuoct" else -look
    x_axis = F.normalize(torch.cross(up, reference_axis, dim=1), dim=1, eps=1.0e-5)
    degenerate = x_axis.abs().amax(dim=1, keepdim=True) < 5.0e-3
    if bool(degenerate.any()):
        fallback_up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(look)
        fallback = F.normalize(torch.cross(fallback_up, reference_axis, dim=1), dim=1, eps=1.0e-5)
        x_axis = torch.where(degenerate, fallback, x_axis)

    if camera_convention == "legacy_blender":
        # Blender uses local -Z as forward.  Its camera-to-world matrix has
        # right/up/backward as columns, where backward points from target to
        # the camera (the opposite of ``look``).
        backward = -look
        right = x_axis
        camera_up = F.normalize(torch.cross(backward, right, dim=1), dim=1, eps=1.0e-5)
        rotation = torch.stack([right, camera_up, backward], dim=2)
    else:
        # NeuOct's historical representation stores the basis as rows and
        # uses local +Z as forward.
        x_axis = F.normalize(torch.cross(up, look, dim=1), dim=1, eps=1.0e-5)
        if bool(degenerate.any()):
            fallback_up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(look)
            fallback = F.normalize(torch.cross(fallback_up, look, dim=1), dim=1, eps=1.0e-5)
            x_axis = torch.where(degenerate, fallback, x_axis)
        y_axis = F.normalize(torch.cross(look, x_axis, dim=1), dim=1, eps=1.0e-5)
        rotation = torch.stack([x_axis, y_axis, look], dim=1)

    transforms = torch.eye(4, device=device).repeat(num_views, 1, 1)
    transforms[:, :3, :3] = rotation
    transforms[:, :3, 3] = positions
    return transforms
