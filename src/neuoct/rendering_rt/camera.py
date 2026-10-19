"""Perspective and equidistant-fisheye primary ray generation.

Camera frame
------------
The camera-to-world matrix follows the historical NeuOct convention used by
``neuoct.data.camera`` and the on-disk ``transforms.json`` files:

* ``camera_to_world[:3, :3]`` stores the camera basis vectors as **rows**,
  expressed in scene space.
* The local optical axis is ``+Z`` (``+Z`` forward).
* A camera-space direction ``d`` maps to a scene direction ``d @ R`` (row
  vector times the 3x3 block), and the ray origin is ``camera_to_world[:3, 3]``.

With the ``legacy_blender`` convention the basis is stored in columns and
``-Z`` is forward; the projection math is shared but the direction is mapped
with ``d @ R.transpose()`` and the local forward is mirrored.

For perspective, the normalized image coordinate of pixel ``(x, y)`` is
``u = (x - cx) / fx``, ``v = (y - cy) / fy`` and the local direction is
``-(u, v, 1)``.  For an equidistant fisheye the incidence angle from the
optical axis is ``theta = sqrt(u**2 + v**2)`` (radians) and the local
direction is ``(-sin(theta) * u / rho, -sin(theta) * v / rho, cos(theta))``.
Pixels with ``theta`` beyond ``fisheye_fov / 2`` are masked out and their
directions are a safe, finite fallback (the optical axis).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi, tan

import torch
from torch import Tensor
from torch.nn import functional as F

from .types import Rays


def look_at_matrix(
    eye: Tensor | tuple[float, float, float],
    target: Tensor | tuple[float, float, float],
    up: Tensor | tuple[float, float, float] = (0.0, 1.0, 0.0),
    *,
    convention: str = "neuoct",
) -> Tensor:
    """Build a 4x4 camera-to-world matrix looking from ``eye`` at ``target``.

    The basis layout follows :class:`Camera`: rows and ``+Z`` forward for the
    ``neuoct`` convention, columns and ``-Z`` forward for ``legacy_blender``.
    """
    eye = torch.as_tensor(eye, dtype=torch.float32).reshape(3)
    target = torch.as_tensor(target, dtype=torch.float32).reshape(3)
    up = torch.as_tensor(up, dtype=torch.float32).reshape(3)
    look = F.normalize(target - eye, dim=0, eps=1.0e-8)
    up = F.normalize(up, dim=0, eps=1.0e-8)
    if convention not in ("neuoct", "legacy_blender"):
        raise ValueError("convention must be 'neuoct' or 'legacy_blender'")
    reference = look if convention == "neuoct" else -look
    right = torch.cross(up, reference, dim=0)
    if float(right.norm()) < 1.0e-5:
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        right = torch.cross(fallback, reference, dim=0)
    right = F.normalize(right, dim=0, eps=1.0e-8)
    if convention == "neuoct":
        true_up = F.normalize(torch.cross(look, right, dim=0), dim=0, eps=1.0e-8)
        rotation = torch.stack([right, true_up, look], dim=0)
    else:
        backward = -look
        true_up = F.normalize(torch.cross(backward, right, dim=0), dim=0, eps=1.0e-8)
        rotation = torch.stack([right, true_up, backward], dim=1)
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = eye
    return matrix


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: Tensor
    projection: str = "perspective"
    fisheye_fov: float = pi
    camera_convention: str = "neuoct"
    near: float = 1.0e-4
    far: float = float("inf")
    name: str = "camera"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("Camera focal lengths must be positive")
        if self.projection not in ("perspective", "equidistant"):
            raise ValueError("Camera.projection must be 'perspective' or 'equidistant'")
        if self.camera_convention not in ("neuoct", "legacy_blender"):
            raise ValueError(
                "Camera.camera_convention must be 'neuoct' or 'legacy_blender'"
            )
        if not 0.0 < float(self.fisheye_fov) <= 2.0 * pi:
            raise ValueError("Camera.fisheye_fov must be in (0, 2*pi]")
        camera_to_world = torch.as_tensor(self.camera_to_world, dtype=torch.float32)
        if camera_to_world.shape != (4, 4):
            raise ValueError(
                f"Camera.camera_to_world must have shape (4, 4), got {tuple(camera_to_world.shape)}"
            )
        object.__setattr__(self, "camera_to_world", camera_to_world)
        object.__setattr__(self, "near", float(self.near))
        object.__setattr__(self, "far", float(self.far))

    @classmethod
    def from_horizontal_fov(
        cls,
        camera_angle_x: float,
        width: int,
        height: int,
        camera_to_world: Tensor,
        **kwargs,
    ) -> "Camera":
        focal = 0.5 * width / tan(0.5 * float(camera_angle_x))
        return cls(
            width=width,
            height=height,
            fx=focal,
            fy=focal,
            cx=width / 2.0,
            cy=height / 2.0,
            camera_to_world=camera_to_world,
            **kwargs,
        )

    @classmethod
    def from_fisheye_fov(
        cls,
        fisheye_fov: float,
        width: int,
        height: int,
        camera_to_world: Tensor,
        **kwargs,
    ) -> "Camera":
        # The focal length controls how many pixels one radian spans; choosing
        # f = min(width, height) / fisheye_fov keeps the full disk inside the
        # shorter image dimension.
        focal = min(width, height) / float(fisheye_fov)
        kwargs.setdefault("projection", "equidistant")
        return cls(
            width=width,
            height=height,
            fx=focal,
            fy=focal,
            cx=width / 2.0,
            cy=height / 2.0,
            camera_to_world=camera_to_world,
            fisheye_fov=fisheye_fov,
            **kwargs,
        )

    @classmethod
    def from_vertical_fov(
        cls,
        yfov: float,
        width: int,
        height: int,
        camera_to_world: Tensor,
        **kwargs,
    ) -> "Camera":
        """N-BVH-compatible intrinsics: vertical FOV, square pixels.

        N-BVH builds its image plane as ``img_plane_size.y = 2*tan(fovy/2)`` and
        ``img_plane_size.x = img_plane_size.y * aspect``, which is exactly a
        single focal length in both axes.  Deriving ``fx = fy`` from the
        *vertical* field of view therefore reproduces its rays pixel for pixel
        at any aspect ratio, unlike :meth:`from_horizontal_fov`.
        """
        focal = 0.5 * height / tan(0.5 * float(yfov))
        return cls(
            width=width,
            height=height,
            fx=focal,
            fy=focal,
            cx=width / 2.0,
            cy=height / 2.0,
            camera_to_world=camera_to_world,
            **kwargs,
        )

    @classmethod
    def from_legacy_matrix(
        cls,
        matrix,
        *,
        width: int,
        height: int,
        yfov: float,
        projection: str = "perspective",
        fisheye_fov: float = pi,
        **kwargs,
    ) -> "Camera":
        """Adapter for the N-BVH legacy ``camera.matrix`` JSON field.

        The flat 16-number field is GLM **column-major** and stores the
        **world-to-camera view** matrix (N-BVH keeps its camera as
        ``to_world`` with ``m_inv_to_world = inverse(to_world)`` and derives
        ``dir() = normalize(inv * (0,0,-1,0))``, i.e. local ``-Z`` forward).  So
        the adapter reshapes with ``order='F'``, inverts, and uses the
        ``legacy_blender`` convention.  The raw translation column of the input
        is *not* the camera origin.
        """
        import numpy as np

        array = np.asarray(
            matrix.detach().cpu().numpy() if isinstance(matrix, Tensor) else matrix,
            dtype=np.float64,
        ).reshape(-1)
        if array.size != 16:
            raise ValueError(f"legacy camera matrix must have 16 entries, got {array.size}")
        view = array.reshape((4, 4), order="F")
        if not np.isfinite(view).all():
            raise ValueError("legacy camera matrix must be finite")
        try:
            camera_to_world = np.linalg.inv(view)
        except np.linalg.LinAlgError as error:
            raise ValueError("legacy camera matrix is singular and cannot be inverted") from error
        kwargs.setdefault("camera_convention", "legacy_blender")
        if projection == "equidistant":
            return cls.from_fisheye_fov(
                fisheye_fov,
                width,
                height,
                torch.as_tensor(camera_to_world, dtype=torch.float32),
                **kwargs,
            )
        return cls.from_vertical_fov(
            yfov,
            width,
            height,
            torch.as_tensor(camera_to_world, dtype=torch.float32),
            **kwargs,
        )

    @classmethod
    def look_at(
        cls,
        eye,
        target,
        up=(0.0, 1.0, 0.0),
        *,
        width: int,
        height: int,
        camera_angle_x: float = 0.6911112070083618,
        projection: str = "perspective",
        fisheye_fov: float = pi,
        convention: str = "neuoct",
        **kwargs,
    ) -> "Camera":
        """Construct a camera looking from ``eye`` at ``target``."""
        matrix = look_at_matrix(eye, target, up, convention=convention)
        if projection == "equidistant":
            return cls.from_fisheye_fov(
                fisheye_fov,
                width,
                height,
                matrix,
                camera_convention=convention,
                **kwargs,
            )
        return cls.from_horizontal_fov(
            camera_angle_x,
            width,
            height,
            matrix,
            camera_convention=convention,
            **kwargs,
        )

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def num_pixels(self) -> int:
        return self.height * self.width

    def _pixel_coordinates(self) -> tuple[Tensor, Tensor]:
        device = self.camera_to_world.device
        y, x = torch.meshgrid(
            torch.arange(self.height, device=device, dtype=torch.float32),
            torch.arange(self.width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        return x, y

    def pixel_mask(self) -> Tensor:
        """Boolean ``(H, W)`` mask; all ``True`` for a perspective camera."""
        if self.projection != "equidistant":
            return torch.ones(
                (self.height, self.width),
                dtype=torch.bool,
                device=self.camera_to_world.device,
            )
        x, y = self._pixel_coordinates()
        u = (x + 0.5 - self.cx) / self.fx
        v = (y + 0.5 - self.cy) / self.fy
        theta = torch.sqrt(u * u + v * v)
        return theta <= (0.5 * float(self.fisheye_fov) + 1.0e-9)

    def camera_directions(self) -> Tensor:
        """Local camera-space unit directions with shape ``(H, W, 3)``."""
        x, y = self._pixel_coordinates()
        u = (x + 0.5 - self.cx) / self.fx
        v = (y + 0.5 - self.cy) / self.fy
        forward = torch.ones_like(u)
        if self.camera_convention == "legacy_blender":
            planar = torch.stack([u, -v, -forward], dim=-1)
        else:
            planar = torch.stack([-u, -v, forward], dim=-1)
        if self.projection == "perspective":
            return F.normalize(planar, dim=-1, eps=1.0e-8)
        # Equidistant fisheye: theta = r (normalized image radius).
        theta = torch.sqrt(u * u + v * v)
        rho = theta.clamp_min(1.0e-8)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        if self.camera_convention == "neuoct":
            directions = torch.stack(
                [
                    -sin_theta * (u / rho),
                    -sin_theta * (v / rho),
                    cos_theta,
                ],
                dim=-1,
            )
        else:
            # legacy_blender: local -Z forward.  The image row axis ``v`` grows
            # downwards, so it maps to *-Y* exactly as in the perspective branch
            # above; flipping only the Z sign (as an earlier revision did) made
            # a fisheye image vertically mirrored relative to the same c2w's
            # perspective image.
            directions = torch.stack(
                [
                    sin_theta * (u / rho),
                    -sin_theta * (v / rho),
                    -cos_theta,
                ],
                dim=-1,
            )
        return F.normalize(directions, dim=-1, eps=1.0e-8)

    def camera_to_world_directions(self, local: Tensor) -> Tensor:
        rotation = self.camera_to_world[:3, :3]
        if self.camera_convention == "legacy_blender":
            world = local @ rotation.transpose(0, 1)
        else:
            world = local @ rotation
        return F.normalize(world, dim=-1, eps=1.0e-8)

    def world_to_camera_directions(self, world: Tensor) -> Tensor:
        rotation = self.camera_to_world[:3, :3]
        if self.camera_convention == "legacy_blender":
            local = world @ rotation
        else:
            local = world @ rotation.transpose(0, 1)
        return F.normalize(local, dim=-1, eps=1.0e-8)

    def generate_rays(self, *, device: torch.device | str | None = None) -> Rays:
        """Primary rays for every pixel in row-major order (``y * width + x``)."""
        camera = self if device is None else self.to(device)
        local = camera.camera_directions()
        world = camera.camera_to_world_directions(local).reshape(-1, 3)
        origin = camera.camera_to_world[:3, 3].reshape(1, 3).expand(world.shape[0], 3)
        mask = camera.pixel_mask().reshape(-1)
        if not bool(mask.all()):
            # Outside the fisheye disk the direction is replaced by the optical
            # axis so the ray is finite; callers must respect ``pixel_mask``.
            optical_axis = camera.camera_directions()[camera.height // 2, camera.width // 2]
            axis_world = camera.camera_to_world_directions(optical_axis).reshape(1, 3)
            world = torch.where(mask.unsqueeze(-1), world, axis_world.expand_as(world))
        return Rays(
            origins=origin,
            directions=world,
            t_min=camera.near,
            t_max=camera.far,
        )

    def primary_rays(self, *, device: torch.device | str | None = None) -> tuple[Rays, Tensor]:
        """Convenience wrapper returning ``(rays, mask)`` with ``mask`` shaped ``(H, W)``."""
        camera = self if device is None else self.to(device)
        return camera.generate_rays(), camera.pixel_mask()

    def to(self, device: torch.device | str) -> "Camera":
        return replace(self, camera_to_world=self.camera_to_world.to(device))
