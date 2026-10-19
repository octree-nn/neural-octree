"""Linear lat-long HDRI environment, matching the N-BVH direction lookup.

The lookup implemented here is a transcription of the deployed N-BVH kernel
(``src/cuda/neural/path/path_inference_kernels.cu``, ``background_hit_eval``)::

    glm::mat4 rotation = glm::rotate(glm::mat4(1.f),
                                     glm::radians(envmap_rotation),
                                     glm::vec3(0.0, 1.0, 0.0));
    dir = glm::vec3(rotation * glm::vec4(dir, 1.0));
    float envmap_u = atan2(dir.x, -dir.z) * INV_PI * 0.5f + 0.5f;
    float envmap_v = safe_acos(dir.y) * INV_PI;
    float4 val = tex2D<float4>(envmap_texture, envmap_u, envmap_v);
    glm::vec3 envmap_val = glm::min(glm::vec3(100.f),
                                    glm::vec3(val.xyz) * envmap_strength);

Three details are reproduced exactly and are deliberately not "improved":

1. The **world** ray direction is rotated around ``+Y`` by the configured
   right-handed angle *before* the UV lookup (``glm::rotate`` uses the
   right-hand rule, so ``R = [[c,0,s],[0,1,0],[-s,0,c]]``).
2. ``acos(dir.y)`` measures ``v`` from ``+Y`` downwards, so ``v = 0`` is the
   first scanline of the EXR (N-BVH copies the image top row into texture row
   0 and samples it with ``v = 0``).
3. Sampled channels are multiplied by ``strength`` and then clamped to ``100``
   (N-BVH's firefly cap).  The cap is exposed as ``channel_cap``.

The CUDA texture is created with ``cudaFilterModeLinear`` and
``cudaAddressModeClamp`` on both axes, so sampling is bilinear with texel
centres at ``(i + 0.5) / size`` and edge clamping -- not wrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, radians, sin
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .integrator import Environment

# N-BVH clamps the sampled envmap to this value to kill sun fireflies.
DEFAULT_CHANNEL_CAP = 100.0
# Deterministic quadrature used for the cosine-weighted ambient term.
_AMBIENT_U_SAMPLES = 256
_AMBIENT_V_SAMPLES = 128


def rotation_y_matrix(rotation_deg: float) -> np.ndarray:
    """Right-handed ``glm::rotate(angle, +Y)`` as a 3x3 row-major matrix."""
    angle = radians(float(rotation_deg))
    c, s = cos(angle), sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotate_world_directions(directions: Tensor | np.ndarray, rotation_deg: float) -> Tensor:
    """Actively rotate world directions around ``+Y`` (N-BVH order of operations)."""
    tensor = directions if isinstance(directions, Tensor) else torch.as_tensor(directions)
    matrix = torch.as_tensor(rotation_y_matrix(rotation_deg), dtype=tensor.dtype, device=tensor.device)
    return tensor @ matrix.transpose(0, 1)


def direction_to_uv(directions: Tensor | np.ndarray, rotation_deg: float = 0.0) -> Tensor:
    """Exact N-BVH world-direction -> ``(u, v)`` lat-long mapping.

    ``u`` is ``atan2(x, -z) / (2*pi) + 0.5`` and ``v`` is ``acos(y) / pi`` after
    the active ``+Y`` rotation, so ``u = v = 0.5`` looks along ``-Z`` and ``v``
    runs from the ``+Y`` pole (0) to the ``-Y`` pole (1).
    """
    rotated = rotate_world_directions(directions, rotation_deg)
    x = rotated[..., 0]
    y = rotated[..., 1].clamp(-1.0, 1.0)
    z = rotated[..., 2]
    u = torch.atan2(x, -z) / (2.0 * pi) + 0.5
    v = torch.acos(y) / pi
    return torch.stack([u, v], dim=-1)


def uv_to_direction(uv: Tensor | np.ndarray) -> Tensor:
    """Inverse of :func:`direction_to_uv` (used for analytic sampling tests)."""
    tensor = uv if isinstance(uv, Tensor) else torch.as_tensor(uv)
    u = tensor[..., 0]
    v = tensor[..., 1]
    phi = (u - 0.5) * 2.0 * pi
    theta = v * pi
    sin_theta = torch.sin(theta)
    return torch.stack(
        [sin_theta * torch.sin(phi), torch.cos(theta), -sin_theta * torch.cos(phi)], dim=-1
    )


def load_exr_rgb(path: str | Path) -> np.ndarray:
    """Load a linear EXR into a ``(H, W, 3)`` float32 array.

    Row 0 is the EXR's first scanline, i.e. the top of the lat-long image, which
    is exactly the row N-BVH's ``cudaMemcpy2DToArray`` puts at texture row 0.
    Both the OpenEXR 3.x object API and the legacy ``InputFile``/``Imath`` API
    are supported; a missing dependency raises an actionable error.
    """
    exr_path = Path(path)
    if not exr_path.exists():
        raise FileNotFoundError(f"HDRI not found: {exr_path}")
    try:
        import OpenEXR  # type: ignore
    except Exception as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "HDRI environments require the optional 'OpenEXR' package. Install it "
            "with `pip install OpenEXR`, or build the environment from a synthetic "
            "array with LatLongHDRI.from_array(...) in tests."
        ) from error

    channels = _load_exr_channels(OpenEXR, exr_path)
    rgb = _assemble_rgb(channels, exr_path)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"{exr_path} did not load as an (H, W, 3) image, got {rgb.shape}")
    if not np.isfinite(rgb).all():
        raise ValueError(f"{exr_path} contains non-finite samples")
    return rgb


def _as_interleaved(array: np.ndarray) -> np.ndarray | None:
    """Return a ``(H, W, 3+)`` view of an interleaved channel, else ``None``."""
    if array.ndim != 3:
        return None
    if array.shape[-1] in (3, 4):
        return array[..., :3]
    if array.shape[0] in (3, 4):
        return np.moveaxis(array[:3], 0, -1)
    return None


def _assemble_rgb(channels: dict[str, np.ndarray], source: object) -> np.ndarray:
    """Assemble RGB from either separate ``R``/``G``/``B`` or an interleaved ``RGB``.

    Lat-long EXRs are not written uniformly: some carry three planar channels
    and others a single interleaved ``RGB`` channel.  Both are valid and both
    must load, so the layout is detected from the data instead of assumed.
    """
    planar = [channels.get(name) for name in ("R", "G", "B")]
    if all(array is not None and array.ndim == 2 for array in planar):
        return np.stack(planar, axis=-1)
    for name in ("RGB", "RGBA", "rgb", "rgba"):
        array = channels.get(name)
        if array is not None:
            interleaved = _as_interleaved(array)
            if interleaved is not None:
                return interleaved
    for name, array in channels.items():
        interleaved = _as_interleaved(array)
        if interleaved is not None:
            return interleaved
    raise ValueError(
        f"{source} does not expose an RGB image; found channels "
        f"{ {name: tuple(array.shape) for name, array in channels.items()} }"
    )


def _load_exr_channels(OpenEXR, exr_path: Path) -> dict[str, np.ndarray]:
    """Return ``{channel_name: pixels}`` using whichever API is present."""
    # --- OpenEXR >= 3.0 object API -------------------------------------------------
    try:
        handle = OpenEXR.File(str(exr_path))
        raw = handle.channels()
        if isinstance(raw, dict) and raw:
            out: dict[str, np.ndarray] = {}
            for name, channel in raw.items():
                pixels = getattr(channel, "pixels", None)
                if pixels is None:
                    continue
                out[str(name)] = np.asarray(pixels, dtype=np.float32)
            if out:
                return out
    except Exception:
        pass
    # --- legacy InputFile + Imath --------------------------------------------------
    try:
        import Imath  # type: ignore
    except Exception as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"OpenEXR was importable but neither its 3.x API nor Imath could read "
            f"{exr_path}: {error}"
        ) from error
    handle = OpenEXR.InputFile(str(exr_path))
    header = handle.header()
    window = header["dataWindow"]
    width = int(window.max.x - window.min.x + 1)
    height = int(window.max.y - window.min.y + 1)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    available = list(header.get("channels", {}))
    out = {}
    for name in ("R", "G", "B"):
        if name not in available:
            continue
        out[name] = np.frombuffer(handle.channel(name, pixel_type), dtype=np.float32).reshape(
            height, width
        )
    return out


def _bilinear_sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Clamped bilinear sample of ``(H, W, 3)`` at normalized ``uv`` in ``[0, 1]``."""
    height, width = image.shape[:2]
    x = uv[:, 0] * width - 0.5
    y = uv[:, 1] * height - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = (x - x0)[:, None]
    ty = (y - y0)[:, None]
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x0 + 1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y0 + 1, 0, height - 1)
    c00 = image[y0c, x0c]
    c10 = image[y0c, x1c]
    c01 = image[y1c, x0c]
    c11 = image[y1c, x1c]
    return c00 * (1 - tx) * (1 - ty) + c10 * tx * (1 - ty) + c01 * (1 - tx) * ty + c11 * tx * ty


def ambient_irradiance(image: np.ndarray, *, channel_cap: float) -> np.ndarray:
    """Cosine-weighted ``+Y`` hemisphere average of the lat-long map.

    The integral is evaluated in the environment's own frame with a fixed
    ``256 x 128`` quadrature, so it is deterministic and independent of the
    render.  A uniform rotation about ``+Y`` leaves the ``+Y`` hemisphere
    integral unchanged, so no rotation is applied here.  This is the
    irradiance term used for the terminal diffuse bounce; it is an
    approximation of a full multiple-importance-sampled sky, and is declared as
    such in the render manifest.
    """
    height, width = image.shape[:2]
    u = (np.arange(_AMBIENT_U_SAMPLES, dtype=np.float64) + 0.5) / _AMBIENT_U_SAMPLES
    v = (np.arange(_AMBIENT_V_SAMPLES, dtype=np.float64) + 0.5) / _AMBIENT_V_SAMPLES
    uu, vv = np.meshgrid(u, v, indexing="xy")
    uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)
    # theta is measured from +Y (v = theta / pi), matching direction_to_uv.
    theta = uv[:, 1] * pi
    weight = np.cos(theta) * np.sin(theta)
    positive = weight > 0.0
    sampled = _bilinear_sample(image, uv[positive])
    weights = weight[positive]
    total = float(weights.sum())
    if total <= 0.0:  # pragma: no cover - defensive
        return np.zeros(3, dtype=np.float32)
    # E / pi with a cosine-weighted estimator: sum(L * cos * sin) / sum(cos * sin),
    # scaled by the extra factor of pi that the Lambertian BRDF divides out.
    irradiance = (sampled * weights[:, None]).sum(axis=0) / total
    return np.minimum(irradiance, float(channel_cap)).astype(np.float32)


@dataclass
class LatLongHDRI(Environment):
    """A linear lat-long HDRI with the exact N-BVH direction lookup."""

    image: np.ndarray
    rotation_deg: float = 0.0
    strength: float = 1.0
    channel_cap: float = DEFAULT_CHANNEL_CAP
    path: str | None = None
    _ambient: Tensor = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"LatLongHDRI image must be (H, W, 3), got {image.shape}")
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError("LatLongHDRI image must be non-empty")
        if not np.isfinite(image).all():
            raise ValueError("LatLongHDRI image contains non-finite samples")
        if float(self.channel_cap) <= 0.0:
            raise ValueError("LatLongHDRI.channel_cap must be positive")
        self.image = image
        self.rotation_deg = float(self.rotation_deg)
        self.strength = float(self.strength)
        self.channel_cap = float(self.channel_cap)
        ambient = ambient_irradiance(image, channel_cap=self.channel_cap) * self.strength
        self._ambient = torch.as_tensor(np.minimum(ambient, self.channel_cap), dtype=torch.float32)

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_exr(
        cls,
        path: str | Path,
        *,
        rotation_deg: float = 0.0,
        strength: float = 1.0,
        channel_cap: float = DEFAULT_CHANNEL_CAP,
    ) -> "LatLongHDRI":
        return cls(
            image=load_exr_rgb(path),
            rotation_deg=rotation_deg,
            strength=strength,
            channel_cap=channel_cap,
            path=str(path),
        )

    @classmethod
    def from_array(
        cls,
        image: np.ndarray,
        *,
        rotation_deg: float = 0.0,
        strength: float = 1.0,
        channel_cap: float = DEFAULT_CHANNEL_CAP,
    ) -> "LatLongHDRI":
        """Build from an in-memory image; used by tests and analytic probes."""
        return cls(
            image=np.asarray(image, dtype=np.float32),
            rotation_deg=rotation_deg,
            strength=strength,
            channel_cap=channel_cap,
        )

    # -- Environment interface ------------------------------------------------
    @property
    def ambient(self) -> Tensor:
        return self._ambient

    @property
    def resolution(self) -> tuple[int, int]:
        return (int(self.image.shape[0]), int(self.image.shape[1]))

    def sample_uv(self, uv: Tensor) -> Tensor:
        """Bilinearly sample the (clamped) map at normalized ``uv``, capped."""
        flat = uv.reshape(-1, 2).detach().cpu().numpy().astype(np.float64)
        sampled = _bilinear_sample(self.image.astype(np.float64), flat)
        sampled = sampled * self.strength
        sampled = np.minimum(sampled, float(self.channel_cap))
        return torch.as_tensor(sampled, dtype=torch.float32, device=uv.device).reshape(
            (*uv.shape[:-1], 3)
        )

    def radiance(self, directions: Tensor) -> Tensor:
        directions = torch.as_tensor(directions)
        uv = direction_to_uv(directions, self.rotation_deg)
        return self.sample_uv(uv)

    def to(self, device: torch.device | str) -> "LatLongHDRI":
        self._ambient = self._ambient.to(device)
        return self

    def as_dict(self) -> dict[str, object]:
        """Manifest-friendly description of the environment."""
        return {
            "type": "hdri",
            "path": self.path,
            "rotation_deg": self.rotation_deg,
            "strength": self.strength,
            "channel_cap": self.channel_cap,
            "resolution": list(self.resolution),
            "lookup": "rotate_world_dir_+Y_then_u=atan2(x,-z)/2pi+0.5,v=acos(y)/pi",
            "filter": "bilinear",
            "address_mode": "clamp",
            "ambient": [float(value) for value in self._ambient],
        }


def environment_from_config(section: dict, *, device: torch.device | str = "cpu") -> Environment:
    """Build the configured :class:`Environment` (constant or HDRI)."""
    from .integrator import ConstantEnvironment

    kind = str(section.get("type", "constant"))
    if kind == "constant":
        return ConstantEnvironment(section.get("color", [0.0, 0.0, 0.0])).to(device)
    if kind != "hdri":
        raise ValueError(f"Unsupported environment type {kind!r}; use 'constant' or 'hdri'")
    path = section.get("path")
    if not path:
        raise ValueError("environment.type 'hdri' requires environment.path")
    environment = LatLongHDRI.from_exr(
        path,
        rotation_deg=float(section.get("rotation_deg", 0.0)),
        strength=float(section.get("strength", 1.0)),
        channel_cap=float(section.get("channel_cap", DEFAULT_CHANNEL_CAP)),
    )
    return environment.to(device)
