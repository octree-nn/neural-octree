"""Stateless, counter-based pseudo-random numbers.

A sample's value depends only on ``(seed, stream, counter)`` -- never on how a
render is chunked or in which order rays are processed.  That makes an offline
render bit-reproducible regardless of ``ray_chunk_size`` or ``candidate_chunk_size``,
which is the whole point of using a hash instead of a sequential generator.

The hash is ``splitmix64`` evaluated in ``numpy.uint64`` (where modular
arithmetic wraps by definition).  The top 53 bits become a double in ``[0, 1)``
and are down-cast to the requested floating dtype.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_PHI = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_TWO_POW_53 = 9007199254740992.0


def _splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        values = (values + _PHI) & _MASK64
        values = ((values ^ (values >> np.uint64(30))) * _MIX1) & _MASK64
        values = ((values ^ (values >> np.uint64(27))) * _MIX2) & _MASK64
        values = values ^ (values >> np.uint64(31))
    return values


def _counter_array(counter, *, device: torch.device | None = None):
    tensor = torch.as_tensor(counter, dtype=torch.int64)
    if tensor.numel() == 0:
        return tensor, np.empty(0, dtype=np.uint64), tensor.device
    if bool((tensor < 0).any()):
        raise ValueError("CounterRNG counters must be non-negative integers")
    flat = tensor.reshape(-1).detach().cpu().numpy().astype(np.uint64)
    return tensor, flat, tensor.device


def _to_float(values: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    unit = (values >> np.uint64(11)).astype(np.float64) / _TWO_POW_53
    if dtype == torch.float64:
        return unit
    return unit.astype(np.float32)


@dataclass(frozen=True)
class CounterRNG:
    """Deterministic hash-based random source.

    ``stream`` isolates independent random dimensions (e.g. one for bounce
    direction azimuth, one for elevation, one for roughness lobe selection).
    Incrementing ``stream`` or ``seed`` produces a completely different
    sequence while keeping the same counter addressing.
    """

    seed: int = 0
    stream: int = 0

    def _keys(self, counter) -> tuple[torch.Tensor, np.ndarray, torch.device, np.uint64]:
        tensor, flat, device = _counter_array(counter)
        seed_component = np.uint64(
            (int(self.seed) * 0x9E3779B97F4A7C15 + int(self.stream)) & 0xFFFFFFFFFFFFFFFF
        )
        return tensor, flat, device, seed_component

    def uniform(self, counter, *, dtype: torch.dtype = torch.float32) -> Tensor:
        """Uniform samples in ``[0, 1)`` for each counter value."""
        tensor, flat, device, seed_component = self._keys(counter)
        if flat.size == 0:
            return torch.empty(tuple(tensor.shape), dtype=dtype, device=device)
        mixed = _splitmix64(flat + seed_component)
        values = _to_float(mixed, dtype)
        return torch.from_numpy(values).to(device).reshape(tensor.shape)

    def uniform_pair(self, counter, *, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
        """Two independent uniforms from the same counter (stream and stream+1)."""
        first = self.uniform(counter, dtype=dtype)
        second = replace(self, stream=self.stream + 1).uniform(counter, dtype=dtype)
        return first, second

    def unify(
        self, index: Tensor, sample_index, dimension: int, *, offset: int = 0
    ) -> Tensor:
        """Address ``index``-th ray, ``sample_index``-th sample, ``dimension``-th value.

        This is the canonical addressing used by the integrator: the integer
        key mixes the pixel/ray id, the sample index and the random dimension so
        that increasing samples never shifts earlier samples (progressive).
        The result is masked to 63 bits so it is always a valid non-negative
        ``int64`` counter.
        """
        if dimension < 0 or offset < 0:
            raise ValueError("dimension and offset must be non-negative")
        index_tensor = torch.as_tensor(index, dtype=torch.int64)
        sample_tensor = torch.as_tensor(sample_index, dtype=torch.int64)
        if sample_tensor.numel() > 0 and bool((sample_tensor < 0).any()):
            raise ValueError("sample_index must be non-negative")
        index_np = index_tensor.detach().cpu().numpy().astype(np.uint64)
        sample_np = sample_tensor.detach().cpu().numpy().astype(np.uint64)
        constant = np.uint64((int(dimension + offset) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        with np.errstate(over="ignore"):
            key = index_np * np.uint64(6364136223846793005)
            key = key + sample_np * np.uint64(1442695040888963407)
            key = key + constant
        key = (key & np.uint64(0x7FFFFFFFFFFFFFFF)).astype(np.int64)
        return torch.from_numpy(key).to(index_tensor.device)

    def uniform_index(
        self, index: Tensor, sample_index: int, dimension: int, *, offset: int = 0
    ) -> Tensor:
        return self.uniform(self.unify(index, sample_index, dimension, offset=offset))

    def uniform_index_pair(
        self, index: Tensor, sample_index: int, dimension: int, *, offset: int = 0
    ) -> tuple[Tensor, Tensor]:
        key = self.unify(index, sample_index, dimension, offset=offset)
        first = self.uniform(key)
        second = replace(self, stream=self.stream + 1).uniform(key)
        return first, second


def counter_seed_from(*parts: int) -> int:
    """Combine several small integers into a stable 64-bit seed."""
    value = 0
    for part in parts:
        value = (value * 1000003 + int(part) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    return value
