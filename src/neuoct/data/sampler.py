from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler


class ObjectDistributedInfSampler(Sampler[int]):
    """Infinite distributed sampler that keeps all views of an object on one rank."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        if not hasattr(dataset, "entries"):
            raise TypeError("ObjectDistributedInfSampler requires dataset.entries")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("Invalid distributed sampler rank/replica count")
        groups: dict[str, list[int]] = defaultdict(list)
        for index, entry in enumerate(dataset.entries):
            groups[entry.object_key].append(index)
        self._groups = list(groups.values())
        if not self._groups:
            raise ValueError("Dataset contains no object groups")
        self.epoch = 0
        self.indices: list[int] = []
        self.iter_num = 0
        self.set_epoch(0)

    def _build_indices(self, epoch: int) -> list[int]:
        groups = list(self._groups)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(epoch))
            order = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[index] for index in order]
        groups_per_rank = (len(groups) + self.num_replicas - 1) // self.num_replicas
        total_groups = groups_per_rank * self.num_replicas
        if total_groups > len(groups):
            groups.extend(groups[index % len(groups)] for index in range(total_groups - len(groups)))
        local_groups = groups[self.rank * groups_per_rank : (self.rank + 1) * groups_per_rank]
        return [index for group in local_groups for index in group]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.indices = self._build_indices(self.epoch)
        self.iter_num = 0

    def seek(self, sample_offset: int) -> None:
        """Position the sampler at a sample offset within the current epoch."""
        sample_offset = int(sample_offset)
        if not 0 <= sample_offset <= len(self.indices):
            raise ValueError(
                f"sample_offset must be in [0, {len(self.indices)}], got {sample_offset}"
            )
        self.iter_num = sample_offset

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if not self.indices:
            self.set_epoch(self.epoch)
        if self.iter_num >= len(self.indices):
            self.set_epoch(self.epoch + 1)
        index = self.indices[self.iter_num]
        self.iter_num += 1
        if self.iter_num >= len(self.indices):
            self.set_epoch(self.epoch + 1)
        return index

    def __len__(self) -> int:
        return len(self.indices)
