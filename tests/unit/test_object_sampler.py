from dataclasses import dataclass

from neuoct.data.sampler import ObjectDistributedInfSampler


@dataclass(frozen=True)
class _Entry:
    object_key: str


class _Dataset:
    def __init__(self) -> None:
        self.entries = [
            _Entry("cat/a"),
            _Entry("cat/a"),
            _Entry("cat/b"),
            _Entry("cat/b"),
            _Entry("cat/c"),
            _Entry("cat/c"),
        ]

    def __len__(self) -> int:
        return len(self.entries)


def test_seek_resumes_at_exact_sample_offset() -> None:
    dataset = _Dataset()
    sampler = ObjectDistributedInfSampler(
        dataset, num_replicas=1, rank=0, shuffle=True, seed=7
    )
    sampler.set_epoch(3)
    expected = [next(sampler) for _ in range(len(dataset))]

    sampler.set_epoch(3)
    sampler.seek(4)

    assert [next(sampler) for _ in range(2)] == expected[4:]


def test_seek_rejects_offsets_outside_epoch() -> None:
    sampler = ObjectDistributedInfSampler(
        _Dataset(), num_replicas=1, rank=0, shuffle=False
    )

    try:
        sampler.seek(7)
    except ValueError as error:
        assert "sample_offset" in str(error)
    else:
        raise AssertionError("seek should reject offsets beyond the epoch")
