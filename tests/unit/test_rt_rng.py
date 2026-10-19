from __future__ import annotations

import torch

from neuoct.rendering_rt import CounterRNG, counter_seed_from


def test_uniform_range_and_dtype() -> None:
    values = CounterRNG(seed=1).uniform(torch.arange(4096))
    assert values.dtype == torch.float32
    assert values.shape == (4096,)
    assert bool((values >= 0.0).all()) and bool((values < 1.0).all())


def test_uniform_is_deterministic() -> None:
    counters = torch.arange(256)
    first = CounterRNG(seed=42).uniform(counters)
    second = CounterRNG(seed=42).uniform(counters)
    assert torch.equal(first, second)
    other_seed = CounterRNG(seed=43).uniform(counters)
    assert not torch.equal(first, other_seed)
    other_stream = CounterRNG(seed=42, stream=1).uniform(counters)
    assert not torch.equal(first, other_stream)


def test_uniform_is_independent_of_chunk_order() -> None:
    counters = torch.arange(1000)
    expected = CounterRNG(seed=7).uniform(counters)
    permutation = torch.randperm(1000, generator=torch.Generator().manual_seed(0))
    shuffled = CounterRNG(seed=7).uniform(counters[permutation])
    inverse = torch.argsort(permutation)
    assert torch.equal(shuffled[inverse], expected)


def test_uniform_index_addressing_is_progressive() -> None:
    rng = CounterRNG(seed=5)
    pixels = torch.tensor([0, 1, 2, 3])
    first_sample = rng.uniform_index(pixels, sample_index=0, dimension=3)
    # Adding a later sample must not shift the earlier one.
    again = rng.uniform_index(pixels, sample_index=0, dimension=3)
    assert torch.equal(first_sample, again)
    later = rng.uniform_index(pixels, sample_index=1, dimension=3)
    assert not torch.equal(first_sample, later)
    other_dimension = rng.uniform_index(pixels, sample_index=0, dimension=4)
    assert not torch.equal(first_sample, other_dimension)


def test_uniform_pair_is_independent() -> None:
    counters = torch.arange(128)
    first, second = CounterRNG(seed=9).uniform_pair(counters)
    assert not torch.equal(first, second)
    assert torch.equal(first, CounterRNG(seed=9).uniform(counters))


def test_uniform_rejects_negative_counters() -> None:
    try:
        CounterRNG(seed=0).uniform(torch.tensor([-1, 0]))
    except ValueError:
        pass
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("negative counters must raise")


def test_counter_seed_is_stable() -> None:
    assert counter_seed_from(1, 2, 3) == counter_seed_from(1, 2, 3)
    assert counter_seed_from(1, 2, 3) != counter_seed_from(3, 2, 1)
    assert 0 <= counter_seed_from(5) < 2**64
