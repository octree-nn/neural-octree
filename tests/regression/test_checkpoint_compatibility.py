import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch
from ocnn.octree import Points

from neuoct.geometry.octree_adapter import NeuOctOctree, build_input_features
from neuoct.models import NeuralIntersectionModel

CHECKPOINT = os.environ.get("NEUOCT_LEGACY_CHECKPOINT", "checkpoints/legacy_q7.pth")
LEGACY_ROOT = os.environ.get("NEUOCT_LEGACY_ROOT", "legacy")
pytestmark = pytest.mark.skipif(not Path(CHECKPOINT).is_file(), reason="Optional legacy checkpoint unavailable")


def _legacy_model_class():
    if not (Path(LEGACY_ROOT) / "models.py").is_file():
        pytest.skip("Optional legacy code unavailable")
    sys.path.insert(0, LEGACY_ROOT)
    spec = importlib.util.spec_from_file_location(
        "neuoct_legacy_models", str(Path(LEGACY_ROOT) / "models.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NeuralIntersectionModel


@pytest.mark.regression
def test_q7_checkpoint_keys_load_strictly():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = NeuralIntersectionModel(9, 7)
    assert model.load_state_dict(state, strict=True).missing_keys == []
    assert len(model.state_dict()) == 313


@pytest.mark.gpu
@pytest.mark.regression
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_legacy_and_refactored_forward_match():
    torch.manual_seed(7)
    state = torch.load(CHECKPOINT, map_location="cuda", weights_only=True)
    legacy = _legacy_model_class()(9, 7).cuda().eval()
    refactored = NeuralIntersectionModel(9, 7).cuda().eval()
    legacy.load_state_dict(state, strict=True)
    refactored.load_state_dict(state, strict=True)
    positions = torch.rand(2048, 3, device="cuda") * 1.8 - 0.9
    colors = torch.rand_like(positions)
    octree = NeuOctOctree(depth=9, full_depth=4, device="cuda")
    octree.build_octree(Points(positions, features=colors))
    octree.construct_all_neigh()
    features = build_input_features(octree, "LPF")
    pair_count = min(128, int(octree.nnum_nempty[7]))
    indices = torch.arange(pair_count, device="cuda")
    pins = torch.rand(pair_count, 3, device="cuda")
    directions = torch.nn.functional.normalize(torch.randn(pair_count, 3, device="cuda"), dim=1)
    with torch.no_grad():
        legacy_output = legacy(features, octree, 9, pins, directions, indices)
        new_output = refactored(features, octree, 9, pins, directions, indices).final.as_tuple()
    for old_tensor, new_tensor in zip(legacy_output, new_output, strict=True):
        torch.testing.assert_close(old_tensor, new_tensor, rtol=1.0e-5, atol=1.0e-6)
