import torch

from neuoct.engine.features import load_feature_export, save_feature_export
from neuoct.geometry.vertex_index import VertexIndex
from neuoct.models.vertex_features import VertexFeatureField


def test_feature_export_round_trip(tmp_path):
    index = VertexIndex(torch.tensor([[0, 0, 0, 0]]), torch.zeros((1, 8), dtype=torch.long))
    field = VertexFeatureField(index, torch.ones(1, 2))
    path = save_feature_export(
        tmp_path / "features.pt",
        entries=["category/object/0"],
        query_depth=7,
        voxel_coordinates=torch.tensor([[0, 0, 0, 0]]),
        voxel_features=torch.ones(1, 16),
        field=field,
    )
    payload = load_feature_export(path)
    assert payload["schema_version"] == 1
    assert payload["entries"] == ["category/object/0"]
