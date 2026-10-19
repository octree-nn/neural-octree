import torch

from neuoct.geometry.vertex_index import compute_vertex_index
from neuoct.models.vertex_features import VertexFeatureAggregator


class FakeOctree:
    def xyzb(self, depth, nempty=True):
        del depth, nempty
        xyz = torch.tensor([0, 0], dtype=torch.long)
        batch = torch.tensor([0, 1], dtype=torch.long)
        return xyz, xyz, xyz, batch


def test_batch_id_prevents_cross_object_vertex_merging():
    octree = FakeOctree()
    index = compute_vertex_index(octree, 7, include_batch=True)
    assert index.num_vertices == 16
    features = torch.cat([torch.zeros(1, 16), torch.ones(1, 16)], dim=0)
    field = VertexFeatureAggregator.from_voxel_corner_features(octree, 7, features)
    gathered = field.gather(torch.tensor([0, 1]))
    torch.testing.assert_close(gathered[0], torch.zeros_like(gathered[0]))
    torch.testing.assert_close(gathered[1], torch.ones_like(gathered[1]))
