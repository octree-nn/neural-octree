from pathlib import Path
import os

import pytest
import torch

from neuoct.config.schema import DataConfig
from neuoct.data.dataset import ShapeNetMultiViewDataset


@pytest.mark.real_data
def test_local_real_sample_loads_on_cpu():
    config = DataConfig(
        image_root=Path(os.environ.get("NEUOCT_TEST_IMAGES", "data/shapenet/images")),
        pointcloud_root=Path(os.environ.get("NEUOCT_TEST_POINTS", "data/shapenet/pointcloud")),
        manifest=Path(os.environ.get("NEUOCT_TEST_MANIFEST", "data/shapenet/test.txt")),
        image_width=512,
        image_height=512,
        take=1,
    )
    if not all(p.exists() for p in (config.image_root, config.pointcloud_root, config.manifest)):
        pytest.skip("Optional real dataset is not configured")
    sample = ShapeNetMultiViewDataset(config)[0]
    assert sample.point_positions.device.type == "cpu"
    assert sample.depth_image.shape == (512, 512)
    assert sample.color_image.shape == sample.normal_image.shape == (512, 512, 3)
    assert sample.valid_depth.dtype == torch.bool
