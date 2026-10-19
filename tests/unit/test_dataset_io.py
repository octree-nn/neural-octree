import json

import torch

from neuoct.config.schema import DataConfig
from neuoct.data import dataset as dataset_module
from neuoct.data.dataset import ShapeNetMultiViewDataset


def test_dataset_loads_only_the_requested_view_and_caches_metadata(tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    pointcloud_root = tmp_path / "points"
    image_root.mkdir()
    pointcloud_root.mkdir()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("category/object/0\ncategory/object/1\n", encoding="utf-8")
    frames = [
        {
            "file_path": f"{frame_id}.png",
            "camera_angle_x": 0.7,
            "transform_matrix": torch.eye(4).tolist(),
            "depth": {"min": 0.1, "max": 2.0},
        }
        for frame_id in range(2)
    ]
    json_reads = []
    image_reads = []

    def fake_load_json(path):
        json_reads.append(path)
        return json.loads(json.dumps({"frames": frames}))

    def fake_load_image(path):
        image_reads.append(path.name)
        if path.name.endswith("_depth.png"):
            return torch.zeros((2, 2), dtype=torch.int32)
        return torch.zeros((2, 2, 3), dtype=torch.uint8)

    monkeypatch.setattr(dataset_module, "load_json", fake_load_json)
    monkeypatch.setattr(dataset_module, "load_image_tensor", fake_load_image)
    config = DataConfig(
        image_root=image_root,
        pointcloud_root=pointcloud_root,
        manifest=manifest,
        image_width=2,
        image_height=2,
        sample_cache_size=1,
    )
    dataset = ShapeNetMultiViewDataset(config)
    monkeypatch.setattr(
        dataset.point_cache,
        "load",
        lambda _: (torch.zeros((1, 3)), torch.zeros((1, 3))),
    )

    first = dataset[0]
    assert first.entry.frame_id == 0
    assert image_reads == ["0.png", "0_depth.png", "0_normal.png"]
    assert len(json_reads) == 1

    dataset[0]
    assert image_reads == [
        "0.png",
        "0_depth.png",
        "0_normal.png",
    ]

    second = dataset[1]
    assert second.entry.frame_id == 1
    assert image_reads == [
        "0.png",
        "0_depth.png",
        "0_normal.png",
        "1.png",
        "1_depth.png",
        "1_normal.png",
    ]
    assert len(json_reads) == 1
