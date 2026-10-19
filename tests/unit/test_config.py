from types import SimpleNamespace

import pytest

from neuoct.config.schema import DataConfig, ModelConfig, feature_channels, validate_solver_config
from neuoct.engine.local_optimize import LocalOptimizationConfig


def _data_flags(tmp_path):
    return SimpleNamespace(
        location=str(tmp_path),
        pointcloud_root=str(tmp_path),
        filelist=str(tmp_path / "manifest.txt"),
        depth=9,
        full_depth=4,
        q_depth=7,
        image_width=32,
        image_height=32,
        point_scale=0.99,
        object_cache_size=2,
        take=-1,
    )


def test_feature_channel_validation():
    assert feature_channels("LPF") == 9
    assert feature_channels("NDP") == 7
    with pytest.raises(ValueError, match="Unsupported"):
        feature_channels("LX")


def test_legacy_takes_is_used_when_take_is_unset(tmp_path):
    flags = _data_flags(tmp_path)
    flags.takes = 12
    assert DataConfig.from_flags(flags).take == 12


def test_model_rejects_vae_and_channel_mismatch():
    with pytest.raises(NotImplementedError):
        ModelConfig(9, "LPF", 7, use_vae=True).validate()
    with pytest.raises(ValueError, match="does not match"):
        ModelConfig(6, "LPF", 7).validate()


def test_solver_config_accepts_valid_non_vae_settings(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("category/object/0\n", encoding="utf-8")
    flags = SimpleNamespace(
        MODEL=SimpleNamespace(channel=9, feature="LPF", q_depth=7, use_vae=False),
        DATA=SimpleNamespace(train=_data_flags(tmp_path), test=_data_flags(tmp_path)),
    )
    validate_solver_config(flags)


def test_data_config_validates_normal_source(tmp_path):
    config = DataConfig(
        image_root=tmp_path,
        pointcloud_root=tmp_path,
        manifest=tmp_path,
        normal_source="invalid",
    )
    with pytest.raises(ValueError, match="normal_source"):
        config.validate(require_paths=False)


def test_local_optimization_validates_sampling_mode(tmp_path):
    data = DataConfig(
        image_root=tmp_path,
        pointcloud_root=tmp_path,
        manifest=tmp_path / "manifest.txt",
    )
    invalid = LocalOptimizationConfig(
        data=data,
        object_key="category/object",
        output=tmp_path / "out",
        sampling_mode="adaptive",
    )
    with pytest.raises(ValueError, match="sampling_mode"):
        invalid.validate()
    for mode in (
        "recurrent",
        "uniform",
        "single",
        "recurrent_endpoints",
        "uniform_endpoints",
        "uniform_concat",
        "uniform_concat_endpoints",
        "recurrent_pool",
        "single_endpoints",
    ):
        LocalOptimizationConfig(
            data=data,
            object_key="category/object",
            output=tmp_path / "out",
            sampling_mode=mode,
        ).validate()


def test_local_optimization_validates_compile_mode(tmp_path):
    data = DataConfig(
        image_root=tmp_path,
        pointcloud_root=tmp_path,
        manifest=tmp_path / "manifest.txt",
    )
    invalid = LocalOptimizationConfig(
        data=data,
        object_key="category/object",
        output=tmp_path / "out",
        compile_mode="turbo",
    )
    with pytest.raises(ValueError, match="compile_mode"):
        invalid.validate()
    for mode in ("default", "max-autotune-no-cudagraphs"):
        LocalOptimizationConfig(
            data=data,
            object_key="category/object",
            output=tmp_path / "out",
            compile_mode=mode,
        ).validate()
    # CUDA-graph modes are rejected on purpose: the renderer is invoked once per
    # ray chunk, so a replayed graph output is overwritten before its backward.
    for mode in ("reduce-overhead", "max-autotune"):
        with pytest.raises(ValueError, match="CUDA graphs"):
            LocalOptimizationConfig(
                data=data,
                object_key="category/object",
                output=tmp_path / "out",
                compile_mode=mode,
            ).validate()
