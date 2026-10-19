import sys
from types import SimpleNamespace

import numpy as np
import torch

from neuoct.data.io import load_exr_tensor


def test_load_exr_tensor_reads_float_rgb_without_quantization(monkeypatch, tmp_path):
    expected = np.linspace(-1.0, 1.0, 18, dtype=np.float32).reshape(2, 3, 3)

    class FakeFile:
        def __init__(self, path):
            assert path == str(tmp_path / "normal.exr")

        def channels(self):
            return {"RGB": SimpleNamespace(pixels=expected)}

    monkeypatch.setitem(sys.modules, "OpenEXR", SimpleNamespace(File=FakeFile))
    actual = load_exr_tensor(tmp_path / "normal.exr")
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.from_numpy(expected))
