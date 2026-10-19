import cv2
import numpy as np
import pytest

from neuoct.engine.local_orbit import _video_codec_spec


def test_video_codec_specs():
    assert _video_codec_spec("mp4v") == ("mp4v", ".mp4")
    assert _video_codec_spec("FFV1") == ("FFV1", ".mkv")
    with pytest.raises(ValueError, match="video_codec"):
        _video_codec_spec("unknown")


def test_ffv1_roundtrip_is_pixel_exact(tmp_path):
    path = tmp_path / "lossless.mkv"
    frame = np.arange(16 * 12 * 3, dtype=np.uint8).reshape(12, 16, 3)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"FFV1"),
        5,
        (16, 12),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build does not provide FFV1")
    writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    ok, decoded = capture.read()
    capture.release()
    assert ok
    np.testing.assert_array_equal(decoded, frame)
