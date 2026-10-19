import pytest

from neuoct.data.manifest import ManifestEntry, read_manifest


def test_manifest_entry_parses_forward_and_windows_separators():
    assert ManifestEntry.parse("02958343/object/12") == ManifestEntry("02958343", "object", 12)
    assert ManifestEntry.parse(r"02958343\object\12") == ManifestEntry("02958343", "object", 12)


def test_manifest_reports_bad_entries():
    with pytest.raises(ValueError, match="category/object/frame_id"):
        ManifestEntry.parse("object/12")
    with pytest.raises(ValueError, match="integer"):
        ManifestEntry.parse("category/object/frame")


def test_read_manifest_skips_comments_and_honors_take(tmp_path):
    path = tmp_path / "manifest.txt"
    path.write_text("# comment\na/b/0\na/b/1\n", encoding="utf-8")
    assert [entry.frame_id for entry in read_manifest(path, take=1)] == [0]
