from neuoct.data.manifest import ManifestEntry
from neuoct.data.manifest_tools import split_objects


def test_object_split_has_no_view_leakage():
    objects = {
        "a/one": [ManifestEntry("a", "one", 0), ManifestEntry("a", "one", 1)],
        "a/two": [ManifestEntry("a", "two", 0)],
        "b/three": [ManifestEntry("b", "three", 0)],
        "b/four": [ManifestEntry("b", "four", 0)],
    }
    train, test = split_objects(objects, test_ratio=0.5, seed=3)
    train_objects = {entry.object_key for entry in train}
    test_objects = {entry.object_key for entry in test}
    assert train_objects.isdisjoint(test_objects)
