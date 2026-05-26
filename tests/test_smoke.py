from mrags.config import ensure_parent_dir


def test_ensure_parent_dir(tmp_path):
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    path = tmp_path / "nested" / "file.txt"
    ensure_parent_dir(str(path))
    assert path.parent.exists()
