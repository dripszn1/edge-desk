"""#34: the session guard in conftest catches any write under data/."""
from tests.conftest import data_changes, data_fingerprint


def test_guard_sees_new_changed_and_removed_files(tmp_path):
    (tmp_path / "a.csv").write_text("x\n")
    (tmp_path / "b.csv").write_text("y\n")
    before = data_fingerprint(tmp_path)
    assert data_changes(before, data_fingerprint(tmp_path)) == []
    (tmp_path / "a.csv").write_text("x\nfixture row\n")
    (tmp_path / "b.csv").unlink()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.json").write_text("{}")
    assert data_changes(before, data_fingerprint(tmp_path)) == ["a.csv", "b.csv", "sub/c.json"]
