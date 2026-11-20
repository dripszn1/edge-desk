"""Every workflow file must be valid YAML with no duplicate keys: a duplicate key makes GitHub reject the whole
workflow (11-13: collect.yml stopped running for ~1h after a step edit left `continue-on-error` twice)."""
from pathlib import Path

import yaml

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"


class _NoDupLoader(yaml.SafeLoader):
    pass


def _no_dups(loader, node, deep=False):
    seen = set()
    for k, _ in node.value:
        key = loader.construct_object(k, deep=deep)
        assert key not in seen, f"duplicate key {key!r} at line {k.start_mark.line + 1}"
        seen.add(key)
    return loader.construct_mapping(node, deep)


_NoDupLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dups)


def test_workflows_are_valid_yaml_without_duplicate_keys():
    files = sorted(WF.glob("*.yml"))
    assert files
    for f in files:
        doc = yaml.load(f.read_text(), Loader=_NoDupLoader)
        assert isinstance(doc, dict) and doc.get("jobs"), f.name


def test_automerge_installs_merged_requirements_before_tests():
    """11-13: dev added pyyaml; automerge installed main's requirements, then tested dev's tree -> ImportError, no merge."""
    step = (WF / "automerge.yml").read_text()
    body = step.split("git merge --no-ff", 1)[1]
    assert "pip install" in body.split("python -m pytest", 1)[0]


def test_collect_checks_out_branch_tip_not_trigger_sha():
    """11-20 (#73): collect 208 was queued behind 207 (concurrency group). Default checkout = the trigger-time sha, so
    it read the 07:51Z snapshot as 'latest', ran a full collect on top of 207's push and failed on data conflicts.
    Checking out the branch tip lets the freshness guard see the snapshot the run before it just pushed."""
    doc = yaml.load((WF / "collect.yml").read_text(), Loader=_NoDupLoader)
    steps = doc["jobs"]["run"]["steps"]
    co = steps[0]
    assert co["uses"].startswith("actions/checkout")
    assert "github.ref" in str((co.get("with") or {}).get("ref", "")), "collect must check out the branch tip"
    assert steps[1].get("id") == "fresh", "freshness guard must run right after checkout"
