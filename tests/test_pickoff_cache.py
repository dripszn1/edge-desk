"""#54: the hourly pickoff cache must not serve a summary made by different code (a live run
checked out before a merge kept serving the pre-#52 proposal to collect for up to an hour)."""
import json

import pandas as pd

from edge_desk import pickoff


def _setup(monkeypatch, tmp_path, made, code):
    st = tmp_path / "pickoff_summary.json"
    st.write_text(json.dumps({"made_utc": made, "code": code,
                              "summary": {"proposals": ["OLD"]}}))
    monkeypatch.setattr(pickoff, "STATE", st)
    calls = []
    monkeypatch.setattr(pickoff, "run", lambda now=None: calls.append(now) or {"proposals": [], "in_effect": []})
    return st, calls


def test_same_code_recent_cache_served(monkeypatch, tmp_path):
    _, calls = _setup(monkeypatch, tmp_path, "2026-11-13T05:19:26Z", pickoff.code_hash())
    s = pickoff.run_if_due(pd.Timestamp("2026-11-13T05:32:00Z"))
    assert s["proposals"] == ["OLD"] and not calls


def test_other_code_cache_rerun(monkeypatch, tmp_path):
    st, calls = _setup(monkeypatch, tmp_path, "2026-11-13T05:19:26Z", "deadbeef")
    s = pickoff.run_if_due(pd.Timestamp("2026-11-13T05:32:00Z"))
    assert calls and s["proposals"] == []
    assert json.loads(st.read_text())["code"] == pickoff.code_hash()


def test_cache_without_code_stamp_rerun(monkeypatch, tmp_path):
    st, calls = _setup(monkeypatch, tmp_path, "2026-11-13T05:19:26Z", None)
    d = json.loads(st.read_text()); d.pop("code"); st.write_text(json.dumps(d))
    pickoff.run_if_due(pd.Timestamp("2026-11-13T05:32:00Z"))
    assert calls


def test_future_cache_rerun(monkeypatch, tmp_path):
    _, calls = _setup(monkeypatch, tmp_path, "2026-11-13T06:00:00Z", pickoff.code_hash())
    pickoff.run_if_due(pd.Timestamp("2026-11-13T05:32:00Z"))
    assert calls
