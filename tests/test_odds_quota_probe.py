from edge_desk import odds


def test_note_quota_switches_plan_without_spending():
    st = {"quota": 500, "remaining": 407, "used": 93, "spent": {"2026-11-12": 4}}
    st2 = odds.note_quota(st, "2026-11-12T18:00:00Z",
                          {"x-requests-remaining": "19907", "x-requests-used": "93"})
    assert odds.quota(st2) == 20000 and odds.budget(st2)["high"]
    assert st2["spent"] == st["spent"]           # nothing counted as spent
    assert st["quota"] == 500                    # input untouched


def test_note_quota_missing_headers_is_noop():
    st = {"quota": 500}
    assert odds.note_quota(st, "2026-11-12T18:00:00Z", {}) == st
    assert odds.note_quota(st, "2026-11-12T18:00:00Z", {"x-requests-remaining": "x", "x-requests-used": "1"}) == st


def test_run_uses_event_headers_before_planning(monkeypatch, tmp_path):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setattr(odds, "ODDS_DIR", tmp_path)
    monkeypatch.setattr(odds, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(odds, "LINES", tmp_path / "lines.csv")
    (tmp_path / "state.json").write_text('{"quota": 500, "remaining": 407, "used": 93}')
    seen = {}

    def fake_get(url, params):
        return [], {"x-requests-remaining": "19907", "x-requests-used": "93"}

    def fake_plan(state, now, *a, **k):
        seen["q"] = odds.quota(state)
        return []
    monkeypatch.setattr(odds, "_get", fake_get)
    monkeypatch.setattr(odds, "plan_pulls", fake_plan)
    monkeypatch.setattr(odds, "fair_lines", lambda *a, **k: __import__("pandas").DataFrame())
    odds.run(now="2026-11-12T18:00:00Z", confirm=())
    assert seen["q"] == 20000
