"""Tests for the cross-session knowledge log store (core/knowledge_log.py)."""

from __future__ import annotations

import json

from core.knowledge_log import KnowledgeLogStore, _error_signature


def _issue(
    message="ZeroDivisionError: division by zero",
    file="agents/foo.py",
    line=42,
    kind="traceback",
):
    return {"actionable": True, "file": file, "line": line, "message": message, "kind": kind}


def test_record_error_issues_creates_and_persists_atomically(tmp_path):
    path = tmp_path / "knowledge_log.json"
    store = KnowledgeLogStore(path)

    touched = store.record_error_issues([_issue()])
    assert touched == 1
    assert path.is_file()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["errors"]) == 1
    entry = next(iter(data["errors"].values()))
    assert entry["occurrences"] == 1
    assert entry["file"] == "agents/foo.py"
    assert entry["resolved"] is False


def test_record_error_issues_dedupes_and_increments_occurrences(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")

    store.record_error_issues([_issue()])
    store.record_error_issues([_issue()])
    store.record_error_issues([_issue()])

    data = store._load()
    assert len(data["errors"]) == 1
    entry = next(iter(data["errors"].values()))
    assert entry["occurrences"] == 3


def test_record_error_issues_treats_distinct_messages_separately(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")

    store.record_error_issues([_issue(message="ZeroDivisionError: division by zero")])
    store.record_error_issues([_issue(message="ValueError: bad input")])

    data = store._load()
    assert len(data["errors"]) == 2


def test_normalized_signature_ignores_volatile_numbers(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")

    store.record_error_issues([_issue(message="KeyError: missing key 12345")])
    store.record_error_issues([_issue(message="KeyError: missing key 67890")])

    data = store._load()
    assert len(data["errors"]) == 1
    entry = next(iter(data["errors"].values()))
    assert entry["occurrences"] == 2


def test_mark_error_resolved_flags_existing_entry(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")
    issue = _issue()
    store.record_error_issues([issue])

    resolved = store.mark_error_resolved(issue)
    assert resolved is True

    data = store._load()
    entry = next(iter(data["errors"].values()))
    assert entry["resolved"] is True
    assert entry["resolved_at"] is not None


def test_mark_error_resolved_returns_false_for_unknown_issue(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")
    assert store.mark_error_resolved(_issue()) is False
    assert not (tmp_path / "knowledge_log.json").is_file()


def test_recurrence_after_resolution_clears_resolved_flag(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")
    issue = _issue()
    store.record_error_issues([issue])
    store.mark_error_resolved(issue)

    store.record_error_issues([issue])
    data = store._load()
    entry = next(iter(data["errors"].values()))
    assert entry["resolved"] is False
    assert entry["occurrences"] == 2


def test_error_cap_drops_resolved_entries_first(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json", max_error_signatures=2)

    resolved_issue = _issue(message="OldError: resolved long ago", file="a.py")
    store.record_error_issues([resolved_issue])
    store.mark_error_resolved(resolved_issue)

    store.record_error_issues([_issue(message="ActiveErrorOne", file="b.py")])
    store.record_error_issues([_issue(message="ActiveErrorTwo", file="c.py")])

    data = store._load()
    assert len(data["errors"]) == 2
    messages = {e["message"] for e in data["errors"].values()}
    assert "OldError: resolved long ago" not in messages


def test_add_fact_dedupes_exact_text(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")

    assert store.add_fact("The GPU is an RTX 3070", source="owner") is True
    assert store.add_fact("The GPU is an RTX 3070", source="owner") is False

    data = store._load()
    assert len(data["facts"]) == 1


def test_add_fact_respects_max_facts_cap(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json", max_facts=3)
    for i in range(5):
        store.add_fact(f"fact {i}", source="owner")

    data = store._load()
    assert len(data["facts"]) == 3
    assert data["facts"][-1]["text"] == "fact 4"


def test_add_fact_rejects_empty_text(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")
    assert store.add_fact("   ", source="owner") is False
    assert not (tmp_path / "knowledge_log.json").is_file()


def test_format_owner_summary_lists_unresolved_by_occurrence(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")
    frequent = _issue(message="FrequentError", file="a.py")
    rare = _issue(message="RareError", file="b.py")
    store.record_error_issues([frequent])
    store.record_error_issues([frequent])
    store.record_error_issues([rare])
    store.add_fact("Runs on an RTX 3070", source="owner")

    summary = store.format_owner_summary()
    assert "FrequentError" in summary
    assert "RareError" in summary
    assert summary.index("FrequentError") < summary.index("RareError")
    assert "Runs on an RTX 3070" in summary


def test_format_owner_summary_excludes_resolved_from_recurring_list(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")
    issue = _issue(message="FixedError")
    store.record_error_issues([issue])
    store.mark_error_resolved(issue)

    summary = store.format_owner_summary()
    assert "FixedError" not in summary
    assert "none tracked yet" in summary


def test_format_owner_summary_shows_shared_guest_interests(tmp_path):
    store = KnowledgeLogStore(tmp_path / "knowledge_log.json")
    summaries = [
        {"top_interests": [{"label": "Minecraft", "count": 5}]},
        {"top_interests": [{"label": "Minecraft", "count": 2}]},
        {"top_interests": [{"label": "Chess", "count": 1}]},
    ]
    summary = store.format_owner_summary(guest_profile_summaries=summaries)
    assert "Minecraft" in summary
    assert "2 guests" in summary
    assert "Chess" not in summary  # only one guest mentioned it


def test_error_signature_ignores_line_number_changes():
    a = _issue(line=10)
    b = _issue(line=99)
    assert _error_signature(a) == _error_signature(b)


def test_load_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "knowledge_log.json"
    path.write_text("not valid json{{{", encoding="utf-8")
    store = KnowledgeLogStore(path)
    data = store._load()
    assert data["errors"] == {}
    assert data["facts"] == []
