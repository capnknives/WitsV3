"""Tests for tiered core memory store."""

from core.core_memory import CoreMemoryStore


def test_promote_fact_dedup(tmp_path):
    store = CoreMemoryStore(path=tmp_path / "core.json", max_tokens=512)
    assert store.promote_fact("My dog is named Rex")
    assert not store.promote_fact("My dog is named Rex")
    assert len(store.to_dict()["user_facts"]) == 1


def test_prompt_block_respects_cap(tmp_path):
    store = CoreMemoryStore(path=tmp_path / "core.json", max_tokens=10)
    for i in range(20):
        store.promote_fact(f"Fact number {i} with some extra words")
    block = store.as_prompt_block()
    assert len(block) <= 10 * 4 + 1


def test_last_task_summary(tmp_path):
    store = CoreMemoryStore(path=tmp_path / "core.json")
    store.set_last_task_summary("Explained square roots")
    assert "square roots" in store.as_prompt_block()
