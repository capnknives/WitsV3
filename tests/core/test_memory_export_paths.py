"""WP F2b: every memory export/import method runs the folder-access check first."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.filesystem_policy as fp
from core import runtime_paths
from core.memory_export import MemoryExporter, MemoryImporter


class FakeMemory:
    def __init__(self):
        self.added = []

    async def get_recent_memory(self, limit=10, filter_dict=None):
        return []

    async def get_memory(self, segment_id):
        return None

    async def add_segment(self, segment):
        self.added.append(segment)

    async def add_memory(self, **kwargs):
        self.added.append(kwargs)


@pytest.fixture
def env(tmp_path, monkeypatch):
    base, repo, outside = tmp_path / "base", tmp_path / "repo", tmp_path / "outside"
    for folder in (base / "inbox", base / "outbox", repo, outside):
        folder.mkdir(parents=True)
    monkeypatch.setattr(fp, "PROJECT_ROOT", repo)
    monkeypatch.setattr(runtime_paths, "PROJECT_ROOT", repo)
    config = SimpleNamespace(folders=SimpleNamespace(base=str(base)), security=None)
    monkeypatch.setattr("core.config.load_config", lambda *a, **k: config)
    return SimpleNamespace(base=base, outside=outside)


EXPORTS = ["export_to_json", "export_to_csv", "export_content_only"]
IMPORTS = {
    "import_from_json": ("m.json", '{"segments": []}'),
    "import_from_csv": ("m.csv", "id,type,source\n"),
    "import_text_as_segments": ("m.txt", "one --- two"),
}


@pytest.mark.parametrize("method", EXPORTS)
async def test_exporters_refuse_then_work_in_outbox(env, method):
    exporter = getattr(MemoryExporter(FakeMemory()), method)
    for bad in (env.outside / "m.out", env.base / "outbox" / ".env", env.base / "inbox" / "m"):
        with pytest.raises(PermissionError):
            await exporter(bad)
        assert not bad.exists()
    good = env.base / "outbox" / "m.out"
    assert await exporter(good) == 0
    assert good.is_file()


@pytest.mark.parametrize("method", list(IMPORTS))
async def test_importers_refuse_then_work_in_inbox(env, method):
    name, text = IMPORTS[method]
    memory = FakeMemory()
    importer = getattr(MemoryImporter(memory), method)
    for bad in (env.outside / name, env.base / "inbox" / "secrets.json"):
        bad.write_text(text)
        with pytest.raises(PermissionError):
            await importer(bad)
        assert bad.read_text() == text
    assert memory.added == []
    good = env.base / "inbox" / name
    good.write_text(text)
    expected = 2 if method == "import_text_as_segments" else 0
    assert await importer(good) == expected
