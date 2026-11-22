"""Tests for the ntfy phone push on new approvals (WP 1.6). No real network calls."""

import asyncio
import logging

import httpx
import pytest

from core.config import WorkspacePushSettings
from core.workspace import approvals, notify
from core.workspace.db import open_db

TOPIC = "s3cret-topic-xyz"
URL = "https://ntfy.example"


class _Cfg:
    def __init__(self, provider):
        class _WS:
            push = WorkspacePushSettings(provider=provider, ntfy_url=URL)

        self.workspace = _WS()


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(approvals, "_listeners", [])
    monkeypatch.setattr(notify, "_registered_listener", None)
    monkeypatch.setenv(notify.TOPIC_ENV, TOPIC)


def _setup(monkeypatch, provider, handler):
    requests = []

    def wrapped(request):
        requests.append(request)
        return handler(request)

    monkeypatch.setattr(notify, "_transport", httpx.MockTransport(wrapped))
    monkeypatch.setattr(notify, "load_config", lambda: _Cfg(provider))
    return requests


@pytest.fixture
async def db(tmp_path):
    database = await open_db(tmp_path / "workspace.db")
    async with database.write() as conn:
        await conn.execute(
            "INSERT INTO clients (id, name, token_prefix, token_hash, scopes, created_at)"
            " VALUES ('cli_a', 'alice', 'p', 'h', 'decide:request', 0)"
        )
    yield database
    await database.close()


async def _create(database):
    result = await approvals.create(
        database,
        "cli_a",
        "secret-proj",
        None,
        "Secret title",
        "Secret body",
        None,
        3600,
        0,
        now=1_000_000,
    )
    # Reason: let the fire-and-forget listener task finish without sleeping.
    await asyncio.gather(*list(approvals._listener_tasks))
    return result


def _ok(_request):
    return httpx.Response(200)


async def test_provider_none_adds_nothing(db, monkeypatch):
    requests = _setup(monkeypatch, "none", _ok)
    notify.register(object())
    assert approvals._listeners == []
    await _create(db)
    assert requests == []


async def test_ntfy_sends_one_private_post(db, monkeypatch):
    requests = _setup(monkeypatch, "ntfy", _ok)
    notify.register(object())
    await _create(db)
    assert len(requests) == 1
    req = requests[0]
    assert req.method == "POST"
    assert str(req.url) == f"{URL}/{TOPIC}"
    assert req.content.decode() == "alice: decision waiting"
    sent = req.content.decode() + str(dict(req.headers))
    for leaked in ("Secret title", "Secret body", "secret-proj"):
        assert leaked not in sent
    assert "title" not in {k.lower() for k in req.headers}


@pytest.mark.parametrize("kind", ["500", "connect", "timeout"])
async def test_failures_leave_create_successful(db, monkeypatch, caplog, kind):
    def handler(request):
        if kind == "500":
            return httpx.Response(500)
        if kind == "connect":
            raise httpx.ConnectError("boom", request=request)
        raise httpx.ReadTimeout("slow", request=request)

    _setup(monkeypatch, "ntfy", handler)
    notify.register(object())
    with caplog.at_level(logging.DEBUG):
        result = await _create(db)
    assert result is not None
    assert "ntfy push failed" in caplog.text
    assert TOPIC not in caplog.text


async def test_missing_topic_warns_once_and_sends_nothing(db, monkeypatch, caplog):
    requests = _setup(monkeypatch, "ntfy", _ok)
    monkeypatch.delenv(notify.TOPIC_ENV)
    with caplog.at_level(logging.WARNING):
        notify.register(object())
    assert [r.levelno for r in caplog.records].count(logging.WARNING) == 1
    assert approvals._listeners == []
    await _create(db)
    assert requests == []


async def test_register_twice_sends_one_push(db, monkeypatch):
    requests = _setup(monkeypatch, "ntfy", _ok)
    notify.register(object())
    notify.register(object())
    assert len(approvals._listeners) == 1
    await _create(db)
    assert len(requests) == 1


def test_topic_filter_drops_only_records_with_the_topic():
    flt = notify._TopicFilter("s3cret-topic")

    def make(msg):
        return logging.LogRecord("httpx", logging.INFO, __file__, 1, msg, None, None)

    assert flt.filter(make("HTTP Request: POST https://ntfy.sh/s3cret-topic")) is False
    assert flt.filter(make("HTTP Request: POST http://localhost:11434/api/chat")) is True
