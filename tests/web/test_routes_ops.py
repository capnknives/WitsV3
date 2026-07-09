"""Tests for tool metrics and operator routes."""

from tests.web.test_web_server import client_noauth


def test_tool_metrics_recorder():
    from core.mcp_health import clear_server_health, get_server_health, record_connect_failure, record_connect_success
    from core.tool_metrics import ToolMetricsRecorder

    rec = ToolMetricsRecorder()
    rec.record("web_search", 120.0, success=True)
    rec.record("web_search", 80.0, success=False, error="timeout")
    snap = rec.snapshot()
    assert len(snap) == 1
    row = snap[0]
    assert row["tool"] == "web_search"
    assert row["calls"] == 2
    assert row["failures"] == 1
    assert row["avg_latency_ms"] == 100.0

    clear_server_health()
    record_connect_failure("memory", "connection refused")
    record_connect_success("memory")
    health = get_server_health("memory")
    assert health["connected"] is True
    assert health["last_error"] is None
    clear_server_health()


def test_metrics_tools_endpoint(client_noauth):
    client, _ = client_noauth
    res = client.get("/api/metrics/tools")
    assert res.status_code == 200
    assert "tools" in res.json()


def test_tasks_endpoint(client_noauth):
    client, _ = client_noauth
    res = client.get("/api/tasks")
    assert res.status_code == 200
    body = res.json()
    assert any(t["id"] == "daily_self_repair" for t in body["tasks"])


def test_offline_mode_get(client_noauth):
    client, _ = client_noauth
    res = client.get("/api/security/offline")
    assert res.status_code == 200
    assert "offline_mode" in res.json()
