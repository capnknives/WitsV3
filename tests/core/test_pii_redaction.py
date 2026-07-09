"""Tests for PII redaction."""

from core.config import WitsV3Config
from core.pii_redaction import maybe_redact_for_guest, maybe_redact_for_storage, redact_pii


def test_redact_email():
    out = redact_pii("Contact me at alice@example.com please")
    assert "alice@example.com" not in out
    assert "[REDACTED]" in out


def test_guest_storage_redaction():
    config = WitsV3Config()
    config.security.pii_redaction.enabled = True
    config.security.pii_redaction.redact_before_store = True
    out = maybe_redact_for_storage("phone 555-123-4567", config, role="guest")
    assert "555-123-4567" not in out


def test_owner_storage_not_redacted_by_default():
    config = WitsV3Config()
    config.security.pii_redaction.enabled = True
    raw = "alice@example.com"
    out = maybe_redact_for_storage(raw, config, role="owner")
    assert out == raw


def test_guest_response_redaction():
    config = WitsV3Config()
    config.security.pii_redaction.enabled = True
    out = maybe_redact_for_guest("email bob@test.org", config, role="guest")
    assert "bob@test.org" not in out
