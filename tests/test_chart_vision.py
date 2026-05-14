"""Tests for the chart-vision wrapper.

We never call the real Anthropic API. The AI path is exercised by
monkey-patching the SDK client to return a canned `messages.create`
response, which lets us verify the JSON parser + result shape.
"""
from __future__ import annotations

import sys
import types

import pytest

from monte.intel import chart_vision as cv


@pytest.fixture(autouse=True)
def _reset_cache():
    cv.clear_cache()
    yield
    cv.clear_cache()


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AI_INTEGRATIONS_ANTHROPIC_API_KEY", raising=False)


def _make_fake_anthropic(reply_text: str):
    class _Block:
        def __init__(self, t: str):
            self.text = t
            self.type = "text"

    class _Msg:
        def __init__(self, t: str):
            self.content = [_Block(t)]

    class _Messages:
        def create(self, **_kwargs):
            return _Msg(reply_text)

    class _Client:
        def __init__(self, *_a, **_kw):
            self.messages = _Messages()

    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = _Client
    return fake_module


def test_no_image_bytes_returns_heuristic(no_api_key):
    res = cv.read_chart(b"")
    assert res.source == "heuristic"
    assert "no image" in res.error.lower()


def test_no_api_key_falls_back(no_api_key):
    res = cv.read_chart(b"\x89PNG\r\n\x1a\n", media_type="image/png")
    assert res.source == "heuristic"
    assert "ANTHROPIC_API_KEY" in res.error


def test_ai_disabled_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    res = cv.read_chart(b"abc", enable_ai=False)
    assert res.source == "heuristic"
    assert "disabled" in res.error.lower()


def test_ai_path_parses_clean_reply(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = (
        '{"ticker":"BTCUSD","timeframe":"4h","setup":"Ascending triangle into 70k.",'
        '"key_levels":[68000, 70000, 72000],"suspected_pattern":"ascending triangle",'
        '"bias":"bullish"}'
    )
    fake = _make_fake_anthropic(payload)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    res = cv.read_chart(b"\x89PNG\r\n\x1a\n", media_type="image/png", cache=False)
    assert res.source == "ai"
    assert res.ticker == "BTCUSD"
    assert res.timeframe == "4h"
    assert res.bias == "bullish"
    assert res.suspected_pattern == "ascending triangle"
    assert res.key_levels == [68000.0, 70000.0, 72000.0]


def test_ai_path_recovers_when_reply_wraps_json_in_prose(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = (
        "Here's the read:\n"
        "```json\n"
        '{"ticker":"SPY","timeframe":"15m","setup":"Pullback to VWAP.","key_levels":[],'
        '"suspected_pattern":"","bias":"neutral"}\n'
        "```\nLet me know if you want more detail."
    )
    fake = _make_fake_anthropic(payload)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    res = cv.read_chart(b"\x89PNG\r\n\x1a\n", media_type="image/png", cache=False)
    assert res.source == "ai"
    assert res.ticker == "SPY"
    assert res.bias == "neutral"


def test_unparseable_reply_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake = _make_fake_anthropic("not even close to JSON")
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    res = cv.read_chart(b"\x89PNG\r\n\x1a\n", media_type="image/png", cache=False)
    assert res.source == "heuristic"
    assert res.error
