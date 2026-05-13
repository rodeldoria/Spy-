"""Tests for the premortem engine — heuristic path, parser, cache.

We don't call the real Anthropic API in tests. The AI path is exercised by
monkey-patching the module's `_ai_premortem` to return a canned payload,
which lets us verify the result shape without network or keys.
"""

from __future__ import annotations

import json

import pytest

from monte.intel import premortem as pm


@pytest.fixture(autouse=True)
def _clear_cache():
    pm.clear_cache()
    yield
    pm.clear_cache()


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AI_INTEGRATIONS_ANTHROPIC_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def test_heuristic_returns_three_modes_for_each_horizon(no_api_key):
    for h in ("intraday", "swing", "position", "long"):
        r = pm.premortem(
            title="generic",
            plan="some plan",
            horizon=h,           # type: ignore[arg-type]
            enable_ai=True,      # AI requested, but no key → falls back
        )
        assert r.source == "heuristic"
        assert len(r.failure_modes) == 3
        assert r.biggest_hidden_assumption
        assert 1 <= len(r.prelaunch_checklist) <= 5
        assert len(r.revised_plan) >= 1
        # Sanity: top failure mode has a non-zero risk score and the right
        # likelihood/danger types.
        top = r.top_failure_mode
        assert top is not None
        assert top.likelihood in {"low", "medium", "high", "critical"}
        assert top.danger in {"low", "medium", "high", "critical"}
        assert 1 <= top.risk_score <= 16


def test_empty_plan_returns_empty_result(no_api_key):
    r = pm.premortem(title="x", plan="   ", horizon="swing", enable_ai=False)
    assert r.failure_modes == []
    assert r.source == "heuristic"
    assert "no plan" in (r.ai_error or "")


def test_ai_disabled_skips_network_path(no_api_key):
    r = pm.premortem(
        title="x",
        plan="long btc",
        horizon="swing",
        enable_ai=False,
    )
    assert r.source == "heuristic"
    # No AI error because we never asked.
    assert r.ai_error is None


def test_cache_returns_same_instance(no_api_key):
    r1 = pm.premortem(title="t", plan="some plan", horizon="swing", enable_ai=False)
    r2 = pm.premortem(title="t", plan="some plan", horizon="swing", enable_ai=False)
    assert r1 is r2


def test_cache_key_differs_by_horizon(no_api_key):
    r_swing = pm.premortem(title="t", plan="p", horizon="swing", enable_ai=False)
    r_long = pm.premortem(title="t", plan="p", horizon="long", enable_ai=False)
    assert r_swing is not r_long
    # And the failure-mode names should actually differ across horizons.
    names_swing = {m.name for m in r_swing.failure_modes}
    names_long = {m.name for m in r_long.failure_modes}
    assert names_swing != names_long


# ---------------------------------------------------------------------------
# AI path (mocked)
# ---------------------------------------------------------------------------

_GOOD_PAYLOAD = {
    "failure_modes": [
        {
            "name": "Slippage on thin book",
            "likelihood": "high",
            "danger": "high",
            "chain": "Size is large relative to top-of-book. Fill walks the book.",
            "hidden_assumption": "Your size doesn't move the market.",
            "early_warning": "Depth below 5x intended size.",
        },
        {
            "name": "Gap on news",
            "likelihood": "medium",
            "danger": "critical",
            "chain": "Headline drops, stop gaps.",
            "hidden_assumption": "Stop fills near its price.",
            "early_warning": "Unusual options skew.",
        },
        {
            "name": "Vol expansion",
            "likelihood": "medium",
            "danger": "high",
            "chain": "Realised vol expands, stops trigger on noise.",
            "hidden_assumption": "Today's sigma ≈ recent average.",
            "early_warning": "ATR > 1.5x median.",
        },
    ],
    "biggest_hidden_assumption": "Liquidity holds.",
    "revised_plan": ["Halve size.", "Skip into news.", "Tighten stop.", "Time-stop 90m."],
    "prelaunch_checklist": [
        "Calendar clear",
        "Depth ≥ 5x size",
        "ATR sane",
        "Risk ≤ 0.5%",
        "Time-stop set",
    ],
}


def test_ai_payload_parses_into_result(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_ai(*, title, plan, horizon, context, model):
        return (
            pm.PremortemResult(
                failure_modes=[pm.FailureMode(**m) for m in _GOOD_PAYLOAD["failure_modes"]],
                biggest_hidden_assumption=_GOOD_PAYLOAD["biggest_hidden_assumption"],
                revised_plan=list(_GOOD_PAYLOAD["revised_plan"]),
                prelaunch_checklist=list(_GOOD_PAYLOAD["prelaunch_checklist"]),
                horizon=horizon,
                source="ai",
                model=model,
            ),
            None,
        )

    monkeypatch.setattr(pm, "_ai_premortem", fake_ai)

    r = pm.premortem(title="t", plan="long btc on breakout", horizon="intraday")
    assert r.source == "ai"
    assert len(r.failure_modes) == 3
    assert r.failure_modes[0].name == "Slippage on thin book"
    assert r.biggest_hidden_assumption == "Liquidity holds."
    # Ranking sanity — risk_score is likelihood_rank * danger_rank.
    # Slippage: 3*3=9; Gap: 2*4=8; Vol: 2*3=6. Slippage wins.
    top = r.top_failure_mode
    assert top is not None and top.name == "Slippage on thin book"
    assert top.risk_score == 9


def test_parse_ai_payload_tolerates_code_fences():
    raw = "```json\n" + json.dumps(_GOOD_PAYLOAD) + "\n```"
    parsed, err = pm._parse_ai_payload(raw)
    assert err is None
    assert parsed is not None
    assert len(parsed["failure_modes"]) == 3
    assert parsed["biggest_hidden_assumption"] == "Liquidity holds."


def test_parse_ai_payload_tolerates_leading_prose():
    raw = "Sure, here you go:\n" + json.dumps(_GOOD_PAYLOAD) + "\nLet me know if you need anything else."
    parsed, err = pm._parse_ai_payload(raw)
    assert err is None
    assert parsed is not None
    assert len(parsed["failure_modes"]) == 3


def test_parse_ai_payload_rejects_garbage():
    parsed, err = pm._parse_ai_payload("nope, not json at all")
    assert parsed is None
    assert err is not None


def test_severity_coercion_handles_aliases():
    assert pm._coerce_severity("MED") == "medium"
    assert pm._coerce_severity("moderate") == "medium"
    assert pm._coerce_severity("severe") == "critical"
    assert pm._coerce_severity(None) == "medium"
    assert pm._coerce_severity("high") == "high"


def test_ai_returns_none_when_no_key_and_no_sdk(monkeypatch):
    """Verify the AI helper short-circuits gracefully when there's no API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AI_INTEGRATIONS_ANTHROPIC_API_KEY", raising=False)
    result, err = pm._ai_premortem(
        title="t", plan="p", horizon="swing", context={}, model="x",
    )
    assert result is None
    assert err == "ANTHROPIC_API_KEY not set"
