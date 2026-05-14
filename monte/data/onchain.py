"""Crypto on-chain snapshot for the event-aware chat widget.

Three independent sources, all free / public, each fetched in parallel:

- **Farside** (`farside.co.uk/?p=997` for BTC, `?p=1045` for ETH) — daily
  net spot-ETF flows. We sum the last 5 trading days as a sentiment proxy.
- **CoinGlass** (`open.coinglass.com/api/v2/funding_rates_chart`) — 8h
  funding-rate history; we surface the current rate plus its z-score
  against the last 30 days. An optional `COINGLASS_API_KEY` raises rate
  limits but isn't required for the public chart endpoint.
- **TokenUnlocks** (`token.unlocks.app`) — JSON of upcoming protocol /
  team unlocks; we surface the next event inside the user's hold window.

Every fetch lives behind a short `lru`-style cache and a `urlopen`
timeout. Failures are captured per-source on the snapshot (`errors`
dict) instead of raising — the gate degrades to "data unavailable" for
that single axis rather than blocking the whole submit.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

_CACHE_TTL_SECONDS = 60 * 10   # 10 minutes — these don't move minute-to-minute
_CACHE: dict[str, tuple[float, "OnChainSnapshot"]] = {}

FARSIDE_URLS = {
    "BTC": "https://farside.co.uk/btc/",
    "ETH": "https://farside.co.uk/eth/",
}
COINGLASS_FUNDING_URL = "https://open-api-v3.coinglass.com/api/futures/fundingRate/oi-weight-ohlc-history"
TOKEN_UNLOCKS_URL = "https://token.unlocks.app/api/upcoming"


@dataclass
class OnChainSnapshot:
    symbol: str
    available: bool
    fetched_at: float = 0.0

    # Farside
    etf_net_flow_5d_musd: Optional[float] = None    # last 5d net flow, $M
    etf_last_day_musd: Optional[float] = None       # most recent day, $M

    # CoinGlass
    funding_rate_now_bps: Optional[float] = None    # 8h funding, bps
    funding_rate_z_30d: Optional[float] = None      # z-score vs trailing 30d

    # TokenUnlocks
    next_unlock_at_utc: Optional[float] = None      # epoch
    next_unlock_pct_supply: Optional[float] = None
    next_unlock_label: str = ""

    errors: dict[str, str] = field(default_factory=dict)

    @property
    def is_crypto(self) -> bool:
        return bool(self.symbol)

    def funding_label(self) -> str:
        if self.funding_rate_now_bps is None:
            return "—"
        return f"{self.funding_rate_now_bps:+.2f} bps / 8h"

    def etf_label(self) -> str:
        if self.etf_net_flow_5d_musd is None:
            return "—"
        return f"5d net ${self.etf_net_flow_5d_musd:+,.0f}M"


def snapshot(symbol: str, *, hold_window_hours: float = 168.0) -> OnChainSnapshot:
    """Return an `OnChainSnapshot` for `symbol`. `symbol` is a yfinance-style
    ticker like "BTC-USD" / "ETH-USD"; we pull the underlying base from it.
    """
    base = _base_symbol(symbol)
    if base not in {"BTC", "ETH"}:
        return OnChainSnapshot(symbol=symbol, available=False)

    cache_key = f"{base}::{int(hold_window_hours)}"
    cached = _CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    snap = OnChainSnapshot(symbol=symbol, available=False, fetched_at=time.time())
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_etf = pool.submit(_fetch_farside, base)
        f_fund = pool.submit(_fetch_coinglass_funding, base)
        f_unlk = pool.submit(_fetch_unlocks, base, hold_window_hours)
        for fut, label, applier in (
            (f_etf, "farside", _apply_farside),
            (f_fund, "coinglass", _apply_coinglass),
            (f_unlk, "tokenunlocks", _apply_unlocks),
        ):
            try:
                applier(snap, fut.result(timeout=8))
            except Exception as e:  # noqa: BLE001
                snap.errors[label] = f"{type(e).__name__}: {str(e)[:100]}"

    snap.available = (
        snap.etf_net_flow_5d_musd is not None
        or snap.funding_rate_now_bps is not None
        or snap.next_unlock_at_utc is not None
    )
    _CACHE[cache_key] = (time.time(), snap)
    return snap


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Farside (ETF flows)
# ---------------------------------------------------------------------------

def _fetch_farside(base: str) -> list[tuple[str, float]]:
    """Return the last ~30 (date, total_flow_usd_millions) rows.

    Farside publishes one HTML table per asset at /btc/ or /eth/. We do not
    need a full HTML parser — a regex over the `<tr>` rows of the daily
    table is robust enough and avoids a heavy dependency.
    """
    url = FARSIDE_URLS.get(base)
    if not url:
        return []
    req = urllib.request.Request(url, headers={"User-Agent": "spy-event-chat/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    return _parse_farside_html(html)


_DATE_PATTERNS = [
    re.compile(r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
]


def _parse_farside_html(html: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    # Each row looks like: <tr><td>21 Nov 2025</td><td>...</td>...<td>(213.4)</td></tr>
    # The TOTAL column is the last numeric td on the row.
    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr_match.group(1), re.DOTALL | re.IGNORECASE)
        if len(tds) < 2:
            continue
        cells = [_strip_html(c).strip() for c in tds]
        date_cell = cells[0]
        if not any(p.match(date_cell) for p in _DATE_PATTERNS):
            continue
        total = _parse_farside_number(cells[-1])
        if total is None:
            # Some rows put the "Total" in the second-to-last cell; try that.
            if len(cells) >= 2:
                total = _parse_farside_number(cells[-2])
        if total is None:
            continue
        rows.append((date_cell, total))
    # Newest-first → oldest-first so a slice of [-5:] is the latest 5 days.
    rows.reverse()
    return rows


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def _parse_farside_number(s: str) -> Optional[float]:
    s = s.replace(",", "").replace("–", "-").replace("−", "-").strip()
    if not s or s == "-":
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _apply_farside(snap: OnChainSnapshot, rows: list[tuple[str, float]]) -> None:
    if not rows:
        return
    last_5 = rows[-5:]
    snap.etf_net_flow_5d_musd = sum(v for _, v in last_5)
    snap.etf_last_day_musd = last_5[-1][1]


# ---------------------------------------------------------------------------
# CoinGlass (funding rate)
# ---------------------------------------------------------------------------

def _fetch_coinglass_funding(base: str) -> list[float]:
    """Return funding-rate history (decimal) for `base`. Empty list on failure."""
    qs = urllib.parse.urlencode(
        {"symbol": base, "interval": "h8", "limit": 90}  # ~30 days of 8h prints
    )
    url = f"{COINGLASS_FUNDING_URL}?{qs}"
    headers = {"User-Agent": "spy-event-chat/1.0"}
    api_key = os.environ.get("COINGLASS_API_KEY", "").strip()
    if api_key:
        headers["coinglassSecret"] = api_key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    out: list[float] = []
    for row in data:
        # CoinGlass returns OHLC of the rate; we use the close.
        if isinstance(row, dict):
            v = row.get("c") or row.get("close") or row.get("rate")
        elif isinstance(row, (list, tuple)) and len(row) >= 5:
            v = row[4]
        else:
            v = None
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _apply_coinglass(snap: OnChainSnapshot, rates: list[float]) -> None:
    if not rates:
        return
    now_rate = rates[-1]
    snap.funding_rate_now_bps = now_rate * 10_000.0   # decimal → bps
    if len(rates) >= 10:
        import statistics
        sample = rates[-90:]
        mu = statistics.fmean(sample)
        sd = statistics.pstdev(sample) or 1e-9
        snap.funding_rate_z_30d = (now_rate - mu) / sd


# ---------------------------------------------------------------------------
# TokenUnlocks
# ---------------------------------------------------------------------------

def _fetch_unlocks(base: str, hold_window_hours: float) -> list[dict]:
    qs = urllib.parse.urlencode({"asset": base})
    url = f"{TOKEN_UNLOCKS_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "spy-event-chat/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    if isinstance(payload, dict):
        items = payload.get("data") or payload.get("upcoming") or []
    else:
        items = payload if isinstance(payload, list) else []
    return items


def _apply_unlocks(snap: OnChainSnapshot, items: list) -> None:
    now = time.time()
    best: Optional[tuple[float, float, str]] = None  # (ts, pct, label)
    for it in items:
        if not isinstance(it, dict):
            continue
        when = it.get("date") or it.get("unlock_at") or it.get("nextUnlockDate")
        ts: Optional[float] = None
        if isinstance(when, (int, float)):
            ts = float(when) if when > 1e12 else float(when)
            ts = ts / 1000.0 if ts > 1e12 else ts
        elif isinstance(when, str):
            try:
                ts = datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = None
        if ts is None or ts < now:
            continue
        pct = _coerce_float(it.get("percentSupply") or it.get("percent_of_supply") or it.get("percent"))
        label = str(it.get("token") or it.get("symbol") or it.get("project") or "unlock")
        if best is None or ts < best[0]:
            best = (ts, pct or 0.0, label)
    if best is not None:
        snap.next_unlock_at_utc = best[0]
        snap.next_unlock_pct_supply = best[1] or None
        snap.next_unlock_label = best[2]


def _coerce_float(v: object) -> Optional[float]:
    try:
        return float(v) if v is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if "-" in s:
        return s.split("-")[0]
    return s


__all__ = ["OnChainSnapshot", "snapshot", "clear_cache"]
