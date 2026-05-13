"""Option-chain helper — any optionable ticker (yfinance).

Picks an at-the-money contract in the 30–45 DTE window for stocks/ETFs.
For crypto, returns a structured suggestion note since exchange-traded
crypto options aren't available via yfinance.

Returns `None` when yfinance is unavailable or the chain is empty so the
caller never crashes the dashboard.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def list_expirations(symbol: str = "SPY") -> list[str]:
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        return list(t.options or [])
    except Exception:
        return []


def _pick_expiry(expiries: list[str], *, min_dte: int = 30, max_dte: int = 45) -> str | None:
    today = datetime.now(timezone.utc).date()
    best: tuple[int, str] | None = None
    for ex in expiries:
        try:
            d = datetime.strptime(ex, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (d - today).days
        if min_dte <= dte <= max_dte:
            return ex
        if dte >= min_dte and (best is None or dte < best[0]):
            best = (dte, ex)
    return best[1] if best else (expiries[-1] if expiries else None)


def _atm_row(chain_df, spot: float):
    """Return the option row whose strike is closest to spot."""
    if chain_df is None or chain_df.empty:
        return None
    diffs = (chain_df["strike"] - spot).abs()
    idx = diffs.idxmin()
    return chain_df.loc[idx]


def _is_crypto(symbol: str) -> bool:
    s = symbol.upper()
    return "-" in s and s.split("-")[-1] in {"USD", "USDC", "USDT", "BTC"}


def suggest_contract(
    direction: str,
    spot: float,
    *,
    symbol: str = "SPY",
    horizon_days: int = 35,
) -> dict[str, Any] | None:
    """Suggest one ATM debit contract aligned with `direction` ('long' / 'short').

    'long'  → ATM call (bullish)
    'short' → ATM put  (bearish)

    For crypto symbols, returns a structured note about available instruments
    since exchange-traded options aren't available via yfinance for crypto.

    Returns a dict with strike, expiry, premium, breakeven, est_delta, and
    a one-line `rationale`. Returns None when no chain data is available.
    """
    if direction not in {"long", "short"}:
        return None
    if spot <= 0:
        return None

    # Crypto: suggest futures/perpetual swap alternatives
    if _is_crypto(symbol):
        side = "CALL / long perpetual" if direction == "long" else "PUT / short perpetual"
        return {
            "symbol": symbol,
            "side": side,
            "direction": direction,
            "strike": round(spot, 2),
            "expiry": "perpetual",
            "premium": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "breakeven": round(spot * (1.01 if direction == "long" else 0.99), 2),
            "est_delta": 1.0 if direction == "long" else -1.0,
            "iv": None,
            "rationale": (
                f"Crypto options aren't available via yfinance. For {symbol} consider: "
                f"{'long spot or buy a call on Deribit/Coinbase' if direction == 'long' else 'short spot or buy a put on Deribit/Coinbase'}. "
                f"Alternatively, use CME Bitcoin futures (BTC) or micro contracts (MBT)."
            ),
            "max_risk_per_contract": 0.0,
            "is_crypto_note": True,
        }

    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        ticker = yf.Ticker(symbol)
        expiries = list(ticker.options or [])
        if not expiries:
            return None
        expiry = _pick_expiry(
            expiries,
            min_dte=max(7, horizon_days - 10),
            max_dte=horizon_days + 15,
        )
        if not expiry:
            return None
        chain = ticker.option_chain(expiry)
    except Exception:
        return None

    side = "CALL" if direction == "long" else "PUT"
    df = chain.calls if direction == "long" else chain.puts
    row = _atm_row(df, spot)
    if row is None:
        return None

    strike = float(row.get("strike", spot))
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    last = float(row.get("lastPrice", 0) or 0)
    mid = ((bid + ask) / 2.0) if (bid > 0 and ask > 0) else (last or bid or ask)
    if mid <= 0:
        mid = max(0.05, abs(spot - strike) + 0.5)

    moneyness = (spot - strike) / max(spot, 1e-9)
    delta_proxy = 0.5 + moneyness * 5.0
    delta_proxy = max(0.1, min(0.9, delta_proxy))
    if direction == "short":
        delta_proxy = -delta_proxy

    breakeven = strike + mid if direction == "long" else strike - mid
    iv = row.get("impliedVolatility")
    try:
        iv_val = float(iv) if iv is not None else None
    except (TypeError, ValueError):
        iv_val = None

    rationale = (
        f"ATM {side} at the 30-45 DTE sweet spot: highest gamma per dollar of "
        f"theta. Defined risk = ${mid:.2f}/share premium; max loss is the "
        f"premium paid. Breakeven {breakeven:.2f}."
    )

    return {
        "symbol": symbol,
        "side": side,
        "direction": direction,
        "strike": round(strike, 2),
        "expiry": expiry,
        "premium": round(mid, 2),
        "bid": bid,
        "ask": ask,
        "breakeven": round(breakeven, 2),
        "est_delta": round(delta_proxy, 2),
        "iv": iv_val,
        "rationale": rationale,
        "max_risk_per_contract": round(mid * 100, 2),
        "is_crypto_note": False,
    }


__all__ = ["suggest_contract", "list_expirations"]
