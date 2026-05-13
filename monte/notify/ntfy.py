"""ntfy.sh push notifications.

Topic-based, no API key. Install the ntfy app on your phone, subscribe to
your topic, and these calls land as native notifications. Silent no-op when
`MONTE_NTFY_TOPIC` is not configured so nothing breaks in dev.

We deliberately use stdlib `urllib` (no `requests` dependency added).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable


DEFAULT_SERVER = "https://ntfy.sh"


def _topic() -> str | None:
    topic = (os.environ.get("MONTE_NTFY_TOPIC") or "").strip()
    return topic or None


def _server() -> str:
    return (os.environ.get("MONTE_NTFY_SERVER") or DEFAULT_SERVER).rstrip("/")


def push(
    title: str,
    body: str,
    *,
    topic: str | None = None,
    priority: str = "default",
    tags: Iterable[str] | None = None,
    click_url: str | None = None,
    timeout: float = 3.0,
) -> bool:
    """Send a single push. Returns True if accepted by the server.

    `priority`: one of "min" / "low" / "default" / "high" / "urgent"
    `tags`: ntfy emoji shortcodes — e.g. ["money_with_wings", "chart"]
    """
    target = (topic or _topic() or "").strip()
    if not target:
        return False
    url = f"{_server()}/{target}"
    headers = {
        "Title": title.encode("ascii", "ignore").decode() or "Monte Edge",
        "Priority": priority,
        "Content-Type": "text/plain; charset=utf-8",
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    if click_url:
        headers["Click"] = click_url
    req = urllib.request.Request(url, data=body.encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def push_alert(alert_dict: dict, *, base_url: str | None = None) -> bool:
    """Format an EdgeSignal-style dict and push it.

    Expects the keys produced by `monte.strategy.monte_edge.evaluate()`:
    symbol, action, tier, confidence, spot, entry, stop, target, reasoning.
    """
    tier = str(alert_dict.get("tier", "")).upper()
    if tier not in {"ACT_NOW", "WATCH"}:
        return False
    priority = "high" if tier == "ACT_NOW" else "default"
    tag = "rotating_light" if tier == "ACT_NOW" else "eyes"
    direction_tag = (
        "money_with_wings" if "BUY" in str(alert_dict.get("action", "")) else "warning"
    )

    sym = alert_dict.get("symbol", "?")
    act = str(alert_dict.get("action", "?")).replace("_", " ")
    conf = float(alert_dict.get("confidence", 0))
    spot = float(alert_dict.get("spot", alert_dict.get("entry", 0)))
    stop = float(alert_dict.get("stop", 0))
    tgt = float(alert_dict.get("target", 0))
    rr = float(alert_dict.get("rr", 0))

    if tier == "ACT_NOW":
        title = f"ACT NOW · {sym} {act} · {conf:.0f}%"
    else:
        title = f"Watch · {sym} {act} · {conf:.0f}%"

    body_lines = [
        f"Spot ${spot:,.2f}  Stop ${stop:,.2f}  Target ${tgt:,.2f}  R:R {rr:.2f}",
    ]
    reasoning = alert_dict.get("reasoning")
    if reasoning:
        body_lines.append(f"Why: {reasoning}")
    options = alert_dict.get("options_ticket")
    if options:
        body_lines.append(
            f"Options: {options.get('side','')} ${options.get('strike',0):.0f} "
            f"{options.get('expiry','')} @ ~${options.get('premium',0):.2f}"
        )

    return push(
        title=title,
        body="\n".join(body_lines),
        priority=priority,
        tags=[tag, direction_tag],
        click_url=base_url,
    )


__all__ = ["push", "push_alert"]
