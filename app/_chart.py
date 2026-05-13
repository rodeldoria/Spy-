"""Full TradingView-style chart for a symbol.

Four stacked subplots:
  1. Price candles + Bollinger bands + SMA20 + SMA50 + SMA200
  2. Volume (red/green)
  3. RSI(14) with 30/70 dashed bands
  4. MACD(12,26,9) histogram + signal + macd line

Optionally annotates the most recent SMA20/SMA50 cross with a vertical
dashed line and emoji marker so the user can spot golden/death crosses
at a glance.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd

from monte.indicators.ma_cross import MACross, detect_cross
from monte.indicators.technical import bollinger, macd, rsi


PRICE_GREEN = "#0a7d2a"
PRICE_RED = "#a8261f"
BB_FILL = "rgba(60, 110, 200, 0.08)"
BB_LINE = "rgba(60, 110, 200, 0.5)"
SMA20_LINE = "#1d8237"
SMA50_LINE = "#a16207"
SMA200_LINE = "#7c3aed"


def _fmt_candles(df: pd.DataFrame) -> go.Candlestick:
    return go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=PRICE_GREEN, decreasing_line_color=PRICE_RED,
        increasing_fillcolor=PRICE_GREEN, decreasing_fillcolor=PRICE_RED,
        name="Price", showlegend=False,
    )


def _bb_traces(df: pd.DataFrame) -> list[go.Scatter]:
    bb = bollinger(df["Close"])
    traces = [
        go.Scatter(
            x=df.index, y=bb["upper"], line=dict(color=BB_LINE, width=1),
            name="BB upper", showlegend=False, hovertemplate="BB up %{y:.2f}<extra></extra>",
        ),
        go.Scatter(
            x=df.index, y=bb["lower"], line=dict(color=BB_LINE, width=1),
            fill="tonexty", fillcolor=BB_FILL,
            name="BB lower", showlegend=False, hovertemplate="BB lo %{y:.2f}<extra></extra>",
        ),
        go.Scatter(
            x=df.index, y=bb["mid"], line=dict(color=BB_LINE, width=1, dash="dot"),
            name="BB mid", showlegend=False, hovertemplate="BB mid %{y:.2f}<extra></extra>",
        ),
    ]
    return traces


def _sma_traces(df: pd.DataFrame) -> list[go.Scatter]:
    close = df["Close"]
    traces = []
    for period, color, name in (
        (20, SMA20_LINE, "SMA20"),
        (50, SMA50_LINE, "SMA50"),
        (200, SMA200_LINE, "SMA200"),
    ):
        if len(close) >= period:
            sma = close.rolling(period).mean()
            traces.append(
                go.Scatter(
                    x=df.index, y=sma, line=dict(color=color, width=1.6),
                    name=name, showlegend=True,
                    hovertemplate=f"{name} %{{y:.2f}}<extra></extra>",
                )
            )
    return traces


def _volume_trace(df: pd.DataFrame) -> go.Bar | None:
    if "Volume" not in df.columns:
        return None
    colors = [
        PRICE_GREEN if c >= o else PRICE_RED
        for c, o in zip(df["Close"], df["Open"])
    ]
    return go.Bar(
        x=df.index, y=df["Volume"], marker_color=colors,
        name="Volume", showlegend=False,
        hovertemplate="Vol %{y:,.0f}<extra></extra>",
    )


def _rsi_traces(df: pd.DataFrame) -> list[go.Scatter]:
    r = rsi(df["Close"])
    return [
        go.Scatter(
            x=df.index, y=r, line=dict(color="#7c3aed", width=1.6),
            name="RSI(14)", showlegend=False,
            hovertemplate="RSI %{y:.1f}<extra></extra>",
        ),
    ]


def _macd_traces(df: pd.DataFrame) -> list:
    m = macd(df["Close"])
    hist = m["hist"]
    bar_colors = [
        PRICE_GREEN if v >= 0 else PRICE_RED for v in hist
    ]
    return [
        go.Bar(
            x=df.index, y=hist, marker_color=bar_colors,
            name="MACD hist", showlegend=False, opacity=0.6,
            hovertemplate="Hist %{y:.3f}<extra></extra>",
        ),
        go.Scatter(
            x=df.index, y=m["macd"], line=dict(color="#1d8237", width=1.6),
            name="MACD", showlegend=False,
            hovertemplate="MACD %{y:.3f}<extra></extra>",
        ),
        go.Scatter(
            x=df.index, y=m["signal"], line=dict(color="#a16207", width=1.6, dash="dot"),
            name="Signal", showlegend=False,
            hovertemplate="Signal %{y:.3f}<extra></extra>",
        ),
    ]


def build_chart(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    show_volume: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
    ma_cross: MACross | None = None,
    height: int = 640,
) -> go.Figure:
    """Build a multi-panel chart. Returns a Plotly Figure.

    Requires a DataFrame with Open/High/Low/Close/Volume columns and a
    datetime-like index.
    """
    if df is None or df.empty or "Close" not in df.columns:
        fig = go.Figure()
        fig.update_layout(height=height, title=title or "")
        return fig

    panels = ["price"]
    if show_volume and "Volume" in df.columns:
        panels.append("volume")
    if show_rsi:
        panels.append("rsi")
    if show_macd:
        panels.append("macd")

    # Row heights: price gets the lion's share, others get 15-20%.
    height_map = {"price": 0.55, "volume": 0.12, "rsi": 0.16, "macd": 0.17}
    row_heights = [height_map[p] for p in panels]
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=len(panels), cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=row_heights,
    )

    # Row 1 — price + BB + SMAs.
    fig.add_trace(_fmt_candles(df), row=1, col=1)
    for tr in _bb_traces(df):
        fig.add_trace(tr, row=1, col=1)
    for tr in _sma_traces(df):
        fig.add_trace(tr, row=1, col=1)

    # MA cross annotation.
    if ma_cross is not None and ma_cross.fired_recently:
        idx = len(df.index) - 1 - ma_cross.bars_ago
        if 0 <= idx < len(df.index):
            cross_ts = df.index[idx]
            cross_color = PRICE_GREEN if ma_cross.kind == "golden" else PRICE_RED
            fig.add_vline(
                x=cross_ts, line_dash="dash", line_color=cross_color,
                opacity=0.65, row=1, col=1,
            )
            fig.add_annotation(
                x=cross_ts, y=df["High"].iloc[idx],
                text="🟢 Golden" if ma_cross.kind == "golden" else "🔴 Death",
                showarrow=True, arrowhead=2,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=cross_color, borderwidth=1,
                font=dict(color=cross_color, size=11),
                row=1, col=1,
            )

    row_idx = 2
    if "volume" in panels:
        v = _volume_trace(df)
        if v is not None:
            fig.add_trace(v, row=row_idx, col=1)
        row_idx += 1

    if "rsi" in panels:
        for tr in _rsi_traces(df):
            fig.add_trace(tr, row=row_idx, col=1)
        # 30 / 70 dashed reference lines
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(160,160,160,0.6)", row=row_idx, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(160,160,160,0.6)", row=row_idx, col=1)
        fig.update_yaxes(range=[0, 100], row=row_idx, col=1, title_text="RSI")
        row_idx += 1

    if "macd" in panels:
        for tr in _macd_traces(df):
            fig.add_trace(tr, row=row_idx, col=1)
        fig.add_hline(y=0, line_color="rgba(160,160,160,0.5)", row=row_idx, col=1)
        fig.update_yaxes(title_text="MACD", row=row_idx, col=1)
        row_idx += 1

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=10, t=30 if title else 6, b=0),
        title=title or None,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.10)", zeroline=False)

    return fig


__all__ = ["build_chart"]
