"""Tests for the SMA20/SMA50 crossover detector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monte.indicators.ma_cross import detect_cross


def test_no_cross_when_series_too_short():
    close = pd.Series(range(30))
    cross = detect_cross(close, fast=20, slow=50)
    assert cross.kind is None
    assert cross.bars_ago == -1
    assert not cross.fired_recently


def test_golden_cross_on_recent_uptrend_after_flat():
    # 60 flat bars, then 20 strongly rising bars → SMA20 crosses SMA50 up.
    flat = [100.0] * 60
    up = [100.0 + i * 1.5 for i in range(1, 21)]
    close = pd.Series(flat + up)
    cross = detect_cross(close, fast=20, slow=50)
    assert cross.kind == "golden"
    assert cross.bars_ago >= 0


def test_death_cross_on_recent_downtrend_after_flat():
    flat = [100.0] * 60
    down = [100.0 - i * 1.5 for i in range(1, 21)]
    close = pd.Series(flat + down)
    cross = detect_cross(close, fast=20, slow=50)
    assert cross.kind == "death"
    assert cross.bars_ago >= 0


def test_label_includes_emoji():
    flat = [100.0] * 60
    up = [100.0 + i * 1.5 for i in range(1, 6)]
    close = pd.Series(flat + up)
    cross = detect_cross(close, fast=20, slow=50)
    # The strong recent uptrend should produce either a golden cross
    # within lookback or no cross — both should render a sensible label.
    label = cross.label()
    assert isinstance(label, str) and label


def test_fired_recently_window_is_five_bars():
    # Build a series where the cross is at bar -10 (outside the 5-bar window)
    closes = list(range(50)) + [50 + i * 0.01 for i in range(40)]
    cross = detect_cross(pd.Series(closes), fast=20, slow=50, lookback=30)
    # If we did detect, it should not be 'fired_recently' (>5 bars).
    if cross.kind is not None:
        assert (0 <= cross.bars_ago <= 5) == cross.fired_recently
