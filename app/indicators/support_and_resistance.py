from __future__ import annotations
from typing import List, Dict
import pandas as pd

from app.models.sr_level_model import SRLevel


class SupportResistanceDetector:
    """
    single-TF engine that reproduces the pivot, break and retest
    logic of the original Pine script.  Heavy loops are avoided; only the
    per-level state mutation is iterative.
    """

    # ---------- init ----------
    def __init__(self, pivot_len: int = 15, min_strength: int = 1, atr_len: int = 20, avoid_false_breaks: bool = False,
        break_vol_thresh: float = 0.30, retest_volume_filter: bool = True, retest_vol_thresh: float = 0.80,):
        self.pivot_len = pivot_len
        self.min_strength = min_strength
        self.atr_len = atr_len

        self.avoid_false_breaks = avoid_false_breaks
        self.break_vol_thresh = break_vol_thresh
        self.retest_volume_filter = retest_volume_filter
        self.retest_vol_thresh = retest_vol_thresh

        self.levels: List[SRLevel] = []

    # ---------- public ----------
    def update(self, df: pd.DataFrame) -> None:
        """
        Ingest a complete OHLCV DataFrame (index = Timestamp).
        Re-runs detection end-to-end; cheap enough for intraday feeds.
        Columns required: high, low, close, volume
        """
        if df.empty:
            return

        df = df.copy()
        df["atr"] = self._atr(df, self.atr_len)

        # 1. new pivot discovery
        for ts, px in self._pivots(df["low"], kind="low"):
            if self._is_unique(px):
                self.levels.append(SRLevel(ts, px, "Support"))

        for ts, px in self._pivots(df["high"], kind="high"):
            if self._is_unique(px):
                self.levels.append(SRLevel(ts, px, "Resistance"))

        # 2. per-level state update
        for lv in self.levels:
            self._process_level(lv, df.loc[lv.start_time:])

        # 3. prune weak
        self.levels = [lv for lv in self.levels if lv.strength >= self.min_strength]

    # ---------- helpers ----------
    def _process_level(self, lv: SRLevel, sub: pd.DataFrame) -> None:
        if sub.empty:
            return
        last = sub.iloc[-1]

        # a) break
        if lv.break_time is None:
            if lv.sr_type == "Resistance" and last.high > lv.price:
                if self._break_ok(last, sub):
                    lv.break_time, lv.has_breakout = last.name, True
            elif lv.sr_type == "Support" and last.low < lv.price:
                if self._break_ok(last, sub):
                    lv.break_time, lv.has_breakout = last.name, True

        # b) retest (only if unbroken)
        if lv.break_time is None and self._touched(lv, last):
            if self._retest_ok(last, sub):
                lv.retest_times.append(last.name)
                lv.strength += 1
                lv.has_retest = True

    # ---------- tiny utils ----------
    def _break_ok(self, row, sub) -> bool:
        if not self.avoid_false_breaks:
            return True
        return row.volume > sub.volume.mean() * self.break_vol_thresh

    def _retest_ok(self, row, sub) -> bool:
        if not self.retest_volume_filter:
            return True
        ma10 = sub.volume.rolling(10).mean().iloc[-1]
        return row.volume > ma10 * self.retest_vol_thresh

    @staticmethod
    def _atr(df, n):
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
        ).max(axis=1)
        return tr.rolling(n).mean()

    def _pivots(self, s: pd.Series, *, kind: str):
        n = self.pivot_len
        vals, idx = s.values, s.index
        for i in range(n, len(s) - n):
            win = vals[i - n : i + n + 1]
            cen = vals[i]
            if (kind == "high" and cen == win.max()) or (kind == "low" and cen == win.min()):
                yield idx[i], cen

    def _is_unique(self, price: float, tol: float = 1e-3) -> bool:
        return all(abs(price - lv.price) > tol for lv in self.levels)

    def _touched(self, lv: SRLevel, row) -> bool:
        if lv.sr_type == "Resistance":
            return row.high >= lv.price >= row.close
        return row.low <= lv.price <= row.close

    # ---------- convenience ----------
    def to_dict(self) -> List[dict]:
        return [
            {
                "start_time": lv.start_time,
                "price": lv.price,
                "type": lv.sr_type,
                "strength": lv.strength,
                "break_time": lv.break_time,
                "has_retest": lv.has_retest,
            }
            for lv in self.levels
        ]
