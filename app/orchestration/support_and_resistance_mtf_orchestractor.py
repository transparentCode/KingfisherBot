from typing import List, Dict

from app.indicators.support_and_resistance import SupportResistanceDetector
from app.models.sr_level_model import SRLevel
import pandas as pd


class SupportResistanceOrchestrator:
    """
    Holds one detector per timeframe string, merges levels on demand.
    """

    def __init__(self, timeframes: List[str]):
        self.detectors: Dict[str, SupportResistanceDetector] = {
            tf: SupportResistanceDetector() for tf in timeframes
        }

    def update(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        data = {"1m": df_1m, "15m": df_15m, ...}
        """
        for tf, df in data.items():
            self.detectors[tf].update(df)

    # — results ————————————————————
    def get_all_levels(self) -> List[dict]:
        out = []
        for tf, det in self.detectors.items():
            for d in det.to_dict():
                d["timeframe"] = tf
                out.append(d)
        return out

    def merge_levels(self, price_tol: float = 0.001) -> List[SRLevel]:
        """
        De-duplicates across TFs (first come, first kept).
        """
        merged: List[SRLevel] = []
        for tf, det in self.detectors.items():
            for lv in det.levels:
                if any(abs(lv.price - x.price) < price_tol for x in merged):
                    continue
                lv.timeframe_str = tf
                merged.append(lv)
        return merged