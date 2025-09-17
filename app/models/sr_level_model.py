from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd


@dataclass
class SRLevel:
    start_time: pd.Timestamp
    price: float
    sr_type: str                       # "Support" | "Resistance"
    strength: int = 1

    break_time: Optional[pd.Timestamp] = None
    retest_times: List[pd.Timestamp] = field(default_factory=list)
    test_times: List[pd.Timestamp] = field(default_factory=list)
    rejection_times: List[pd.Timestamp] = field(default_factory=list)
    manipulation_times: List[pd.Timestamp] = field(default_factory=list)

    # flags
    has_breakout: bool = False
    has_test: bool = False
    has_retest: bool = False
    has_rejection: bool = False
    has_manipulation: bool = False

    # misc
    ephemeral: bool = False
    timeframe_str: str = ""            # filled by MTF wrapper

