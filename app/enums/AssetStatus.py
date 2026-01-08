from enum import Enum

class AssetStatus(Enum):
    PENDING = "PENDING"
    FETCHING_HISTORY = "FETCHING_HISTORY"
    STARTING_STREAM = "STARTING_STREAM"
    READY = "READY"
    ERROR = "ERROR"
