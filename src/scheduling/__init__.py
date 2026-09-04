"""
Market Scheduling and Autonomous Lifecycle Management Module.
"""
from enum import Enum

class MarketPhase(str, Enum):
    PRE_MARKET = "PRE_MARKET"
    REGULAR_HOURS = "REGULAR_HOURS"
    POST_MARKET = "POST_MARKET"
    CLOSED = "CLOSED"
