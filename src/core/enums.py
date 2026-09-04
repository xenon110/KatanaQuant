"""
Core domain enumerations for the multi-agent trading system.
"""
from enum import Enum


class TradingMode(str, Enum):
    DRY_RUN = "DRY_RUN"      # Pure local simulation, no orders sent
    PAPER = "PAPER"          # Sent to Alpaca paper trading API
    LIVE = "LIVE"            # Real money execution with broker


class AccountType(str, Enum):
    MARGIN = "MARGIN"
    CASH = "CASH"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    PENDING_RISK = "PENDING_RISK"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class RiskGateStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


class SignalDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class AgentRole(str, Enum):
    SIGNAL_RESEARCH = "SIGNAL_RESEARCH"
    MARKET_CONTEXT = "MARKET_CONTEXT"
    RISK_REVIEWER = "RISK_REVIEWER"
    PORTFOLIO_SIZING = "PORTFOLIO_SIZING"
    ANOMALY_MONITOR = "ANOMALY_MONITOR"
