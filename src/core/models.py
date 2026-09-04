"""
Domain models for the multi-agent trading system.
All schemas are strictly typed with Pydantic for validation and serialization.
"""
from datetime import datetime, date, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from src.core.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    RiskGateStatus,
    SignalDirection,
    AccountType
)


class MarketBar(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None


class Quote(BaseModel):
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float


# --------------------------------------------------------------------
# Multi-Agent Outputs (Strict JSON Schemas)
# --------------------------------------------------------------------

class SignalProposal(BaseModel):
    symbol: str
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reasoning: str
    key_levels: Dict[str, float] = Field(default_factory=dict, description="e.g. support, resistance, stop")


class MarketContextOutput(BaseModel):
    symbol: str
    regime: str = Field(description="e.g. trending_bullish, range_bound, high_volatility")
    event_risk_detected: bool = Field(default=False)
    flags: List[str] = Field(default_factory=list, description="e.g. earnings_soon, macro_announcement, sector_weakness")
    reasoning: str


class RiskReviewOutput(BaseModel):
    symbol: str
    verdict: str = Field(description="APPROVE | CAUTION | REJECT")
    concerns: List[str] = Field(default_factory=list)
    correlation_warning: bool = False
    reasoning: str


class SizingProposal(BaseModel):
    symbol: str
    suggested_shares: int = Field(ge=0)
    suggested_notional_usd: float = Field(ge=0.0)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reasoning: str


class AnomalyOutput(BaseModel):
    is_anomalous: bool = False
    severity: str = Field(default="NONE", description="NONE | LOW | MEDIUM | HIGH | CRITICAL")
    anomalies_detected: List[str] = Field(default_factory=list)
    recommended_action: str = Field(default="CONTINUE", description="CONTINUE | PAUSE_SYMBOL | HALT_SYSTEM")


# --------------------------------------------------------------------
# Orchestrated Proposed Trade
# --------------------------------------------------------------------

class ProposedTrade(BaseModel):
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Agent Analysis Snapshot
    signal_proposal: Optional[SignalProposal] = None
    market_context: Optional[MarketContextOutput] = None
    risk_review: Optional[RiskReviewOutput] = None
    sizing_proposal: Optional[SizingProposal] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------
# Deterministic Risk Gate Decision
# --------------------------------------------------------------------

class RiskGateDecision(BaseModel):
    approved: bool
    status: RiskGateStatus
    original_quantity: int
    allowed_quantity: int
    rule_violations: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------
# Order & Execution Models
# --------------------------------------------------------------------

class Order(BaseModel):
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int = 0
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING_RISK
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    avg_fill_price: Optional[float] = None
    raw_broker_response: Optional[Dict[str, Any]] = None


class Fill(BaseModel):
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    fee: float = 0.0


class Position(BaseModel):
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    updated_at: datetime


class AccountState(BaseModel):
    account_id: str
    account_type: AccountType = AccountType.MARGIN
    equity: float
    cash: float
    settled_cash: float
    unsettled_cash: float
    buying_power: float
    day_trade_count: int = 0
    is_pdt: bool = False
    is_trading_blocked: bool = False
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    starting_daily_equity: float
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------
# Reconciliation & Settlement Models
# --------------------------------------------------------------------

class CashLedgerEntry(BaseModel):
    entry_id: str
    timestamp: datetime
    settlement_date: date
    amount: float
    is_settled: bool
    source_order_id: Optional[str] = None
    description: str


class DayTradeRecord(BaseModel):
    record_id: str
    symbol: str
    trade_date: date
    buy_order_id: str
    sell_order_id: str
    quantity: int
    realized_pnl: float
