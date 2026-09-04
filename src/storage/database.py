"""
Persistence layer using SQLite and SQLAlchemy async engine.
Provides persistent audit trails for signals, agent reasoning, risk decisions, and orders.
"""
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import json
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Date,
    Text,
    create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class SignalLog(Base):
    __tablename__ = "signal_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    direction = Column(String(16))
    confidence = Column(Float)
    reasoning = Column(Text)
    key_levels = Column(Text)  # JSON


class AgentAuditLog(Base):
    __tablename__ = "agent_audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), index=True)
    agent_role = Column(String(32), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prompt = Column(Text)
    raw_response = Column(Text)
    parsed_output = Column(Text)  # JSON


class RiskDecisionLog(Base):
    __tablename__ = "risk_decision_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean)
    status = Column(String(16))
    original_quantity = Column(Integer)
    allowed_quantity = Column(Integer)
    violations = Column(Text)  # JSON
    reasons = Column(Text)     # JSON


class OrderRecord(Base):
    __tablename__ = "orders"
    order_id = Column(String(64), primary_key=True)
    client_order_id = Column(String(64), unique=True, index=True)
    symbol = Column(String(16), index=True)
    side = Column(String(8))
    quantity = Column(Integer)
    filled_quantity = Column(Integer, default=0)
    order_type = Column(String(16))
    limit_price = Column(Float, nullable=True)
    status = Column(String(32), index=True)
    submitted_at = Column(DateTime, nullable=True)
    filled_at = Column(DateTime, nullable=True)
    avg_fill_price = Column(Float, nullable=True)
    raw_response = Column(Text, nullable=True)


class CashLedgerRecord(Base):
    __tablename__ = "cash_ledger"
    entry_id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    settlement_date = Column(Date, index=True)
    amount = Column(Float)
    is_settled = Column(Boolean, default=False)
    source_order_id = Column(String(64), nullable=True)
    description = Column(String(255))


class DayTradeLog(Base):
    __tablename__ = "day_trades"
    record_id = Column(String(64), primary_key=True)
    symbol = Column(String(16), index=True)
    trade_date = Column(Date, index=True)
    buy_order_id = Column(String(64))
    sell_order_id = Column(String(64))
    quantity = Column(Integer)
    realized_pnl = Column(Float)


class DatabaseManager:
    def __init__(self, db_url: str = "sqlite:///./trading_system.db"):
        # Normalize sqlite+aiosqlite to synchronous sqlite if needed for sync ORM
        sync_url = db_url.replace("sqlite+aiosqlite:///", "sqlite:///")
        self.engine = create_engine(sync_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def log_signal(self, symbol: str, direction: str, confidence: float, reasoning: str, key_levels: Dict[str, Any]):
        with self.SessionLocal() as session:
            record = SignalLog(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                reasoning=reasoning,
                key_levels=json.dumps(key_levels)
            )
            session.add(record)
            session.commit()

    def log_agent_audit(self, symbol: str, role: str, prompt: str, raw_response: str, parsed: Dict[str, Any]):
        with self.SessionLocal() as session:
            record = AgentAuditLog(
                symbol=symbol,
                agent_role=role,
                prompt=prompt,
                raw_response=raw_response,
                parsed_output=json.dumps(parsed)
            )
            session.add(record)
            session.commit()

    def log_risk_decision(self, symbol: str, approved: bool, status: str, orig_qty: int, allow_qty: int, violations: List[str], reasons: List[str]):
        with self.SessionLocal() as session:
            record = RiskDecisionLog(
                symbol=symbol,
                approved=approved,
                status=status,
                original_quantity=orig_qty,
                allowed_quantity=allow_qty,
                violations=json.dumps(violations),
                reasons=json.dumps(reasons)
            )
            session.add(record)
            session.commit()

    def save_order(self, order_id: str, client_order_id: str, symbol: str, side: str, qty: int, status: str, order_type: str = "MARKET"):
        with self.SessionLocal() as session:
            record = OrderRecord(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                status=status,
                order_type=order_type,
                submitted_at=datetime.utcnow()
            )
            session.merge(record)
            session.commit()

    def update_order_fill(self, order_id: str, filled_qty: int, avg_fill_price: float, status: str):
        with self.SessionLocal() as session:
            order = session.query(OrderRecord).filter_by(order_id=order_id).first()
            if order:
                order.filled_quantity = filled_qty
                order.avg_fill_price = avg_fill_price
                order.status = status
                order.filled_at = datetime.utcnow()
                session.commit()
