"""
Deterministic Risk Gate.
Zero AI logic — strict, unit-tested, hard-coded rules to protect real money capital.
"""
from datetime import datetime
from typing import List, Optional
import logging

from src.core.models import (
    ProposedTrade,
    RiskGateDecision,
    AccountState,
    Position
)
from src.core.enums import (
    RiskGateStatus,
    OrderSide,
    AccountType
)
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class DeterministicRiskGate:
    def __init__(self, config: Settings):
        self.config = config
        self.is_circuit_breaker_tripped: bool = False
        self.circuit_breaker_reason: Optional[str] = None

    def trip_circuit_breaker(self, reason: str) -> None:
        """Manually or automatically engage the kill switch."""
        self.is_circuit_breaker_tripped = True
        self.circuit_breaker_reason = reason
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")

    def reset_circuit_breaker(self) -> None:
        """Reset the kill switch."""
        self.is_circuit_breaker_tripped = False
        self.circuit_breaker_reason = None
        logger.info("Circuit breaker has been reset.")

    def evaluate(
        self,
        proposed_trade: ProposedTrade,
        account: AccountState,
        current_positions: List[Position],
        is_day_trade: bool = False,
        data_staleness_seconds: float = 0.0
    ) -> RiskGateDecision:
        """
        Evaluate a proposed trade against all deterministic risk constraints.
        Returns a RiskGateDecision with approval/rejection and allowed quantity.
        """
        violations: List[str] = []
        reasons: List[str] = []

        # 0. Global Circuit Breaker / Trading Block Check
        if self.is_circuit_breaker_tripped:
            violations.append("CIRCUIT_BREAKER_ACTIVE")
            reasons.append(f"Global circuit breaker is active: {self.circuit_breaker_reason}")
            return RiskGateDecision(
                approved=False,
                status=RiskGateStatus.REJECTED,
                original_quantity=proposed_trade.quantity,
                allowed_quantity=0,
                rule_violations=violations,
                rejection_reasons=reasons
            )

        if account.is_trading_blocked:
            violations.append("BROKER_TRADING_BLOCKED")
            reasons.append("Account trading is currently blocked by broker.")
            return RiskGateDecision(
                approved=False,
                status=RiskGateStatus.REJECTED,
                original_quantity=proposed_trade.quantity,
                allowed_quantity=0,
                rule_violations=violations,
                rejection_reasons=reasons
            )

        # 1. Market Data Staleness Check
        if data_staleness_seconds > self.config.max_staleness_seconds:
            violations.append("DATA_FEED_STALE")
            reasons.append(
                f"Data feed latency ({data_staleness_seconds:.1f}s) exceeds threshold ({self.config.max_staleness_seconds}s)."
            )

        # 2. Daily Loss Limit Check
        # Loss is negative PnL. Check both absolute $ and % of starting equity.
        daily_loss = -(account.daily_realized_pnl + account.daily_unrealized_pnl)
        if daily_loss >= self.config.max_daily_loss_usd:
            violations.append("MAX_DAILY_LOSS_USD_BREACHED")
            reasons.append(f"Daily loss (${daily_loss:.2f}) exceeds max allowed (${self.config.max_daily_loss_usd:.2f}).")

        if account.starting_daily_equity > 0:
            daily_loss_pct = daily_loss / account.starting_daily_equity
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                violations.append("MAX_DAILY_LOSS_PCT_BREACHED")
                reasons.append(
                    f"Daily loss percentage ({daily_loss_pct*100:.2f}%) exceeds limit ({self.config.max_daily_loss_pct*100:.2f}%)."
                )

        # Check if closing an existing position (risk-reducing execution)
        pos = next((p for p in current_positions if p.symbol == proposed_trade.symbol), None)

        # 2a. Liquidating / Selling Long Position
        if proposed_trade.side == OrderSide.SELL:
            owned_shares = pos.quantity if (pos and pos.quantity > 0) else 0

            # If fully or partially selling an existing Long position:
            if owned_shares > 0:
                if proposed_trade.quantity <= owned_shares:
                    return RiskGateDecision(
                        approved=True,
                        status=RiskGateStatus.APPROVED,
                        original_quantity=proposed_trade.quantity,
                        allowed_quantity=proposed_trade.quantity
                    )
                else:
                    # In CASH account: cannot sell more than owned
                    if account.account_type == AccountType.CASH:
                        violations.append("INSUFFICIENT_SHARES_TO_SELL")
                        reasons.append(f"Attempting to sell {proposed_trade.quantity} shares, but only own {owned_shares}.")
                        return RiskGateDecision(
                            approved=False,
                            status=RiskGateStatus.REJECTED,
                            original_quantity=proposed_trade.quantity,
                            allowed_quantity=0,
                            rule_violations=violations,
                            rejection_reasons=reasons
                        )
            else:
                # Short selling when 0 long shares owned
                if account.account_type == AccountType.CASH:
                    violations.append("INSUFFICIENT_SHARES_TO_SELL")
                    reasons.append("Short selling is prohibited in Cash accounts (SEC Reg T).")
                    return RiskGateDecision(
                        approved=False,
                        status=RiskGateStatus.REJECTED,
                        original_quantity=proposed_trade.quantity,
                        allowed_quantity=0,
                        rule_violations=violations,
                        rejection_reasons=reasons
                    )

        # 2b. Covering / Buying to close Short Position
        if proposed_trade.side == OrderSide.BUY and pos and pos.quantity < 0:
            short_shares = abs(pos.quantity)
            if proposed_trade.quantity <= short_shares:
                # Buy-to-cover is risk-reducing and always allowed up to short quantity
                return RiskGateDecision(
                    approved=True,
                    status=RiskGateStatus.APPROVED,
                    original_quantity=proposed_trade.quantity,
                    allowed_quantity=proposed_trade.quantity
                )

        # --- NEW POSITION / POSITION EXPANSION CHECKS ---

        # 3. Maximum Concurrent Open Positions Check
        existing_symbols = {p.symbol for p in current_positions if p.quantity != 0}
        if proposed_trade.symbol not in existing_symbols:
            if len(existing_symbols) >= self.config.max_concurrent_positions:
                violations.append("MAX_CONCURRENT_POSITIONS_REACHED")
                reasons.append(
                    f"Open positions count ({len(existing_symbols)}) at limit ({self.config.max_concurrent_positions})."
                )

        # 4. Pattern Day Trader (PDT) Check
        # Rule applies if margin account and equity < $25,000
        if self.config.enforce_pdt and account.account_type == AccountType.MARGIN:
            if account.equity < 25000.0 and is_day_trade:
                if account.day_trade_count >= 3:
                    violations.append("PDT_LIMIT_REACHED")
                    reasons.append(
                        f"Day trade count ({account.day_trade_count}) reaches limit (3) for account equity under $25k."
                    )

        # 5. Settled Cash Check for Cash Accounts (Good Faith Violation Prevention)
        notional_order_val = proposed_trade.quantity * proposed_trade.price
        if account.account_type == AccountType.CASH:
            if notional_order_val > account.settled_cash:
                violations.append("INSUFFICIENT_SETTLED_CASH")
                reasons.append(
                    f"Order notional (${notional_order_val:.2f}) exceeds settled cash (${account.settled_cash:.2f})."
                )

        # 6. Live Buying Power Check
        if notional_order_val > account.buying_power:
            violations.append("INSUFFICIENT_BUYING_POWER")
            reasons.append(
                f"Order notional (${notional_order_val:.2f}) exceeds live buying power (${account.buying_power:.2f})."
            )

        # 7. Max Position Size Check & Quantity Sizing Adjustment
        # Cap 1: Max USD limit
        # Cap 2: Max % of NAV limit
        max_usd_from_pct = account.equity * self.config.max_position_size_pct_nav
        effective_max_usd = min(self.config.max_position_size_usd, max_usd_from_pct)
        
        existing_pos = next((p for p in current_positions if p.symbol == proposed_trade.symbol), None)
        existing_val = (existing_pos.quantity * proposed_trade.price) if existing_pos else 0.0
        available_position_capacity = max(0.0, effective_max_usd - existing_val)
        
        max_allowed_shares = int(available_position_capacity // proposed_trade.price) if proposed_trade.price > 0 else 0

        if max_allowed_shares == 0 and proposed_trade.quantity > 0:
            violations.append("POSITION_SIZE_CAP_EXCEEDED")
            reasons.append(
                f"Max position cap (${effective_max_usd:.2f}) already reached for symbol {proposed_trade.symbol}."
            )

        if violations:
            return RiskGateDecision(
                approved=False,
                status=RiskGateStatus.REJECTED,
                original_quantity=proposed_trade.quantity,
                allowed_quantity=0,
                rule_violations=violations,
                rejection_reasons=reasons
            )

        # If original quantity is within limits
        if proposed_trade.quantity <= max_allowed_shares:
            return RiskGateDecision(
                approved=True,
                status=RiskGateStatus.APPROVED,
                original_quantity=proposed_trade.quantity,
                allowed_quantity=proposed_trade.quantity
            )
        else:
            # Automatically trim quantity to max allowable
            return RiskGateDecision(
                approved=True,
                status=RiskGateStatus.MODIFIED,
                original_quantity=proposed_trade.quantity,
                allowed_quantity=max_allowed_shares,
                rejection_reasons=[f"Quantity scaled down from {proposed_trade.quantity} to {max_allowed_shares} to respect position cap."]
            )
