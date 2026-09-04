"""
Orchestrator Pipeline.
Plain deterministic code that coordinates agent execution, enforces schemas,
synthesizes a single ProposedTrade, and routes to the Deterministic Risk Gate.
"""
import asyncio
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import logging

from src.core.models import (
    MarketBar,
    SignalProposal,
    MarketContextOutput,
    RiskReviewOutput,
    SizingProposal,
    AnomalyOutput,
    ProposedTrade,
    RiskGateDecision,
    AccountState,
    Position
)
from src.core.enums import SignalDirection, OrderSide, OrderType, TimeInForce
from src.strategy.base import BaseStrategy
from src.agents.signal_agent import SignalResearchAgent
from src.agents.market_context_agent import MarketContextAgent
from src.agents.risk_reviewer_agent import RiskReviewerAgent
from src.agents.portfolio_sizing_agent import PortfolioSizingAgent
from src.agents.anomaly_agent import AnomalyMonitoringAgent
from src.risk.gate import DeterministicRiskGate

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    def __init__(
        self,
        strategy: BaseStrategy,
        risk_gate: DeterministicRiskGate,
    ):
        self.strategy = strategy
        self.risk_gate = risk_gate
        self.signal_agent = SignalResearchAgent()
        self.market_context_agent = MarketContextAgent()
        self.risk_reviewer_agent = RiskReviewerAgent()
        self.portfolio_sizing_agent = PortfolioSizingAgent()
        self.anomaly_agent = AnomalyMonitoringAgent()

    async def process_bar(
        self,
        bar: MarketBar,
        history_df,
        account: AccountState,
        positions: List[Position],
        data_staleness_seconds: float = 0.0
    ) -> Optional[Tuple[ProposedTrade, RiskGateDecision]]:
        """
        Full pipeline cycle:
        1. Evaluate quantitative rule strategy.
        2. If signal generated, run multi-agent analysis layer concurrently.
        3. Merge agent outputs into a ProposedTrade.
        4. Pass ProposedTrade to the Deterministic Risk Gate.
        """
        # 1. Deterministic Rule Engine Trigger
        raw_signal: Optional[SignalProposal] = self.strategy.evaluate(history_df, bar)
        if not raw_signal or raw_signal.direction == SignalDirection.NEUTRAL:
            return None

        symbol = bar.symbol
        price = bar.close
        atr_val = raw_signal.key_levels.get("atr", price * 0.02)

        # 2. Concurrently invoke Multi-Agent Analysis Layer
        agent_context = {
            "symbol": symbol,
            "close": price,
            "atr": atr_val,
            "equity": account.equity,
            "raw_signal": raw_signal.model_dump(),
            "raw_direction": raw_signal.direction,
            "positions": [p.model_dump() for p in positions],
            "metrics": {"data_staleness_seconds": data_staleness_seconds}
        }

        # Run agents in parallel
        try:
            results = await asyncio.gather(
                self.signal_agent.execute(agent_context),
                self.market_context_agent.execute(agent_context),
                self.risk_reviewer_agent.execute(agent_context),
                self.portfolio_sizing_agent.execute(agent_context),
                self.anomaly_agent.execute(agent_context),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error during parallel agent execution: {e}")
            return None

        # Extract agent outputs with fallback on exception
        sig_out = results[0] if isinstance(results[0], SignalProposal) else self.signal_agent.get_mock_response(agent_context)
        mkt_out = results[1] if isinstance(results[1], MarketContextOutput) else self.market_context_agent.get_mock_response(agent_context)
        risk_out = results[2] if isinstance(results[2], RiskReviewOutput) else self.risk_reviewer_agent.get_mock_response(agent_context)
        size_out = results[3] if isinstance(results[3], SizingProposal) else self.portfolio_sizing_agent.get_mock_response(agent_context)
        anomaly_out = results[4] if isinstance(results[4], AnomalyOutput) else self.anomaly_agent.get_mock_response(agent_context)

        # If Anomaly Agent detected critical condition, trip the kill switch immediately
        if anomaly_out.recommended_action == "HALT_SYSTEM":
            self.risk_gate.trip_circuit_breaker(
                f"Anomaly agent halt: {', '.join(anomaly_out.anomalies_detected)}"
            )

        # 3. Synthesize ProposedTrade
        side = OrderSide.BUY if sig_out.direction == SignalDirection.BULLISH else OrderSide.SELL
        proposed_qty = size_out.suggested_shares

        # Fallback if sizing agent proposed 0
        if proposed_qty <= 0:
            proposed_qty = max(1, int((account.equity * 0.05) // price))

        proposed_trade = ProposedTrade(
            symbol=symbol,
            side=side,
            quantity=proposed_qty,
            price=price,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            signal_proposal=sig_out,
            market_context=mkt_out,
            risk_review=risk_out,
            sizing_proposal=size_out,
            metadata={"strategy": self.strategy.name}
        )

        # 4. Route ProposedTrade to Deterministic Risk Gate (NO AI in this step)
        decision = self.risk_gate.evaluate(
            proposed_trade=proposed_trade,
            account=account,
            current_positions=positions,
            is_day_trade=False, # Can be determined by position opening timestamp
            data_staleness_seconds=data_staleness_seconds
        )

        return proposed_trade, decision
