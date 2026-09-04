"""
Portfolio Sizing Agent:
Recommends volatility-adjusted position sizing and risk-reward geometry.
Note: Deterministic Risk Gate retains final authority to cap/reduce this size.
"""
from typing import Dict, Any
from src.core.enums import AgentRole
from src.core.models import SizingProposal
from src.agents.base import BaseAgent


class PortfolioSizingAgent(BaseAgent):
    def __init__(self):
        super().__init__(role=AgentRole.PORTFOLIO_SIZING, schema_class=SizingProposal)

    def build_prompt(self, context: Dict[str, Any]) -> str:
        symbol = context.get("symbol", "UNKNOWN")
        equity = context.get("equity", 30000.0)
        price = context.get("price", 100.0)
        atr = context.get("atr", 2.0)

        return f"""You are a Portfolio Sizing & Volatility Specialist.
Recommend a position size for {symbol}.
Current Equity: ${equity:.2f}
Asset Price: ${price:.2f}
ATR Volatility: ${atr:.2f}

Target risk per trade: ~1.0% of portfolio equity (${equity * 0.01:.2f}).
Calculate suggested shares = Target Risk / (1.5 * ATR).
Return JSON:
{{
  "symbol": "{symbol}",
  "suggested_shares": integer,
  "suggested_notional_usd": float,
  "stop_loss_price": float,
  "take_profit_price": float,
  "reasoning": "summary"
}}
"""

    def get_mock_response(self, context: Dict[str, Any]) -> SizingProposal:
        symbol = context.get("symbol", "SPY")
        equity = context.get("equity", 30000.0)
        price = context.get("price", 150.0)
        atr = context.get("atr", 2.5)

        risk_amount = equity * 0.01 # 1% risk
        stop_dist = 1.5 * atr
        shares = int(risk_amount // stop_dist) if stop_dist > 0 else 10
        shares = max(1, shares)
        notional = round(shares * price, 2)

        return SizingProposal(
            symbol=symbol,
            suggested_shares=shares,
            suggested_notional_usd=notional,
            stop_loss_price=round(price - stop_dist, 2),
            take_profit_price=round(price + (3.0 * atr), 2),
            reasoning=f"Sized for 1% risk ($ {risk_amount:.2f}) with 1.5x ATR stop buffer."
        )
