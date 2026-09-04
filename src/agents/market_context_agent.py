"""
Market Context Agent:
Analyzes macro backdrop, earnings proximity, and flags event risks.
"""
from typing import Dict, Any
from src.core.enums import AgentRole
from src.core.models import MarketContextOutput
from src.agents.base import BaseAgent


class MarketContextAgent(BaseAgent):
    def __init__(self):
        super().__init__(role=AgentRole.MARKET_CONTEXT, schema_class=MarketContextOutput)

    def build_prompt(self, context: Dict[str, Any]) -> str:
        symbol = context.get("symbol", "UNKNOWN")
        return f"""You are a Market Context and Event Risk Analyst.
Analyze macro environment and event risk for {symbol}.
Return JSON strictly adhering to:
{{
  "symbol": "{symbol}",
  "regime": "trending_bullish" | "ranging" | "high_volatility",
  "event_risk_detected": boolean,
  "flags": ["list", "of", "flags"],
  "reasoning": "brief summary"
}}
"""

    def get_mock_response(self, context: Dict[str, Any]) -> MarketContextOutput:
        symbol = context.get("symbol", "SPY")
        return MarketContextOutput(
            symbol=symbol,
            regime="trending_bullish",
            event_risk_detected=False,
            flags=["normal_macro_backdrop", "no_imminent_earnings_within_24h"],
            reasoning=f"Market regime is favorable for {symbol} with no major binary event risk."
        )
