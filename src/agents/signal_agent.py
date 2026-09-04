"""
Signal / Research Agent:
Analyzes price action structure, candle momentum, and qualitative pattern confirmation.
"""
from typing import Dict, Any
from src.core.enums import AgentRole, SignalDirection
from src.core.models import SignalProposal
from src.agents.base import BaseAgent


class SignalResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(role=AgentRole.SIGNAL_RESEARCH, schema_class=SignalProposal)

    def build_prompt(self, context: Dict[str, Any]) -> str:
        symbol = context.get("symbol", "UNKNOWN")
        close_p = context.get("close", 0.0)
        indicators = context.get("indicators", {})
        raw_signal = context.get("raw_signal", {})

        return f"""You are a professional quantitative technical research agent.
Evaluate the current market setup for symbol {symbol}.
Current Close: {close_p}
Indicators: {indicators}
Raw Rule Engine Output: {raw_signal}

Your job: Assess whether the pattern shows genuine breakout momentum or false noise.
Return JSON conforming to this exact schema:
{{
  "symbol": "{symbol}",
  "direction": "BULLISH" or "BEARISH" or "NEUTRAL",
  "confidence": 0.0 to 1.0,
  "reasoning": "string explanation",
  "key_levels": {{"entry": float, "stop_loss": float, "take_profit": float}}
}}
"""

    def get_mock_response(self, context: Dict[str, Any]) -> SignalProposal:
        symbol = context.get("symbol", "SPY")
        close_p = context.get("close", 150.0)
        raw_dir = context.get("raw_direction", SignalDirection.BULLISH)
        
        return SignalProposal(
            symbol=symbol,
            direction=raw_dir,
            confidence=0.85,
            reasoning=f"Confirmed healthy candle momentum and volume participation for {symbol}.",
            key_levels={
                "entry": close_p,
                "stop_loss": round(close_p * 0.98, 2),
                "take_profit": round(close_p * 1.04, 2)
            }
        )
