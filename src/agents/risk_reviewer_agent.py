"""
Risk Reviewer Agent:
Adversarially scrutinizes the proposed trade against existing portfolio holdings.
"""
from typing import Dict, Any
from src.core.enums import AgentRole
from src.core.models import RiskReviewOutput
from src.agents.base import BaseAgent


class RiskReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(role=AgentRole.RISK_REVIEWER, schema_class=RiskReviewOutput)

    def build_prompt(self, context: Dict[str, Any]) -> str:
        symbol = context.get("symbol", "UNKNOWN")
        positions = context.get("positions", [])
        return f"""You are an adversarial risk review agent.
A trade is proposed for {symbol}.
Current portfolio positions: {positions}

Identify potential correlation risks, sector crowding, or timing conflicts.
Return JSON:
{{
  "symbol": "{symbol}",
  "verdict": "APPROVE" or "CAUTION" or "REJECT",
  "concerns": ["list of concerns"],
  "correlation_warning": boolean,
  "reasoning": "summary"
}}
"""

    def get_mock_response(self, context: Dict[str, Any]) -> RiskReviewOutput:
        symbol = context.get("symbol", "SPY")
        return RiskReviewOutput(
            symbol=symbol,
            verdict="APPROVE",
            concerns=[],
            correlation_warning=False,
            reasoning=f"No adverse cross-asset correlation or sector concentration detected for {symbol}."
        )
