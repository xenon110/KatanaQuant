"""
Anomaly & Operational Monitoring Agent:
Monitors feed health, agent divergence, and latency.
Has halt authority ONLY (can trigger the emergency kill switch), never trade authority.
"""
from typing import Dict, Any
from src.core.enums import AgentRole
from src.core.models import AnomalyOutput
from src.agents.base import BaseAgent


class AnomalyMonitoringAgent(BaseAgent):
    def __init__(self):
        super().__init__(role=AgentRole.ANOMALY_MONITOR, schema_class=AnomalyOutput)

    def build_prompt(self, context: Dict[str, Any]) -> str:
        metrics = context.get("metrics", {})
        return f"""You are an Anomaly & Operational Health Monitoring Agent.
System Metrics: {metrics}

Check for data feed degradation, extreme agent divergence, or error bursts.
Return JSON:
{{
  "is_anomalous": boolean,
  "severity": "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "anomalies_detected": ["list of issues"],
  "recommended_action": "CONTINUE" | "PAUSE_SYMBOL" | "HALT_SYSTEM"
}}
"""

    def get_mock_response(self, context: Dict[str, Any]) -> AnomalyOutput:
        staleness = context.get("metrics", {}).get("data_staleness_seconds", 0.0)
        if staleness > 120:
            return AnomalyOutput(
                is_anomalous=True,
                severity="CRITICAL",
                anomalies_detected=[f"Severe data feed outage: {staleness:.1f}s staleness."],
                recommended_action="HALT_SYSTEM"
            )
        return AnomalyOutput(
            is_anomalous=False,
            severity="NONE",
            anomalies_detected=[],
            recommended_action="CONTINUE"
        )
