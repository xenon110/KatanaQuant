"""
Base interface and abstractions for deterministic quantitative strategies.
"""
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from src.core.models import MarketBar, SignalProposal
from src.core.enums import SignalDirection


class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, current_bar: MarketBar) -> Optional[SignalProposal]:
        """
        Evaluate market history and the current bar.
        Returns a SignalProposal if a trigger condition is met, else None.
        """
        pass
