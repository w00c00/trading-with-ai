from abc import ABC, abstractmethod

from app.models import MarketSnapshot, StrategySignal


class Strategy(ABC):
    name: str

    @abstractmethod
    def evaluate(self, snapshot: MarketSnapshot) -> StrategySignal:
        raise NotImplementedError
