from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Optional

from app.models import Candle, OrderResult, TradeAction


class ExchangeClient(ABC):
    id: str

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 120) -> list[Candle]:
        raise NotImplementedError

    async def fetch_ohlcv_since(self, symbol: str, timeframe: str, since: int, limit: int = 120) -> list[Candle]:
        return await self.fetch_ohlcv(symbol, timeframe, limit)

    @abstractmethod
    async def fetch_balances(self) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    async def create_market_order(self, symbol: str, action: TradeAction, quote_size: float) -> OrderResult:
        raise NotImplementedError

    async def create_plan_order(
        self,
        symbol: str,
        action: TradeAction,
        quote_size: float,
        execution_intent: Optional[dict[str, Any]] = None,
        market_type: str = "spot",
    ) -> OrderResult:
        return await self.create_market_order(symbol, action, quote_size)
