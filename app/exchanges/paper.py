from __future__ import annotations

import random
import time
from itertools import count

from app.exchanges.base import ExchangeClient
from app.models import Candle, OrderResult, TradeAction


class PaperExchange(ExchangeClient):
    id = "paper"

    def __init__(self) -> None:
        self._orders = count(1)
        self._balances = {"USDT": 10_000.0, "BTC": 0.0}

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 120) -> list[Candle]:
        now = int(time.time() * 1000)
        step_ms = _timeframe_to_ms(timeframe)
        start = now - limit * step_ms
        return _generate_candles(symbol, start, step_ms, limit)

    async def fetch_ohlcv_since(self, symbol: str, timeframe: str, since: int, limit: int = 120) -> list[Candle]:
        step_ms = _timeframe_to_ms(timeframe)
        return _generate_candles(symbol, since, step_ms, limit)

    async def fetch_balances(self) -> dict[str, float]:
        return dict(self._balances)

    async def create_market_order(self, symbol: str, action: TradeAction, quote_size: float) -> OrderResult:
        base, quote = symbol.split("/")
        price = _base_price(symbol)
        amount = quote_size / price
        if action == TradeAction.buy:
            self._balances[quote] = self._balances.get(quote, 0.0) - quote_size
            self._balances[base] = self._balances.get(base, 0.0) + amount
        elif action == TradeAction.sell:
            self._balances[base] = self._balances.get(base, 0.0) - amount
            self._balances[quote] = self._balances.get(quote, 0.0) + quote_size
        return OrderResult(
            exchange=self.id,
            symbol=symbol,
            action=action,
            quote_size=quote_size,
            status="paper_filled",
            order_id=f"paper-{next(self._orders)}",
            detail={"estimated_price": price},
        )


def _generate_candles(symbol: str, start: int, step_ms: int, limit: int) -> list[Candle]:
    price = _base_price(symbol)
    candles: list[Candle] = []
    for index in range(limit):
        drift = 1 + random.uniform(-0.006, 0.007)
        open_price = price
        close = max(0.00000001, open_price * drift)
        high = max(open_price, close) * (1 + random.uniform(0, 0.003))
        low = min(open_price, close) * (1 - random.uniform(0, 0.003))
        volume = random.uniform(20, 180)
        timestamp = start + index * step_ms
        candles.append(Candle(timestamp=timestamp, open=open_price, high=high, low=low, close=close, volume=volume))
        price = close
    return candles


def _timeframe_to_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    amount = int(timeframe[:-1] or "1")
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 3_600_000
    if unit == "d":
        return amount * 86_400_000
    return 3_600_000


def _base_price(symbol: str) -> float:
    base = symbol.split("/")[0].upper()
    prices = {
        "BTC": 65_000.0,
        "ETH": 3_200.0,
        "SOL": 150.0,
        "BNB": 600.0,
        "XRP": 0.55,
        "DOGE": 0.12,
        "KAS": 0.033,
        "ADA": 0.45,
        "AVAX": 35.0,
        "LINK": 15.0,
        "MATIC": 0.75,
        "DOT": 7.0,
        "OP": 2.5,
        "ARB": 1.2,
    }
    return prices.get(base, 100.0)
