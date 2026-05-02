from __future__ import annotations

import asyncio
import logging
from typing import Any

import ccxt

from app.exchanges.base import ExchangeClient
from app.models import Candle, OrderResult, TradeAction

logger = logging.getLogger(__name__)


class CCXTExchange(ExchangeClient):
    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        secret: str = "",
        password: str = "",
        sandbox: bool = True,
        market_type: str = "spot",
    ) -> None:
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Unsupported ccxt exchange: {exchange_id}")
        self.id = exchange_id
        self._exchange_id = exchange_id
        self._sandbox = sandbox
        self._market_type = market_type
        self._has_credentials = bool(api_key and secret)
        self._credential_config: dict[str, str] = {}
        if api_key:
            self._credential_config["apiKey"] = api_key
        if secret:
            self._credential_config["secret"] = secret
        if password:
            self._credential_config["password"] = password
        self._client = self._build_client(with_credentials=True)

    def _build_client(self, with_credentials: bool) -> Any:
        exchange_class = getattr(ccxt, self._exchange_id)
        config: dict[str, Any] = {"enableRateLimit": True, "options": {"defaultType": self._market_type}}
        if with_credentials:
            config.update(self._credential_config)
        client = exchange_class(config)
        if self._sandbox and hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)
        return client

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 120) -> list[Candle]:
        return await self.fetch_ohlcv_since(symbol, timeframe, None, limit)

    async def fetch_ohlcv_since(self, symbol: str, timeframe: str, since: int | None, limit: int = 120) -> list[Candle]:
        try:
            rows = await asyncio.to_thread(self._client.fetch_ohlcv, symbol, timeframe, since, limit)
        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as exc:
            logger.warning("%s candle fetch got credential error; retrying without credentials: %s", self.id, exc)
            public_client = self._build_client(with_credentials=False)
            try:
                rows = await asyncio.to_thread(public_client.fetch_ohlcv, symbol, timeframe, since, limit)
            except ccxt.BaseError as public_exc:
                raise RuntimeError(f"{self.id} failed to fetch {symbol} {timeframe} candles without credentials: {public_exc}") from public_exc
        except ccxt.BaseError as exc:
            raise RuntimeError(f"{self.id} failed to fetch {symbol} {timeframe} candles: {exc}") from exc
        return [
            Candle(timestamp=int(row[0]), open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]), volume=float(row[5]))
            for row in rows
        ]

    async def fetch_balances(self) -> dict[str, float]:
        if not self._has_credentials:
            logger.info("%s credentials are empty; balances are unavailable", self.id)
            return {}
        try:
            balance = await asyncio.to_thread(self._client.fetch_balance)
        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as exc:
            logger.warning("%s balance fetch skipped: %s", self.id, exc)
            return {}
        totals = balance.get("total", {})
        return {asset: float(amount) for asset, amount in totals.items() if amount}

    async def create_market_order(self, symbol: str, action: TradeAction, quote_size: float) -> OrderResult:
        if not self._has_credentials:
            return OrderResult(
                exchange=self.id,
                symbol=symbol,
                action=action,
                quote_size=quote_size,
                status="failed",
            detail={"reason": f"{self.id} requires API credentials before live orders can be submitted"},
            )
        try:
            ticker = await asyncio.to_thread(self._client.fetch_ticker, symbol)
            last = float(ticker["last"])
            amount = quote_size / last
        except ccxt.BaseError as exc:
            return OrderResult(
                exchange=self.id,
                symbol=symbol,
                action=action,
                quote_size=quote_size,
                status="failed",
                detail={"reason": f"{self.id} failed to price order: {exc}"},
            )
        try:
            order = await asyncio.to_thread(self._client.create_order, symbol, "market", action.value, amount)
        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as exc:
            return OrderResult(
                exchange=self.id,
                symbol=symbol,
                action=action,
                quote_size=quote_size,
                status="failed",
                detail={"reason": str(exc)},
            )
        return OrderResult(
            exchange=self.id,
            symbol=symbol,
            action=action,
            quote_size=quote_size,
            status=str(order.get("status", "submitted")),
            order_id=str(order.get("id")) if order.get("id") else None,
            detail=order,
        )

    async def set_leverage(self, symbol: str, leverage: float) -> None:
        if not self._has_credentials or not hasattr(self._client, "set_leverage"):
            return
        try:
            await asyncio.to_thread(self._client.set_leverage, leverage, symbol)
        except ccxt.BaseError as exc:
            logger.warning("%s set leverage skipped for %s: %s", self.id, symbol, exc)
