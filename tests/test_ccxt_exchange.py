import asyncio

import ccxt

from app.exchanges.ccxt_exchange import CCXTExchange
from app.models import TradeAction


def test_fetch_balances_without_credentials_returns_empty() -> None:
    exchange = CCXTExchange("bybit", sandbox=False)
    assert asyncio.run(exchange.fetch_balances()) == {}


def test_fetch_balances_authentication_error_returns_empty() -> None:
    exchange = CCXTExchange("bybit", api_key="key", secret="secret", sandbox=False)

    def raise_auth_error():
        raise ccxt.AuthenticationError('bybit requires "apiKey" credential')

    exchange._client.fetch_balance = raise_auth_error
    assert asyncio.run(exchange.fetch_balances()) == {}


def test_live_order_without_credentials_returns_failed_result() -> None:
    exchange = CCXTExchange("bybit", sandbox=False)
    result = asyncio.run(exchange.create_market_order("BTC/USDT", TradeAction.buy, 50))
    assert result.status == "failed"
    assert "requires API credentials" in result.detail["reason"]


def test_fetch_ohlcv_retries_without_credentials_after_auth_error() -> None:
    exchange = CCXTExchange("bybit", api_key="bad", secret="bad", sandbox=False)
    calls = {"public": 0}

    def raise_auth_error(*args):
        raise ccxt.AuthenticationError('bybit {"retCode":10003,"retMsg":"API key is invalid."}')

    class PublicClient:
        def fetch_ohlcv(self, *args):
            calls["public"] += 1
            return [[1, 10, 12, 9, 11, 100]]

    exchange._client.fetch_ohlcv = raise_auth_error
    exchange._build_client = lambda with_credentials: PublicClient()
    candles = asyncio.run(exchange.fetch_ohlcv("BTC/USDT", "1h", 1))
    assert calls["public"] == 1
    assert candles[0].close == 11
