import asyncio

import ccxt

from app.exchanges.ccxt_exchange import CCXTExchange
from app.exchanges.ccxt_exchange import _order_params_from_intent
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


def test_fetch_ohlcv_uses_public_client_before_configured_credentials() -> None:
    exchange = CCXTExchange("bybit", api_key="bad", secret="bad", sandbox=False)
    calls = {"public": 0, "private": 0}

    def private_fetch_ohlcv(*args):
        calls["private"] += 1
        raise ccxt.AuthenticationError('bybit {"retCode":10003,"retMsg":"API key is invalid."}')

    class PublicClient:
        def fetch_ohlcv(self, *args):
            calls["public"] += 1
            return [[1, 10, 12, 9, 11, 100]]

    exchange._client.fetch_ohlcv = private_fetch_ohlcv
    exchange._build_client = lambda with_credentials: PublicClient()
    exchange._public_client = PublicClient()
    candles = asyncio.run(exchange.fetch_ohlcv("BTC/USDT", "1h", 1))
    assert calls["public"] == 1
    assert calls["private"] == 0
    assert candles[0].close == 11


def test_fetch_ohlcv_retries_configured_client_after_public_error() -> None:
    exchange = CCXTExchange("bybit", api_key="key", secret="secret", sandbox=False)
    calls = {"private": 0}

    class PublicClient:
        def fetch_ohlcv(self, *args):
            raise ccxt.NetworkError("public timeout")

    def private_fetch_ohlcv(*args):
        calls["private"] += 1
        return [[1, 10, 12, 9, 11, 100]]

    exchange._public_client = PublicClient()
    exchange._client.fetch_ohlcv = private_fetch_ohlcv
    candles = asyncio.run(exchange.fetch_ohlcv("BTC/USDT", "1h", 1))
    assert calls["private"] == 1
    assert candles[0].close == 11


def test_fetch_ohlcv_without_credentials_reports_public_error() -> None:
    exchange = CCXTExchange("bybit", sandbox=False)

    class PublicClient:
        def fetch_ohlcv(self, *args):
            raise ccxt.NetworkError("public timeout")

    exchange._public_client = PublicClient()
    try:
        asyncio.run(exchange.fetch_ohlcv("BTC/USDT", "1h", 1))
    except RuntimeError as exc:
        assert "public candles" in str(exc)
        assert "public timeout" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_contract_intent_maps_to_ccxt_params() -> None:
    params = _order_params_from_intent(
        "bybit",
        {
            "intent": "close_short",
            "position_side": "short",
            "stop_loss": 110,
            "take_profit": 90,
        },
        "swap",
    )
    assert params["reduceOnly"] is True
    assert params["positionIdx"] == 2
    assert params["stopLossPrice"] == 110
    assert params["takeProfitPrice"] == 90


def test_contract_intent_maps_binance_position_side() -> None:
    params = _order_params_from_intent("binance", {"intent": "open_long", "position_side": "long"}, "future")
    assert params["positionSide"] == "LONG"
    assert "reduceOnly" not in params
