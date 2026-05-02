from app.models import Candle, MarketSnapshot, TradeAction
from app.strategies.builtin import TrendMomentumStrategy


def test_trend_momentum_buys_uptrend() -> None:
    candles = [
        Candle(timestamp=index, open=100 + index, high=101 + index, low=99 + index, close=100 + index, volume=1)
        for index in range(60)
    ]
    snapshot = MarketSnapshot(exchange="paper", symbol="BTC/USDT", timeframe="1h", candles=candles, last_price=candles[-1].close)
    signal = TrendMomentumStrategy().evaluate(snapshot)
    assert signal.action == TradeAction.buy
    assert signal.confidence > 0.55
