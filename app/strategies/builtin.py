from __future__ import annotations

from app.models import MarketSnapshot, StrategySignal, TradeAction
from app.strategies.base import Strategy
from app.strategies.indicators import atr, rsi, sma


class TrendMomentumStrategy(Strategy):
    name = "trend_momentum"

    def evaluate(self, snapshot: MarketSnapshot) -> StrategySignal:
        closes = [c.close for c in snapshot.candles]
        fast = sma(closes, 12)
        slow = sma(closes, 36)
        last = closes[-1]
        if fast > slow and last > fast:
            action = TradeAction.buy
            confidence = min(0.82, 0.55 + abs(fast - slow) / last * 12)
        elif fast < slow and last < fast:
            action = TradeAction.sell
            confidence = min(0.82, 0.55 + abs(fast - slow) / last * 12)
        else:
            action = TradeAction.hold
            confidence = 0.5
        return StrategySignal(strategy=self.name, action=action, confidence=confidence, reason=f"SMA12={fast:.4f}, SMA36={slow:.4f}")


class RSIMeanReversionStrategy(Strategy):
    name = "rsi_mean_reversion"

    def evaluate(self, snapshot: MarketSnapshot) -> StrategySignal:
        value = rsi(snapshot.candles)
        if value < 30:
            action = TradeAction.buy
            confidence = min(0.86, 0.6 + (30 - value) / 100)
        elif value > 70:
            action = TradeAction.sell
            confidence = min(0.86, 0.6 + (value - 70) / 100)
        else:
            action = TradeAction.hold
            confidence = 0.48
        return StrategySignal(strategy=self.name, action=action, confidence=confidence, reason=f"RSI14={value:.2f}")


class VolatilityBreakoutStrategy(Strategy):
    name = "volatility_breakout"

    def evaluate(self, snapshot: MarketSnapshot) -> StrategySignal:
        candles = snapshot.candles
        last = candles[-1]
        prior = candles[-25:-1]
        channel_high = max(c.high for c in prior)
        channel_low = min(c.low for c in prior)
        current_atr = atr(candles)
        if last.close > channel_high and current_atr > 0:
            action = TradeAction.buy
            confidence = 0.68
        elif last.close < channel_low and current_atr > 0:
            action = TradeAction.sell
            confidence = 0.68
        else:
            action = TradeAction.hold
            confidence = 0.45
        return StrategySignal(
            strategy=self.name,
            action=action,
            confidence=confidence,
            reason=f"close={last.close:.4f}, channel=({channel_low:.4f}, {channel_high:.4f}), ATR14={current_atr:.4f}",
        )


class StrategyEnsemble(Strategy):
    name = "strategy_ensemble"

    def __init__(self, strategies: list[Strategy]) -> None:
        self.strategies = strategies

    def evaluate(self, snapshot: MarketSnapshot) -> StrategySignal:
        signals = [strategy.evaluate(snapshot) for strategy in self.strategies]
        scores = {TradeAction.buy: 0.0, TradeAction.sell: 0.0, TradeAction.hold: 0.0}
        for signal in signals:
            scores[signal.action] += signal.confidence
        action = max(scores, key=scores.get)
        total = sum(scores.values()) or 1.0
        confidence = scores[action] / total
        return StrategySignal(
            strategy=self.name,
            action=action,
            confidence=confidence,
            reason="Weighted vote across trend momentum, RSI mean reversion, and volatility breakout.",
            metadata={"signals": [signal.model_dump() for signal in signals]},
        )


CORE_STRATEGIES: list[Strategy] = [
    TrendMomentumStrategy(),
    RSIMeanReversionStrategy(),
    VolatilityBreakoutStrategy(),
]

STRATEGIES: dict[str, Strategy] = {
    strategy.name: strategy for strategy in CORE_STRATEGIES
}
STRATEGIES["strategy_ensemble"] = StrategyEnsemble(CORE_STRATEGIES)
