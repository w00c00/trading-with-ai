import asyncio

from app.config import Settings
from app.engine import TradingEngine
from app.models import AIDecision, MarketSnapshot, StrategySignal, TradeAction
from app.strategies import STRATEGIES


class FakeAI:
    async def decide(self, snapshot, signals):
        return AIDecision(action=TradeAction.buy, confidence=0.9, reason="confirmed")


class InternalLimitStrategy:
    name = "internal_limit_strategy"

    def evaluate(self, snapshot: MarketSnapshot) -> StrategySignal:
        return StrategySignal(
            strategy=self.name,
            action=TradeAction.buy,
            confidence=0.85,
            reason="internal limit",
            metadata={
                "nofx_risk_control": {
                    "min_position_size": 12,
                    "btc_eth_max_position_value_ratio": 10,
                    "min_confidence": 60,
                },
                "execution_intent": {},
            },
        )


def test_global_risk_priority_keeps_global_amount_and_limit(monkeypatch) -> None:
    monkeypatch.setitem(STRATEGIES, "internal_limit_strategy", InternalLimitStrategy())
    settings = Settings(
        default_exchange="paper",
        trade_quote_size=50,
        max_position_quote=1,
        min_ai_confidence=0.55,
        risk_limit_priority="global",
    )
    engine = TradingEngine(settings, FakeAI())

    plan = asyncio.run(engine.plan_trade(strategy_name="internal_limit_strategy", symbol="BTC/USDT"))

    assert plan.quote_size == 50
    assert plan.blocked is True
    assert "position limit would exceed 1.00" in plan.block_reason
    assert any("全局限制优先" in step for step in plan.decision_steps)


def test_strategy_risk_priority_uses_strategy_amount_and_limit(monkeypatch) -> None:
    monkeypatch.setitem(STRATEGIES, "internal_limit_strategy", InternalLimitStrategy())
    settings = Settings(
        default_exchange="paper",
        trade_quote_size=50,
        max_position_quote=1,
        min_ai_confidence=0.55,
        risk_limit_priority="strategy",
    )
    engine = TradingEngine(settings, FakeAI())

    plan = asyncio.run(engine.plan_trade(strategy_name="internal_limit_strategy", symbol="BTC/USDT"))

    assert plan.quote_size == 12
    assert plan.blocked is False
    assert any("策略内部限制优先" in step for step in plan.decision_steps)
    assert any("最大持仓=1000.00" in step for step in plan.decision_steps)
