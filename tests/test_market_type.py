from fastapi.testclient import TestClient

from app.main import app
from app.models import TradeAction, TradePlan


def test_run_once_accepts_market_type(monkeypatch) -> None:
    seen = {}

    async def fake_plan_trade(strategy_name, exchange_name, symbol, timeframe, market_type):
        seen["market_type"] = market_type
        return TradePlan(symbol=symbol or "BTC/USDT", action=TradeAction.hold, quote_size=50, confidence=0.5, reason="fake", dry_run=True)

    monkeypatch.setattr("app.main.engine.plan_trade", fake_plan_trade)
    response = TestClient(app).post("/run-once", json={"symbol": "BTC/USDT", "market_type": "swap", "strategy": "strategy_ensemble"})
    assert response.status_code == 200
    assert response.json()["market_type"] == "swap"
    assert seen["market_type"] == "swap"


def test_run_once_returns_ai_decision_details(monkeypatch) -> None:
    async def fake_plan_trade(strategy_name, exchange_name, symbol, timeframe, market_type):
        return TradePlan(
            symbol=symbol or "BTC/USDT",
            action=TradeAction.hold,
            quote_size=50,
            confidence=0.5,
            reason="fake",
            dry_run=True,
            strategy_signal={"action": "buy", "confidence": 0.7, "reason": "strategy"},
            ai_decision={"action": "hold", "confidence": 0.5, "reason": "risk", "risk_notes": ["test risk"]},
            decision_steps=["策略信号：buy", "AI 复核：hold"],
        )

    monkeypatch.setattr("app.main.engine.plan_trade", fake_plan_trade)
    response = TestClient(app).post("/run-once", json={"symbol": "BTC/USDT", "strategy": "strategy_ensemble"})
    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["strategy_signal"]["action"] == "buy"
    assert plan["ai_decision"]["risk_notes"] == ["test risk"]
    assert plan["decision_steps"]


def test_batch_run_accepts_market_type(monkeypatch) -> None:
    seen = []

    async def fake_plan_trade(strategy_name, exchange_name, symbol, timeframe, market_type):
        seen.append(market_type)
        return TradePlan(symbol=symbol, action=TradeAction.hold, quote_size=50, confidence=0.5, reason="fake", dry_run=True)

    monkeypatch.setattr("app.main.engine.plan_trade", fake_plan_trade)
    response = TestClient(app).post(
        "/batch-run",
        json={"exchange": "paper", "symbols": ["BTC/USDT"], "strategies": ["strategy_ensemble"], "market_type": "future"},
    )
    assert response.status_code == 200
    assert response.json()["rounds"][0]["items"][0]["market_type"] == "future"
    assert seen == ["future"]
