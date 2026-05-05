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


def test_spot_execution_blocks_contract_intent(monkeypatch) -> None:
    async def fake_plan_trade(strategy_name, exchange_name, symbol, timeframe, market_type):
        return TradePlan(
            symbol=symbol or "BTC/USDT",
            action=TradeAction.sell,
            quote_size=50,
            confidence=0.9,
            reason="contract short",
            dry_run=False,
            execution_intent={"intent": "open_short", "position_side": "short", "requires_contract": True, "stop_loss": 110, "requires_hard_stop_loss": True},
        )

    monkeypatch.setattr("app.main.engine.plan_trade", fake_plan_trade)
    response = TestClient(app).post(
        "/run-once",
        json={"exchange": "paper", "symbol": "BTC/USDT", "market_type": "spot", "strategy": "strategy_ensemble", "execute": True},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "blocked"
    assert "spot mode" in result["detail"]["reason"]


def test_swap_execution_uses_contract_intent_on_paper(monkeypatch) -> None:
    async def fake_plan_trade(strategy_name, exchange_name, symbol, timeframe, market_type):
        return TradePlan(
            symbol=symbol or "BTC/USDT",
            action=TradeAction.sell,
            quote_size=50,
            confidence=0.9,
            reason="contract short",
            dry_run=False,
            execution_intent={
                "intent": "open_short",
                "position_side": "short",
                "requires_contract": True,
                "requires_hard_stop_loss": True,
                "stop_loss": 110,
                "take_profit": 90,
                "leverage": 3,
            },
        )

    monkeypatch.setattr("app.main.engine.plan_trade", fake_plan_trade)
    response = TestClient(app).post(
        "/run-once",
        json={"exchange": "paper", "symbol": "BTC/USDT", "market_type": "swap", "strategy": "strategy_ensemble", "execute": True},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "paper_filled"
    assert result["detail"]["contract_mode"] is True
    assert result["detail"]["execution_intent"]["intent"] == "open_short"


def test_contract_execution_requires_stop_loss(monkeypatch) -> None:
    async def fake_plan_trade(strategy_name, exchange_name, symbol, timeframe, market_type):
        return TradePlan(
            symbol=symbol or "BTC/USDT",
            action=TradeAction.buy,
            quote_size=50,
            confidence=0.9,
            reason="contract long",
            dry_run=False,
            execution_intent={"intent": "open_long", "position_side": "long", "requires_contract": True, "requires_hard_stop_loss": True},
        )

    monkeypatch.setattr("app.main.engine.plan_trade", fake_plan_trade)
    response = TestClient(app).post(
        "/run-once",
        json={"exchange": "paper", "symbol": "BTC/USDT", "market_type": "swap", "strategy": "strategy_ensemble", "execute": True},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "blocked"
    assert "hard stop_loss" in result["detail"]["reason"]


def test_contract_execution_blocks_invalid_stop_direction(monkeypatch) -> None:
    async def fake_plan_trade(strategy_name, exchange_name, symbol, timeframe, market_type):
        return TradePlan(
            symbol=symbol or "BTC/USDT",
            action=TradeAction.buy,
            quote_size=50,
            confidence=0.9,
            reason="bad stop",
            dry_run=False,
            execution_intent={
                "intent": "open_long",
                "position_side": "long",
                "requires_contract": True,
                "requires_hard_stop_loss": True,
                "reference_price": 100,
                "stop_loss": 101,
                "take_profit": 120,
                "leverage": 3,
                "max_leverage": 3,
            },
        )

    monkeypatch.setattr("app.main.engine.plan_trade", fake_plan_trade)
    response = TestClient(app).post(
        "/run-once",
        json={"exchange": "paper", "symbol": "BTC/USDT", "market_type": "swap", "strategy": "strategy_ensemble", "execute": True},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "blocked"
    assert "stop_loss" in result["detail"]["reason"]


def test_contract_execution_blocks_excessive_leverage(monkeypatch) -> None:
    async def fake_plan_trade(strategy_name, exchange_name, symbol, timeframe, market_type):
        return TradePlan(
            symbol=symbol or "BTC/USDT",
            action=TradeAction.sell,
            quote_size=50,
            confidence=0.9,
            reason="bad leverage",
            dry_run=False,
            execution_intent={
                "intent": "open_short",
                "position_side": "short",
                "requires_contract": True,
                "requires_hard_stop_loss": True,
                "reference_price": 100,
                "stop_loss": 110,
                "take_profit": 90,
                "leverage": 10,
                "max_leverage": 3,
            },
        )

    monkeypatch.setattr("app.main.engine.plan_trade", fake_plan_trade)
    response = TestClient(app).post(
        "/run-once",
        json={"exchange": "paper", "symbol": "BTC/USDT", "market_type": "swap", "strategy": "strategy_ensemble", "execute": True},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "blocked"
    assert "exceeds strategy max" in result["detail"]["reason"]
