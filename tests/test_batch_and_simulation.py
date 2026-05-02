from fastapi.testclient import TestClient

from app.main import app
from app.models import TradeAction, TradePlan


def test_batch_run_multiple_symbols(monkeypatch) -> None:
    async def fake_plan_trade(strategy_name, exchange_name, symbol, timeframe, market_type="spot"):
        return TradePlan(
            symbol=symbol,
            action=TradeAction.hold,
            quote_size=50,
            confidence=0.5,
            reason=f"{strategy_name} fake",
            dry_run=True,
        )

    monkeypatch.setattr("app.main.engine.plan_trade", fake_plan_trade)
    response = TestClient(app).post(
        "/batch-run",
        json={
            "exchange": "paper",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "strategies": ["strategy_ensemble"],
            "rounds": 1,
        },
    )
    assert response.status_code == 200
    items = response.json()["rounds"][0]["items"]
    assert len(items) == 2
    assert {item["symbol"] for item in items} == {"BTC/USDT", "ETH/USDT"}


def test_simulate_returns_equity_curve() -> None:
    response = TestClient(app).post(
        "/simulate",
        json={
            "exchange": "paper",
            "symbols": ["BTC/USDT"],
            "strategies": ["trend_momentum"],
            "lookback": 80,
        },
    )
    assert response.status_code == 200
    simulation = response.json()["simulations"][0]
    assert simulation["symbol"] == "BTC/USDT"
    assert simulation["strategy"] == "trend_momentum"
    assert simulation["equity_curve"]
    assert simulation["candles"]
    assert simulation["start_time"] <= simulation["end_time"]


def test_simulate_filters_by_time_range() -> None:
    client = TestClient(app)
    baseline = client.post(
        "/simulate",
        json={"exchange": "paper", "symbols": ["BTC/USDT"], "strategies": ["trend_momentum"], "lookback": 120},
    ).json()["simulations"][0]
    candles = baseline["candles"]
    start_time = candles[-70]["timestamp"]
    end_time = candles[-20]["timestamp"]
    response = client.post(
        "/simulate",
        json={
            "exchange": "paper",
            "symbols": ["BTC/USDT"],
            "strategies": ["trend_momentum"],
            "lookback": 120,
            "start_time": start_time,
            "end_time": end_time,
        },
    )
    assert response.status_code == 200
    simulation = response.json()["simulations"][0]
    assert simulation["start_time"] >= start_time
    assert simulation["end_time"] <= end_time


def test_simulate_fetches_from_selected_start_time() -> None:
    start_time = 1735681020000
    end_time = start_time + 120 * 3_600_000
    response = TestClient(app).post(
        "/simulate",
        json={
            "exchange": "paper",
            "symbols": ["BTC/USDT"],
            "strategies": ["trend_momentum"],
            "timeframe": "1h",
            "lookback": 120,
            "start_time": start_time,
            "end_time": end_time,
        },
    )
    assert response.status_code == 200
    simulation = response.json()["simulations"][0]
    assert simulation["start_time"] == start_time
    assert simulation["end_time"] <= end_time


def test_uploaded_custom_strategy_can_be_simulated() -> None:
    client = TestClient(app)
    template = client.get("/strategy-template").json()["template"]
    template["name"] = "custom_sim_rsi"
    upload = client.post("/strategies/upload", json={"definition": template})
    assert upload.status_code == 200
    response = client.post(
        "/simulate",
        json={"exchange": "paper", "symbols": ["BTC/USDT"], "strategies": ["custom_sim_rsi"], "lookback": 80},
    )
    assert response.status_code == 200
    assert response.json()["simulations"][0]["strategy"] == "custom_sim_rsi"
