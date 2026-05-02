from fastapi.testclient import TestClient

from app.main import app


def test_grid_plan_for_spot_paper() -> None:
    response = TestClient(app).post(
        "/grid/plan",
        json={
            "exchange": "paper",
            "symbol": "BTC/USDT",
            "market_type": "spot",
            "strategy": "neutral_range",
            "grid_count": 8,
            "investment_quote": 400,
        },
    )
    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["market_type"] == "spot"
    assert len(plan["levels"]) == 8
    assert plan["quote_per_grid"] == 50


def test_grid_start_for_swap_paper() -> None:
    response = TestClient(app).post(
        "/grid/start",
        json={
            "exchange": "paper",
            "symbol": "BTC/USDT",
            "market_type": "swap",
            "strategy": "trend_follow",
            "grid_count": 6,
            "investment_quote": 300,
            "rounds": 1,
            "interval_seconds": 0,
        },
    )
    assert response.status_code == 200
    bot_id = response.json()["bot_id"]
    status = TestClient(app).get(f"/grid/bots/{bot_id}")
    assert status.status_code == 200
    assert status.json()["bot"]["id"] == bot_id


def test_grid_ai_adaptive_falls_back_without_working_ai() -> None:
    response = TestClient(app).post(
        "/grid/plan",
        json={
            "exchange": "paper",
            "symbol": "BTC/USDT",
            "market_type": "spot",
            "strategy": "ai_adaptive",
            "grid_count": 5,
        },
    )
    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["strategy"] == "ai_adaptive"
    assert len(plan["levels"]) == 5


def test_grid_plan_geometric_spacing() -> None:
    response = TestClient(app).post(
        "/grid/plan",
        json={
            "exchange": "paper",
            "symbol": "BTC/USDT",
            "market_type": "spot",
            "strategy": "neutral_range",
            "spacing_mode": "geometric",
            "lower_price": 100,
            "upper_price": 400,
            "grid_count": 3,
            "investment_quote": 300,
        },
    )
    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["spacing_mode"] == "geometric"
    assert [round(level["price"], 2) for level in plan["levels"]] == [100, 200, 400]


def test_grid_request_accepts_summary_push_switch() -> None:
    response = TestClient(app).post(
        "/grid/start",
        json={
            "exchange": "paper",
            "symbol": "BTC/USDT",
            "market_type": "spot",
            "strategy": "neutral_range",
            "summary_push_enabled": True,
            "summary_interval_seconds": 7200,
            "rounds": 1,
            "interval_seconds": 0,
        },
    )
    assert response.status_code == 200


def test_paper_grid_uses_symbol_specific_price() -> None:
    client = TestClient(app)
    eth_response = client.post(
        "/grid/plan",
        json={"exchange": "paper", "symbol": "ETH/USDT", "strategy": "neutral_range", "grid_count": 5},
    )
    sol_response = client.post(
        "/grid/plan",
        json={"exchange": "paper", "symbol": "SOL/USDT", "strategy": "neutral_range", "grid_count": 5},
    )
    assert eth_response.status_code == 200
    assert sol_response.status_code == 200
    eth_price = eth_response.json()["plan"]["current_price"]
    sol_price = sol_response.json()["plan"]["current_price"]
    assert 2_000 < eth_price < 5_000
    assert 80 < sol_price < 250
    assert eth_response.json()["plan"]["symbol"] == "ETH/USDT"
    assert sol_response.json()["plan"]["symbol"] == "SOL/USDT"


def test_paper_grid_supports_sub_dollar_kas_price() -> None:
    response = TestClient(app).post(
        "/grid/plan",
        json={"exchange": "paper", "symbol": "KAS/USDT", "strategy": "neutral_range", "grid_count": 5},
    )
    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["symbol"] == "KAS/USDT"
    assert plan["price_source"] == "paper_estimate"
    assert 0.015 < plan["current_price"] < 0.08
    assert plan["upper_price"] < 0.1
