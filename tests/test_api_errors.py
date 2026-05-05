from fastapi.testclient import TestClient

from app.main import app


def test_run_once_errors_return_json(monkeypatch) -> None:
    async def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("app.main.engine.plan_trade", raise_runtime_error)
    response = TestClient(app).post("/run-once", json={"strategy": "strategy_ensemble"})
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert "RuntimeError: synthetic failure" in response.json()["detail"]


def test_balances_returns_paper_balances() -> None:
    response = TestClient(app).get("/balances", params={"exchange": "paper", "market_type": "spot"})
    assert response.status_code == 200
    data = response.json()
    assert data["exchange"] == "paper"
    assert data["available"] is True
    assert data["balances"]["USDT"] == 10000.0


def test_serverchan_test_requires_sendkey(monkeypatch) -> None:
    monkeypatch.setattr("app.main.settings.serverchan_sendkey", "", raising=False)
    response = TestClient(app).post("/notifications/serverchan/test", json={})
    assert response.status_code == 400
    assert "未配置方糖 SendKey" in response.json()["detail"]


def test_serverchan_test_can_use_typed_sendkey(monkeypatch) -> None:
    calls = []

    async def fake_send(self, title, desp):
        calls.append((self.sendkey, title, desp))
        return True

    monkeypatch.setattr("app.main.ServerChanNotifier.send", fake_send)
    response = TestClient(app).post("/notifications/serverchan/test", json={"sendkey": "SCT123"})
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert calls[0][0] == "SCT123"
