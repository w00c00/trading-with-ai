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
