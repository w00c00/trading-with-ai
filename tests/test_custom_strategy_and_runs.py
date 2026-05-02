import time

from fastapi.testclient import TestClient

from app.main import app


def test_strategy_template_and_upload() -> None:
    client = TestClient(app)
    template = client.get("/strategy-template").json()["template"]
    template["name"] = "custom_test_rsi"
    response = client.post("/strategies/upload", json={"definition": template})
    assert response.status_code == 200
    assert response.json()["strategy"] == "custom_test_rsi"
    assert "custom_test_rsi" in client.get("/strategies").json()["strategies"]


def test_nofx_strategy_template_uploads_as_compatible_rule_strategy() -> None:
    client = TestClient(app)
    template = client.get("/strategy-template/nofx").json()["template"]
    template["strategyName"] = "nofx_test_strategy"
    response = client.post("/strategies/upload", json={"definition": template})
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "nofx_test_strategy"
    assert data["definition"]["source_format"] == "nofx"
    assert data["definition"]["rules"]


def test_nofx_strategy_with_chinese_or_short_name_gets_safe_internal_id() -> None:
    client = TestClient(app)
    template = client.get("/strategy-template/nofx").json()["template"]
    template["strategyName"] = "中文策略-1"
    response = client.post("/strategies/upload", json={"definition": template})
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "nofx_1"
    assert data["label"] == "NOFX RSI EMA 策略"

    template["strategyName"] = "a"
    template["displayName"] = "短名称 NOFX 策略"
    response = client.post("/strategies/upload", json={"definition": template})
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"].startswith("nofx_a_")
    assert data["label"] == "短名称 NOFX 策略"


def test_strategy_with_rules_and_chinese_name_gets_safe_internal_id() -> None:
    client = TestClient(app)
    template = client.get("/strategy-template").json()["template"]
    template["name"] = "中文规则策略"
    response = client.post("/strategies/upload", json={"definition": template})
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"].startswith("nofx_strategy_")
    assert data["label"] == "自定义 RSI 均线策略"

    template.pop("display_name")
    template["name"] = "中文策略无展示名"
    response = client.post("/strategies/upload", json={"definition": template})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "中文策略无展示名"


def test_exported_nofx_config_nested_under_config_uploads() -> None:
    client = TestClient(app)
    definition = {
        "name": "屌丝策略动态调节版本",
        "description": "NOFX 导出原版",
        "config": {
            "coin_source": {"source_type": "mixed", "static_coins": ["ETHUSDT"]},
            "indicators": {
                "enable_ema": True,
                "ema_periods": [20, 50],
                "enable_rsi": False,
                "enable_boll": True,
                "boll_periods": [20],
            },
            "risk_control": {"min_confidence": 70},
            "prompt_sections": {"role_definition": "角色定义"},
        },
        "version": "1.0",
    }
    response = client.post("/strategies/upload", json={"definition": definition})
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"].startswith("nofx_strategy_")
    assert data["label"] == "屌丝策略动态调节版本"
    assert data["definition"]["source_format"] == "nofx"
    assert data["definition"]["rules"]
    assert data["definition"]["nofx_prompt_sections"]["role_definition"] == "角色定义"


def test_nofx_coin_source_expands_symbols_for_run(monkeypatch) -> None:
    client = TestClient(app)

    async def fake_chat_json(messages, temperature=0.1):
        return {"symbols": ["SOL/USDT", "ETH/USDT"], "reason": "test selection"}

    monkeypatch.setattr("app.main.ai_client.chat_json", fake_chat_json)
    definition = {
        "name": "AI选币测试策略",
        "config": {
            "coin_source": {"source_type": "mixed", "static_coins": ["ETHUSDT"], "use_ai500": True, "ai500_limit": 2},
            "indicators": {"enable_ema": True, "ema_periods": [20, 50]},
            "risk_control": {"min_confidence": 70},
        },
    }
    upload = client.post("/strategies/upload", json={"definition": definition})
    assert upload.status_code == 200
    strategy = upload.json()["strategy"]
    response = client.post(
        "/batch-run",
        json={"exchange": "paper", "symbols": ["BTC/USDT"], "strategies": [strategy], "rounds": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["coin_selection"]["enabled"] is True
    assert "ETH/USDT" in data["symbols"]
    assert "SOL/USDT" in data["symbols"]


def test_start_run_dashboard_status() -> None:
    client = TestClient(app)
    response = client.post(
        "/runs/start",
        json={"exchange": "paper", "symbols": ["BTC/USDT"], "strategies": ["strategy_ensemble"], "rounds": 1},
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    status = client.get(f"/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["run"]["id"] == run_id
    deadline = time.time() + 3
    while time.time() < deadline:
        run = client.get(f"/runs/{run_id}").json()["run"]
        if run["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert client.get(f"/runs/{run_id}").json()["run"]["status"] in {"running", "completed", "failed"}
