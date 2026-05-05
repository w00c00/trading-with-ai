from fastapi.testclient import TestClient

from app.main import app


def test_root_redirects_to_dashboard() -> None:
    response = TestClient(app, follow_redirects=False).get("/")
    assert response.status_code == 307
    assert response.headers["location"] == "/dashboard"


def test_module_pages_render() -> None:
    client = TestClient(app)
    for path in ["/dashboard", "/trade", "/batch", "/grid", "/strategy-center", "/config"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "page-section" in response.text
        assert "Trading with AI 控制台" in response.text


def test_pages_explain_strategy_run_flow() -> None:
    client = TestClient(app)
    trade = client.get("/trade").text
    assert "单策略运行" in trade
    assert "上传的新策略会出现在“策略”下拉菜单里" in trade
    assert "启动循环运行" in trade

    batch = client.get("/batch").text
    assert "多策略运行与历史回测" in batch
    assert "历史回测" in batch
    assert "启动多策略运行" in batch

    strategies = client.get("/strategy-center").text
    assert "这里负责导入和管理策略，不负责执行" in strategies


def test_config_page_has_ai_dropdowns() -> None:
    response = TestClient(app).get("/config")
    assert response.status_code == 200
    assert "cfg_ai_model_custom" in response.text
    assert "cfg_ai_base_url_custom" in response.text
    assert "MiniMax 中国区" in response.text
    assert "自定义 Base URL" in response.text
    assert "cfg_risk_limit_priority" in response.text
    assert "策略内部限制优先" in response.text


def test_dashboard_has_balance_refresh_controls() -> None:
    response = TestClient(app).get("/dashboard")
    assert response.status_code == 200
    assert "refreshBalancesButton" in response.text
    assert "balanceTable" in response.text
    assert "交易所可用余额" in response.text
    assert "loadLatestRun" in response.text
    assert "tradingWithAi.activeRunId" in response.text


def test_config_has_serverchan_test_button() -> None:
    response = TestClient(app).get("/config")
    assert response.status_code == 200
    assert "testServerChanButton" in response.text
    assert "serverChanTestStatus" in response.text
