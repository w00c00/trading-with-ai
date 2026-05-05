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


def test_config_page_has_ai_dropdowns() -> None:
    response = TestClient(app).get("/config")
    assert response.status_code == 200
    assert "cfg_ai_model_custom" in response.text
    assert "cfg_ai_base_url_custom" in response.text
    assert "MiniMax 中国区" in response.text
    assert "自定义 Base URL" in response.text


def test_dashboard_has_balance_refresh_controls() -> None:
    response = TestClient(app).get("/dashboard")
    assert response.status_code == 200
    assert "refreshBalancesButton" in response.text
    assert "balanceTable" in response.text
    assert "交易所可用余额" in response.text


def test_config_has_serverchan_test_button() -> None:
    response = TestClient(app).get("/config")
    assert response.status_code == 200
    assert "testServerChanButton" in response.text
    assert "serverChanTestStatus" in response.text
