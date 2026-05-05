from __future__ import annotations

import logging
import asyncio
import json
import time
import uuid
import re
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.ai import AIDecisionMaker, AIClient
from app.config import get_settings
from app.engine import TradingEngine
from app.grid import GRID_STRATEGIES, GridPlanRequest, build_grid_plan
from app.notifications import ServerChanNotifier
from app.secure_config import load_encrypted_config, merge_config_update, public_config_snapshot, save_encrypted_config
from app.strategies import STRATEGIES
from app.strategies.custom import nofx_strategy_template, save_custom_strategy, strategy_template

logger = logging.getLogger(__name__)

settings = get_settings()
ai_client = AIClient(settings)
engine = TradingEngine(settings, AIDecisionMaker(ai_client))

app = FastAPI(title=settings.app_name, version="0.1.0")
RUNS: dict[str, dict[str, Any]] = {}
GRID_BOTS: dict[str, dict[str, Any]] = {}
MAX_SIMULATION_CANDLES = 10_000
BUILTIN_STRATEGY_LABELS = {
    "trend_momentum": "趋势动量策略",
    "rsi_mean_reversion": "RSI 均值回归策略",
    "volatility_breakout": "波动突破策略",
    "strategy_ensemble": "综合投票策略",
}
DEFAULT_AI_COIN_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "DOT/USDT",
    "OP/USDT",
    "ARB/USDT",
    "KAS/USDT",
]
OI_TOP_FALLBACK = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "BNB/USDT", "LINK/USDT", "AVAX/USDT", "ADA/USDT", "OP/USDT"]


class RunRequest(BaseModel):
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    market_type: str = "spot"
    timeframe: Optional[str] = None
    strategy: str = "strategy_ensemble"
    execute: bool = False


class BatchRunRequest(BaseModel):
    exchange: Optional[str] = None
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT"])
    market_type: str = "spot"
    timeframe: Optional[str] = None
    strategies: list[str] = Field(default_factory=lambda: ["strategy_ensemble"])
    execute: bool = False
    rounds: int = Field(default=1, ge=1, le=10_000)
    duration_seconds: int = Field(default=0, ge=0, le=2_592_000)
    interval_seconds: float = Field(default=0.0, ge=0, le=86_400)
    until_stopped: bool = False


class SimulateRequest(BaseModel):
    exchange: Optional[str] = None
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT"])
    market_type: str = "spot"
    timeframe: Optional[str] = None
    strategies: list[str] = Field(default_factory=lambda: ["strategy_ensemble"])
    initial_quote: float = Field(default=1000.0, gt=0)
    trade_quote_size: Optional[float] = Field(default=None, gt=0)
    lookback: int = Field(default=120, ge=60, le=MAX_SIMULATION_CANDLES)
    start_time: Optional[int] = Field(default=None, ge=0)
    end_time: Optional[int] = Field(default=None, ge=0)


class StrategyUploadRequest(BaseModel):
    definition: dict[str, Any]


class SettingsUpdate(BaseModel):
    live_trading_enabled: Optional[bool] = None
    default_exchange: Optional[str] = None
    default_symbol: Optional[str] = None
    default_timeframe: Optional[str] = None
    trade_quote_size: Optional[float] = None
    max_position_quote: Optional[float] = None
    risk_limit_priority: Optional[Literal["global", "strategy"]] = None
    min_ai_confidence: Optional[float] = None
    exchange_id: Optional[str] = None
    exchange_api_key: Optional[str] = None
    exchange_secret: Optional[str] = None
    exchange_password: Optional[str] = None
    exchange_sandbox: Optional[bool] = None
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None
    ai_api_key: Optional[str] = None
    ai_base_url: Optional[str] = None
    serverchan_sendkey: Optional[str] = None
    notify_trade_success: Optional[bool] = None
    notify_dry_run: Optional[bool] = None


class ServerChanTestRequest(BaseModel):
    sendkey: Optional[str] = None


def _reload_runtime() -> None:
    global settings, ai_client, engine
    get_settings.cache_clear()
    settings = get_settings()
    ai_client = AIClient(settings)
    engine = TradingEngine(settings, AIDecisionMaker(ai_client))


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page() -> str:
    return _dashboard_html("dashboard")


@app.get("/trade", response_class=HTMLResponse)
async def trade_page() -> str:
    return _dashboard_html("trade")


@app.get("/batch", response_class=HTMLResponse)
async def batch_page() -> str:
    return _dashboard_html("batch")


@app.get("/grid", response_class=HTMLResponse)
async def grid_page() -> str:
    return _dashboard_html("grid")


@app.get("/strategy-center", response_class=HTMLResponse)
async def strategy_center_page() -> str:
    return _dashboard_html("strategies")


@app.get("/config", response_class=HTMLResponse)
async def config_page() -> str:
    return _dashboard_html("config")


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/providers")
async def providers() -> dict[str, list[str]]:
    return {"providers": ["openrouter", "openai", "minimax", "mimo", "custom_openai_compatible"]}


@app.get("/exchanges")
async def exchanges() -> dict[str, list[str]]:
    return {"exchanges": ["paper", "binance", "bybit", "okx", "bitget", "kucoin", "gateio", "hyperliquid"]}


@app.get("/strategies")
async def strategies() -> dict[str, Any]:
    return {"strategies": list(STRATEGIES), "labels": _strategy_labels()}


@app.get("/grid/strategies")
async def grid_strategies() -> dict[str, Any]:
    return {"strategies": list(GRID_STRATEGIES.values())}


@app.get("/strategy-template")
async def get_strategy_template() -> dict[str, Any]:
    return {"template": strategy_template()}


@app.get("/strategy-template/nofx")
async def get_nofx_strategy_template() -> dict[str, Any]:
    return {"template": nofx_strategy_template()}


@app.post("/strategies/upload")
async def upload_strategy(request: StrategyUploadRequest) -> dict[str, Any]:
    try:
        strategy = save_custom_strategy(request.definition)
        STRATEGIES[strategy.name] = strategy
        return {
            "strategy": strategy.name,
            "label": _strategy_label(strategy.name),
            "definition": strategy.definition,
            "strategies": list(STRATEGIES),
            "labels": _strategy_labels(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/settings")
async def read_settings() -> dict[str, Any]:
    return {"settings": public_config_snapshot(settings.model_dump(mode="json"))}


@app.post("/settings")
async def write_settings(update: SettingsUpdate) -> dict[str, Any]:
    values = update.model_dump(exclude_unset=True)
    current = load_encrypted_config()
    merged = merge_config_update(current, values)
    save_encrypted_config(merged)
    _reload_runtime()
    return {"settings": public_config_snapshot(settings.model_dump(mode="json"))}


@app.get("/balances")
async def balances(exchange: Optional[str] = None, market_type: str = "spot") -> dict[str, Any]:
    selected_exchange = exchange or settings.default_exchange
    try:
        client = engine.build_exchange(selected_exchange, market_type=market_type)
        raw_balances = await client.fetch_balances()
    except Exception as exc:
        logger.exception("balance refresh failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    assets = [
        {"asset": asset, "amount": amount}
        for asset, amount in sorted(raw_balances.items(), key=lambda item: item[0])
    ]
    available = bool(assets) or client.id == "paper"
    message = "余额已刷新" if available else "余额不可用：请检查 API Key、Secret、账户权限或交易所类型。"
    return {
        "exchange": client.id,
        "market_type": market_type,
        "available": available,
        "message": message,
        "updated_at": time.time(),
        "balances": raw_balances,
        "assets": assets,
    }


@app.post("/notifications/serverchan/test")
async def test_serverchan(request: Optional[ServerChanTestRequest] = None) -> dict[str, Any]:
    sendkey = ((request.sendkey if request else None) or settings.serverchan_sendkey or "").strip()
    if not sendkey:
        raise HTTPException(status_code=400, detail="未配置方糖 SendKey")
    notifier = ServerChanNotifier(settings)
    notifier.sendkey = sendkey
    try:
        sent = await notifier.send(
            "Trading with AI 方糖测试",
            f"这是一条方糖可用性测试消息。\n\n发送时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        )
    except Exception as exc:
        logger.exception("serverchan test failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    if not sent:
        raise HTTPException(status_code=400, detail="方糖推送未发送，请检查 SendKey")
    return {"ok": True, "message": "方糖测试推送已发送"}


@app.post("/run-once")
async def run_once(request: RunRequest) -> dict:
    if request.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
    try:
        plan = await engine.plan_trade(
            strategy_name=request.strategy,
            exchange_name=request.exchange,
            symbol=request.symbol,
            timeframe=request.timeframe,
            market_type=request.market_type,
        )
        result = None
        if request.execute:
            result = await engine.execute_plan(plan, exchange_name=request.exchange, market_type=request.market_type)
        return {"market_type": request.market_type, "plan": plan.model_dump(mode="json"), "result": result.model_dump(mode="json") if result else None}
    except Exception as exc:
        logger.exception("run-once failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/batch-run")
async def batch_run(request: BatchRunRequest) -> dict:
    _validate_strategies(request.strategies)
    started_at = time.time()
    interval = _run_interval(request)
    rounds = []
    try:
        symbols, coin_selection = await _resolve_run_symbols(request.symbols, request.strategies, request.market_type)
        for round_index in range(request.rounds):
            tasks = [
                _run_symbol_strategy(request.exchange, symbol, request.timeframe, strategy, request.execute, request.market_type)
                for symbol in symbols
                for strategy in request.strategies
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            rounds.append(
                {
                    "round": round_index + 1,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                    "items": [_serialize_batch_item(item) for item in results],
                }
            )
            if interval and round_index < request.rounds - 1:
                await asyncio.sleep(interval)
        return {"rounds": rounds, "elapsed_seconds": round(time.time() - started_at, 3), "symbols": symbols, "coin_selection": coin_selection}
    except Exception as exc:
        logger.exception("batch-run failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/runs/start")
async def start_run(request: BatchRunRequest) -> dict:
    _validate_strategies(request.strategies)
    symbols, coin_selection = await _resolve_run_symbols(request.symbols, request.strategies, request.market_type)
    run_id = uuid.uuid4().hex[:12]
    request_payload = request.model_dump(mode="json")
    request_payload["symbols"] = symbols
    RUNS[run_id] = {
        "id": run_id,
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "request": request_payload,
        "coin_selection": coin_selection,
        "round": 0,
        "events": [],
        "items": [],
        "error": None,
        "stop_requested": False,
    }
    asyncio.create_task(_run_dashboard_job(run_id, request, symbols))
    return {"run_id": run_id, "run": _public_run(RUNS[run_id])}


@app.post("/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") == "running":
        run["stop_requested"] = True
        run["status"] = "stopping"
    return {"run": _public_run(run)}


@app.get("/runs/{run_id}")
async def read_run(run_id: str) -> dict:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run": _public_run(run)}


@app.post("/simulate")
async def simulate(request: SimulateRequest) -> dict:
    _validate_strategies(request.strategies)
    try:
        exchange = engine.build_exchange(request.exchange, market_type=request.market_type)
        timeframe = request.timeframe or settings.default_timeframe
        trade_size = request.trade_quote_size or settings.trade_quote_size
        simulations = []
        if request.start_time is not None and request.end_time is not None and request.end_time <= request.start_time:
            raise ValueError("end_time must be greater than start_time")
        for symbol in request.symbols:
            candles = await _fetch_simulation_candles(
                exchange,
                symbol,
                timeframe,
                request.start_time,
                request.end_time,
                request.lookback,
            )
            if len(candles) < 40:
                raise ValueError(f"Not enough candles for {symbol} in selected time range; got {len(candles)}")
            for strategy_name in request.strategies:
                simulations.append(_simulate_strategy(symbol, timeframe, strategy_name, candles, request.initial_quote, trade_size))
        return {"simulations": simulations}
    except Exception as exc:
        logger.exception("simulate failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/grid/plan")
async def grid_plan(request: GridPlanRequest) -> dict[str, Any]:
    try:
        exchange = engine.build_exchange(request.exchange, market_type=request.market_type)
        candles = await exchange.fetch_ohlcv(request.symbol, request.timeframe, limit=120)
        plan = await build_grid_plan(request, candles, ai_client)
        return {"plan": plan.model_dump(mode="json")}
    except Exception as exc:
        logger.exception("grid-plan failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/grid/start")
async def start_grid_bot(request: GridPlanRequest) -> dict[str, Any]:
    bot_id = uuid.uuid4().hex[:12]
    GRID_BOTS[bot_id] = {
        "id": bot_id,
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "request": request.model_dump(mode="json"),
        "plan": None,
        "round": 0,
        "events": [],
        "error": None,
        "last_summary_at": time.time(),
        "summary_count": 0,
    }
    asyncio.create_task(_run_grid_bot(bot_id, request))
    return {"bot_id": bot_id, "bot": _public_grid_bot(GRID_BOTS[bot_id])}


@app.get("/grid/bots/{bot_id}")
async def read_grid_bot(bot_id: str) -> dict[str, Any]:
    bot = GRID_BOTS.get(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Grid bot not found")
    return {"bot": _public_grid_bot(bot)}


async def _run_symbol_strategy(exchange: Optional[str], symbol: str, timeframe: Optional[str], strategy: str, execute: bool, market_type: str = "spot") -> dict:
    plan = await engine.plan_trade(strategy_name=strategy, exchange_name=exchange, symbol=symbol, timeframe=timeframe, market_type=market_type)
    result = await engine.execute_plan(plan, exchange_name=exchange, market_type=market_type) if execute else None
    return {
        "symbol": symbol,
        "market_type": market_type,
        "strategy": strategy,
        "strategy_label": _strategy_label(strategy),
        "plan": plan.model_dump(mode="json"),
        "result": result.model_dump(mode="json") if result else None,
    }


async def _run_dashboard_job(run_id: str, request: BatchRunRequest, symbols: list[str]) -> None:
    run = RUNS[run_id]
    started_at = time.time()
    interval = _run_interval(request)
    deadline = started_at + request.duration_seconds if request.until_stopped and request.duration_seconds else None
    try:
        round_index = 0
        while True:
            if run.get("stop_requested"):
                run["status"] = "stopped"
                break
            if not request.until_stopped and round_index >= request.rounds:
                run["status"] = "completed"
                break
            if deadline and time.time() >= deadline:
                run["status"] = "completed"
                break
            run["round"] = round_index + 1
            tasks = [
                _run_symbol_strategy(request.exchange, symbol, request.timeframe, strategy, request.execute, request.market_type)
                for symbol in symbols
                for strategy in request.strategies
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                item = _serialize_batch_item(result)
                item["round"] = round_index + 1
                item["elapsed_seconds"] = round(time.time() - started_at, 3)
                run["items"].append(item)
                event = _event_from_item(item)
                if event:
                    run["events"].append(event)
            round_index += 1
            should_continue = request.until_stopped or round_index < request.rounds
            if interval and should_continue:
                await asyncio.sleep(interval)
    except Exception as exc:
        logger.exception("dashboard run failed")
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        run["finished_at"] = time.time()


def _event_from_item(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    if item.get("error"):
        return {"type": "error", "message": item["error"], "timestamp": time.time()}
    plan = item.get("plan") or {}
    action = plan.get("action")
    if action not in {"buy", "sell"}:
        return None
    return {
        "type": "open" if action == "buy" else "close",
        "action": action,
        "symbol": item.get("symbol"),
        "strategy": item.get("strategy"),
        "strategy_label": item.get("strategy_label") or _strategy_label(str(item.get("strategy", ""))),
        "round": item.get("round"),
        "confidence": plan.get("confidence"),
        "status": (item.get("result") or {}).get("status", "planned"),
        "reason": plan.get("reason"),
        "timestamp": time.time(),
    }


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    public = dict(run)
    public["elapsed_seconds"] = round(time.time() - float(run["started_at"]), 3)
    public["events"] = list(run["events"])[-200:]
    public["items"] = list(run["items"])[-200:]
    return public


async def _resolve_run_symbols(symbols: list[str], strategies: list[str], market_type: str) -> tuple[list[str], dict[str, Any]]:
    selected: list[str] = []
    details: list[dict[str, Any]] = []
    for strategy_name in strategies:
        coin_source = _nofx_coin_source(strategy_name)
        if not coin_source:
            continue
        strategy_symbols, detail = await _symbols_from_nofx_coin_source(strategy_name, coin_source, market_type)
        selected.extend(strategy_symbols)
        details.append(detail)
    if not selected:
        selected = symbols
    selected = _unique_symbols(_normalize_symbol(symbol) for symbol in selected if symbol)
    if not selected:
        selected = [settings.default_symbol]
    return selected, {"enabled": bool(details), "symbols": selected, "details": details}


def _nofx_coin_source(strategy_name: str) -> Optional[dict[str, Any]]:
    strategy = STRATEGIES.get(strategy_name)
    definition = getattr(strategy, "definition", {}) if strategy else {}
    if not isinstance(definition, dict):
        return None
    nofx_config = definition.get("nofx_config") or {}
    if not isinstance(nofx_config, dict):
        return None
    config = nofx_config.get("config") if isinstance(nofx_config.get("config"), dict) else nofx_config
    coin_source = _get_any(config, "coin_source", "coinSource", "coin_config", "coinConfig")
    return coin_source if isinstance(coin_source, dict) else None


async def _symbols_from_nofx_coin_source(strategy_name: str, coin_source: dict[str, Any], market_type: str) -> tuple[list[str], dict[str, Any]]:
    static_symbols = [_normalize_symbol(symbol) for symbol in _get_any(coin_source, "static_coins", "staticCoins", "symbols") or []]
    selected = list(static_symbols)
    notes = []
    ai_limit = int(_get_any(coin_source, "ai500_limit", "ai500Limit") or 0)
    if _boolish(_get_any(coin_source, "use_ai500", "useAI500"), default=False):
        ai_symbols, ai_note = await _ai_select_symbols(strategy_name, ai_limit or 5, market_type)
        selected.extend(ai_symbols)
        notes.append(ai_note)
    oi_limit = int(_get_any(coin_source, "oi_top_limit", "oiTopLimit") or 0)
    if _boolish(_get_any(coin_source, "use_oi_top", "useOITop"), default=False):
        oi_symbols = OI_TOP_FALLBACK[: max(1, oi_limit or 10)]
        selected.extend(oi_symbols)
        notes.append("OI Top 暂用内置高流动性候选池；后续可接交易所 OI 排名源。")
    selected = _unique_symbols(selected)
    return selected, {
        "strategy": strategy_name,
        "strategy_label": _strategy_label(strategy_name),
        "source_type": _get_any(coin_source, "source_type", "sourceType") or "mixed",
        "static_symbols": static_symbols,
        "selected_symbols": selected,
        "notes": notes,
    }


async def _ai_select_symbols(strategy_name: str, limit: int, market_type: str) -> tuple[list[str], str]:
    limit = min(max(limit, 1), 20)
    messages = [
        {
            "role": "system",
            "content": "你是保守的加密货币选币助手。只返回 JSON：symbols 字符串数组，reason 字符串。symbols 必须来自用户给定候选池。",
        },
        {
            "role": "user",
            "content": (
                f"策略={_strategy_label(strategy_name)}，市场类型={market_type}，最多选择 {limit} 个。"
                f"候选池={DEFAULT_AI_COIN_UNIVERSE}。请优先选择流动性好、波动适中、适合策略运行的 USDT 交易对。"
            ),
        },
    ]
    try:
        data = await ai_client.chat_json(messages, temperature=0.2)
        raw_symbols = data.get("symbols") or data.get("coins") or []
        if isinstance(raw_symbols, str):
            raw_symbols = [raw_symbols]
        symbols = _unique_symbols(_normalize_symbol(symbol) for symbol in raw_symbols)[:limit]
        symbols = [symbol for symbol in symbols if symbol in DEFAULT_AI_COIN_UNIVERSE]
        if symbols:
            return symbols, f"AI 选币：{data.get('reason', '已根据候选池选择。')}"
        return DEFAULT_AI_COIN_UNIVERSE[:limit], "AI 未返回有效 symbols，已使用内置高流动性候选池。"
    except Exception as exc:
        logger.warning("AI coin selection failed: %s", exc)
        return DEFAULT_AI_COIN_UNIVERSE[:limit], f"AI 选币失败，已降级使用内置候选池：{type(exc).__name__}: {exc}"


def _normalize_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip().upper().replace("-", "/").replace("_", "/")
    if "/" in text:
        base, quote = text.split("/", 1)
        return f"{base}/{quote or 'USDT'}"
    match = re.fullmatch(r"([A-Z0-9]+)(USDT|USDC|USD|BTC|ETH)", text)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return f"{text}/USDT" if text else ""


def _unique_symbols(symbols: Any) -> list[str]:
    seen = set()
    result = []
    for symbol in symbols:
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return result


def _get_any(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    lowered = {str(key).lower(): value for key, value in source.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}


async def _run_grid_bot(bot_id: str, request: GridPlanRequest) -> None:
    bot = GRID_BOTS[bot_id]
    try:
        exchange = engine.build_exchange(request.exchange, market_type=request.market_type)
        candles = await exchange.fetch_ohlcv(request.symbol, request.timeframe, limit=120)
        plan = await build_grid_plan(request, candles, ai_client)
        bot["plan"] = plan.model_dump(mode="json")
        if request.market_type != "spot" and hasattr(exchange, "set_leverage"):
            await exchange.set_leverage(request.symbol, request.leverage)
        last_price = plan.current_price
        for round_index in range(request.rounds):
            bot["round"] = round_index + 1
            candles = await exchange.fetch_ohlcv(request.symbol, request.timeframe, limit=3)
            price = candles[-1].close
            for level in plan.levels:
                if level.triggered:
                    continue
                crossed_down = last_price > level.price >= price
                crossed_up = last_price < level.price <= price
                if (level.action.value == "buy" and crossed_down) or (level.action.value == "sell" and crossed_up):
                    level.triggered = True
                    result = None
                    if request.execute:
                        dry_run = exchange.id != "paper" and not settings.live_trading_enabled
                        if dry_run:
                            result = {
                                "exchange": exchange.id,
                                "symbol": request.symbol,
                                "action": level.action.value,
                                "quote_size": level.quote_size,
                                "status": "dry_run",
                                "detail": {"reason": "Grid bot dry-run; LIVE_TRADING_ENABLED is false."},
                            }
                        else:
                            order = await exchange.create_market_order(request.symbol, level.action, level.quote_size)
                            result = order.model_dump(mode="json")
                    bot["events"].append(
                        {
                            "type": "grid_fill",
                            "round": round_index + 1,
                            "symbol": request.symbol,
                            "market_type": request.market_type,
                            "action": level.action.value,
                            "level": level.index,
                            "level_price": level.price,
                            "market_price": price,
                            "quote_size": level.quote_size,
                            "result": result,
                            "timestamp": time.time(),
                        }
                    )
            if request.summary_push_enabled and time.time() - float(bot["last_summary_at"]) >= request.summary_interval_seconds:
                await _send_grid_summary(bot)
                bot["last_summary_at"] = time.time()
                bot["summary_count"] += 1
            bot["plan"] = plan.model_dump(mode="json")
            last_price = price
            if request.interval_seconds and round_index < request.rounds - 1:
                await asyncio.sleep(request.interval_seconds)
        bot["status"] = "completed"
    except Exception as exc:
        logger.exception("grid bot failed")
        bot["status"] = "failed"
        bot["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        bot["finished_at"] = time.time()


def _public_grid_bot(bot: dict[str, Any]) -> dict[str, Any]:
    public = dict(bot)
    public["elapsed_seconds"] = round(time.time() - float(bot["started_at"]), 3)
    public["events"] = list(bot["events"])[-200:]
    return public


def _run_interval(request: BatchRunRequest) -> float:
    if request.interval_seconds:
        return request.interval_seconds
    if request.duration_seconds and request.rounds > 1 and not request.until_stopped:
        return request.duration_seconds / max(request.rounds - 1, 1)
    return 0.0


async def _send_grid_summary(bot: dict[str, Any]) -> None:
    plan = bot.get("plan") or {}
    events = bot.get("events") or []
    buys = [event for event in events if event.get("action") == "buy"]
    sells = [event for event in events if event.get("action") == "sell"]
    filled = [event for event in events if event.get("result") and event["result"].get("status") not in {"failed", "blocked"}]
    triggered = len(events)
    total_levels = len(plan.get("levels") or [])
    title = f"网格机器人总结 {plan.get('symbol', '-')}"
    rows = [
        "## 网格机器人交易总结",
        "",
        f"- Bot ID：{bot.get('id')}",
        f"- 状态：{bot.get('status')}",
        f"- 标的：{plan.get('symbol', '-')}",
        f"- 市场：{plan.get('market_type', '-')}",
        f"- 策略：{GRID_STRATEGIES.get(plan.get('strategy', ''), {}).get('label', plan.get('strategy', '-'))}",
        f"- 间距：{'等比间距' if plan.get('spacing_mode') == 'geometric' else '等差间距'}",
        f"- 区间：{plan.get('lower_price', '-'):.4f} - {plan.get('upper_price', '-'):.4f}" if plan.get("lower_price") else "- 区间：-",
        f"- 当前轮次：{bot.get('round', 0)}",
        f"- 网格数：{total_levels}",
        f"- 已触发：{triggered}",
        f"- 买入触发：{len(buys)}",
        f"- 卖出触发：{len(sells)}",
        f"- 已提交/模拟订单：{len(filled)}",
        f"- 未触发：{max(total_levels - triggered, 0)}",
        "",
        "## 最近事件",
        "",
    ]
    for event in events[-10:]:
        rows.append(
            f"- R{event.get('round')} {_action_label(event.get('action'))} level={event.get('level')} "
            f"level_price={float(event.get('level_price', 0)):.4f} market={float(event.get('market_price', 0)):.4f} "
            f"status={(event.get('result') or {}).get('status', 'planned')}"
        )
    try:
        await engine.notifier.send(title, "\n".join(rows))
    except Exception as exc:
        logger.warning("grid summary notification failed: %s", exc)


def _serialize_batch_item(item: Any) -> dict:
    if isinstance(item, Exception):
        return {"error": f"{type(item).__name__}: {item}"}
    return item


def _simulate_strategy(symbol: str, timeframe: str, strategy_name: str, candles: list, initial_quote: float, trade_quote_size: float) -> dict:
    strategy = STRATEGIES[strategy_name]
    cash = initial_quote
    position = 0.0
    trades = []
    equity_curve = []
    start_index = min(40, max(1, len(candles) - 1))
    for index in range(start_index, len(candles)):
        window = candles[: index + 1]
        snapshot = engine.snapshot_from_candles("simulation", symbol, timeframe, window)
        signal = strategy.evaluate(snapshot)
        price = window[-1].close
        if signal.action.value == "buy" and cash > 0:
            cost = min(cash, trade_quote_size)
            amount = cost / price
            cash -= cost
            position += amount
            trades.append({"index": index, "timestamp": window[-1].timestamp, "action": "buy", "price": price, "amount": amount, "cost": cost})
        elif signal.action.value == "sell" and position > 0:
            amount = min(position, trade_quote_size / price)
            proceeds = amount * price
            position -= amount
            cash += proceeds
            trades.append({"index": index, "timestamp": window[-1].timestamp, "action": "sell", "price": price, "amount": amount, "proceeds": proceeds})
        equity = cash + position * price
        equity_curve.append({"timestamp": window[-1].timestamp, "price": price, "equity": equity, "action": signal.action.value})
    final_price = candles[-1].close
    final_equity = cash + position * final_price
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "strategy_label": _strategy_label(strategy_name),
        "start_time": candles[0].timestamp,
        "end_time": candles[-1].timestamp,
        "initial_quote": initial_quote,
        "final_equity": final_equity,
        "profit": final_equity - initial_quote,
        "profit_pct": (final_equity - initial_quote) / initial_quote * 100,
        "cash": cash,
        "position": position,
        "trades": trades,
        "equity_curve": equity_curve,
        "candles": [{"timestamp": candle.timestamp, "close": candle.close} for candle in candles],
    }


def _filter_candles_by_time(candles: list, start_time: Optional[int], end_time: Optional[int]) -> list:
    filtered = candles
    if start_time is not None:
        filtered = [candle for candle in filtered if candle.timestamp >= start_time]
    if end_time is not None:
        filtered = [candle for candle in filtered if candle.timestamp <= end_time]
    return filtered


async def _fetch_simulation_candles(
    exchange: Any,
    symbol: str,
    timeframe: str,
    start_time: Optional[int],
    end_time: Optional[int],
    requested_limit: int,
) -> list:
    if start_time is None:
        fetch_limit = requested_limit
        if end_time is not None:
            fetch_limit = min(MAX_SIMULATION_CANDLES, max(requested_limit, 500))
        candles = await exchange.fetch_ohlcv(symbol, timeframe, limit=fetch_limit)
        return _filter_candles_by_time(candles, start_time, end_time)

    fetch_limit = _simulate_fetch_limit(timeframe, start_time, end_time, requested_limit)
    timeframe_ms = _timeframe_to_ms(timeframe)
    candles = []
    seen_timestamps: set[int] = set()
    since = start_time

    while len(candles) < fetch_limit:
        batch_limit = min(1000, fetch_limit - len(candles))
        batch = await exchange.fetch_ohlcv_since(symbol, timeframe, since, limit=batch_limit)
        batch = _filter_candles_by_time(batch, start_time, end_time)
        new_candles = [candle for candle in batch if candle.timestamp not in seen_timestamps]
        if not new_candles:
            break
        candles.extend(new_candles)
        seen_timestamps.update(candle.timestamp for candle in new_candles)
        last_timestamp = max(candle.timestamp for candle in new_candles)
        if end_time is not None and last_timestamp >= end_time:
            break
        next_since = last_timestamp + timeframe_ms
        if next_since <= since:
            break
        since = next_since

    return sorted(candles, key=lambda candle: candle.timestamp)


def _simulate_fetch_limit(timeframe: str, start_time: int, end_time: Optional[int], requested_limit: int) -> int:
    if end_time is None:
        return requested_limit
    step = _timeframe_to_ms(timeframe)
    expected = int((end_time - start_time) / step) + 1
    return max(1, min(MAX_SIMULATION_CANDLES, max(requested_limit, expected)))


def _timeframe_to_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    amount = int(timeframe[:-1] or "1")
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 3_600_000
    if unit == "d":
        return amount * 86_400_000
    return 3_600_000


def _validate_strategies(strategies: list[str]) -> None:
    unknown = [strategy for strategy in strategies if strategy not in STRATEGIES]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown strategies: {', '.join(unknown)}")


def _strategy_label(name: str) -> str:
    if name in BUILTIN_STRATEGY_LABELS:
        return BUILTIN_STRATEGY_LABELS[name]
    strategy = STRATEGIES.get(name)
    definition = getattr(strategy, "definition", {}) if strategy else {}
    display_name = str(definition.get("display_name", "")).strip() if isinstance(definition, dict) else ""
    if display_name:
        return display_name
    return f"自定义策略：{name}"


def _strategy_labels() -> dict[str, str]:
    return {name: _strategy_label(name) for name in STRATEGIES}


def _action_label(action: Any) -> str:
    return {"buy": "买入", "sell": "卖出", "hold": "观望"}.get(str(action), str(action or "-"))


def _dashboard_html(page: str = "dashboard") -> str:
    strategy_labels = _strategy_labels()
    strategy_options = "\n".join(f'<option value="{name}">{strategy_labels[name]}</option>' for name in STRATEGIES)
    strategy_toggles = "\n".join(
        f'<label class="toggle compact"><span>{strategy_labels[name]}</span><input class="multiStrategy" type="checkbox" value="{name}" {"checked" if name == "strategy_ensemble" else ""}></label>'
        for name in STRATEGIES
    )
    strategy_labels_json = json.dumps(strategy_labels, ensure_ascii=False)
    nav_items = [
        ("dashboard", "/dashboard", "实时看板"),
        ("trade", "/trade", "单策略运行"),
        ("batch", "/batch", "多策略与回测"),
        ("grid", "/grid", "网格机器人"),
        ("strategies", "/strategy-center", "策略中心"),
        ("config", "/config", "配置"),
    ]
    nav_links = "\n".join(
        f'<a class="{"active" if key == page else ""}" href="{href}">{label}</a>' for key, href, label in nav_items
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trading with AI 控制台</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657282;
      --line: #d8dee6;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #b45309;
      --danger: #b91c1c;
      --ok: #15803d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    .topbar {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }}
    .brand {{
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    .subtle {{ color: var(--muted); font-size: 13px; }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px;
      display: grid;
      grid-template-columns: minmax(290px, 360px) 1fr;
      gap: 18px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .section-head {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    h2 {{
      margin: 0;
      font-size: 15px;
      font-weight: 720;
      letter-spacing: 0;
    }}
    .form {{
      padding: 16px;
      display: grid;
      gap: 14px;
    }}
    label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    input, select, textarea {{
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }}
    textarea {{
      min-height: 180px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
      resize: vertical;
    }}
    input:focus, select:focus, textarea:focus {{
      outline: 2px solid rgba(15, 118, 110, 0.18);
      border-color: var(--accent);
    }}
    .toggle {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      color: var(--ink);
    }}
    .toggle input {{ width: 18px; min-height: 18px; }}
    .actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 4px;
    }}
    button {{
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 12px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      background: #fff;
      color: var(--ink);
    }}
    button.primary {{
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }}
    button.primary:hover {{ background: var(--accent-dark); }}
    button:disabled {{
      cursor: wait;
      opacity: 0.65;
    }}
    .status-grid {{
      padding: 16px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 82px;
      background: #fbfcfd;
      display: grid;
      align-content: space-between;
      gap: 8px;
    }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .metric strong {{
      font-size: 18px;
      overflow-wrap: anywhere;
    }}
    .content {{
      padding: 16px;
      display: grid;
      gap: 14px;
    }}
    .wide {{ grid-column: 1 / -1; }}
    .settings-grid {{
      padding: 16px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .settings-actions {{
      padding: 0 16px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .notice {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .runner-grid {{
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(260px, 340px) 1fr;
      gap: 16px;
    }}
    .strategy-list {{
      display: grid;
      gap: 8px;
    }}
    .toggle.compact {{
      min-height: 38px;
      padding: 8px 10px;
    }}
    .inline-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .chart-wrap {{
      display: grid;
      gap: 12px;
    }}
    .dashboard-grid {{
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(260px, 420px) 1fr;
      gap: 16px;
    }}
    .grid-levels {{
      max-height: 320px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .event-open {{ color: var(--ok); font-weight: 720; }}
    .event-close {{ color: var(--danger); font-weight: 720; }}
    canvas {{
      width: 100%;
      height: 280px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      table-layout: fixed;
    }}
    th, td {{
      padding: 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      overflow-wrap: anywhere;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 700; }}
    .result-line {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 720;
      background: #eef2f7;
      color: var(--muted);
    }}
    .badge.buy {{ background: #dcfce7; color: var(--ok); }}
    .badge.sell {{ background: #fee2e2; color: var(--danger); }}
    .badge.hold {{ background: #fef3c7; color: var(--warn); }}
    pre {{
      margin: 0;
      min-height: 260px;
      max-height: 480px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111827;
      color: #e5e7eb;
      padding: 14px;
      font-size: 12px;
      line-height: 1.55;
    }}
    .links {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    a {{ color: var(--accent-dark); text-decoration: none; font-weight: 650; }}
    a:hover {{ text-decoration: underline; }}
    .links a {{
      min-height: 32px;
      display: inline-flex;
      align-items: center;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 4px 8px;
    }}
    .links a.active {{
      border-color: rgba(15, 118, 110, 0.24);
      background: rgba(15, 118, 110, 0.08);
      color: var(--accent);
    }}
    .page-section {{ display: none; }}
    .page-section.active {{ display: block; }}
    @media (max-width: 860px) {{
      main {{ grid-template-columns: 1fr; padding: 14px; }}
      .topbar {{ padding: 14px; align-items: flex-start; flex-direction: column; }}
      .status-grid, .result-line, .settings-grid, .dashboard-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 520px) {{
      .status-grid, .result-line, .actions, .settings-grid, .dashboard-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <h1>Trading with AI 控制台</h1>
        <div class="subtle">先在策略中心导入策略，再到单策略运行或多策略与回测里执行</div>
      </div>
      <div class="links">
        {nav_links}
        <a href="/docs">API 文档</a>
        <a href="/health" target="_blank" rel="noreferrer">健康检查</a>
      </div>
    </div>
  </header>
  <main>
    <section class="page-section" data-page="trade">
      <div class="section-head">
        <h2>单策略运行</h2>
        <span class="badge" id="serviceStatus">检查中</span>
      </div>
      <form class="form" id="runForm">
        <div class="notice">
          适合运行一个策略和一个主标的。上传的新策略会出现在“策略”下拉菜单里；要多币种、多策略或历史回测，请去“多策略与回测”。
        </div>
        <label>交易所
          <select id="exchange">
            <option value="paper">paper</option>
            <option value="binance">binance</option>
            <option value="bybit">bybit</option>
            <option value="okx">okx</option>
            <option value="bitget">bitget</option>
            <option value="kucoin">kucoin</option>
            <option value="gateio">gateio</option>
            <option value="hyperliquid">hyperliquid</option>
          </select>
        </label>
        <label>交易标的
          <input id="symbol" value="{settings.default_symbol}" autocomplete="off">
          <span class="subtle">普通策略使用这里的币种；NOFX 策略会优先使用策略内置选币配置。</span>
        </label>
        <label>市场类型
          <select id="tradeMarketType">
            <option value="spot">现货 spot</option>
            <option value="swap">永续合约 swap</option>
            <option value="future">交割合约 future</option>
          </select>
        </label>
        <label>行情周期
          <select id="timeframe">
            <option value="5m">5 分钟 K 线</option>
            <option value="15m">15 分钟 K 线</option>
            <option value="1h" selected>1 小时 K 线</option>
            <option value="4h">4 小时 K 线</option>
            <option value="1d">1 天 K 线</option>
          </select>
          <span class="subtle">策略判断时看的 K 线级别，不是运行间隔。</span>
        </label>
        <label>策略
          <select id="strategy">
            {strategy_options}
          </select>
        </label>
        <div class="inline-grid">
          <label>执行间隔
            <div class="inline-grid">
              <input id="tradeInterval" type="number" min="1" max="86400" step="1" value="1">
              <select id="tradeIntervalUnit">
                <option value="60" selected>分钟</option>
                <option value="1">秒</option>
                <option value="3600">小时</option>
              </select>
            </div>
            <span class="subtle">机器人多久检查并执行一次策略。</span>
          </label>
          <label>最大轮次
            <input id="tradeRounds" type="number" min="1" max="10000" step="1" value="100">
            <span class="subtle">每检查一次算 1 轮；跑满后自动停止。</span>
          </label>
        </div>
        <div class="inline-grid">
          <label>运行时长
            <div class="inline-grid">
              <input id="tradeDuration" type="number" min="0" max="2592000" step="1" value="0">
              <select id="tradeDurationUnit">
                <option value="60" selected>分钟</option>
                <option value="1">秒</option>
                <option value="3600">小时</option>
                <option value="86400">天</option>
              </select>
            </div>
            <span class="subtle">填 0 表示不按时长限制，只看轮次或手动停止。</span>
          </label>
          <label class="toggle">
            <span>一直运行直到手动停止</span>
            <input id="tradeUntilStopped" type="checkbox">
          </label>
        </div>
        <label class="toggle">
          <span>
            执行订单
            <span class="subtle">未开启实盘时会 dry-run</span>
          </span>
          <input id="execute" type="checkbox">
        </label>
        <div class="actions">
          <button type="submit" class="primary" id="runButton">单次决策</button>
          <button type="button" id="startTradeRunButton">启动循环运行</button>
          <button type="button" id="stopTradeRunButton" disabled>停止循环运行</button>
          <button type="button" id="clearButton">清空结果</button>
        </div>
      </form>
    </section>
    <section class="page-section" data-page="trade">
      <div class="section-head">
        <h2>运行结果</h2>
        <span class="subtle" id="lastRun">尚未运行</span>
      </div>
      <div class="status-grid">
        <div class="metric"><span>动作</span><strong id="actionValue">-</strong></div>
        <div class="metric"><span>置信度</span><strong id="confidenceValue">-</strong></div>
        <div class="metric"><span>执行模式</span><strong id="modeValue">-</strong></div>
        <div class="metric"><span>市场/订单</span><strong id="orderStatus">-</strong></div>
      </div>
      <div class="content">
        <div class="result-line">
          <div><span class="subtle">标的</span><br><strong id="resultSymbol">-</strong></div>
          <div><span class="subtle">金额</span><br><strong id="quoteSize">-</strong></div>
          <div><span class="subtle">阻止原因</span><br><strong id="blockReason">-</strong></div>
        </div>
        <div class="notice" id="aiDecisionSummary">AI 决策过程会在运行后显示。</div>
        <div class="result-line">
          <div><span class="subtle">策略信号</span><br><strong id="strategySignalValue">-</strong></div>
          <div><span class="subtle">AI 复核</span><br><strong id="aiDecisionValue">-</strong></div>
          <div><span class="subtle">AI 风险提示</span><br><strong id="aiRiskNotesValue">-</strong></div>
        </div>
        <pre id="rawOutput">等待运行...</pre>
      </div>
    </section>
    <section class="wide page-section" data-page="batch">
      <div class="section-head">
        <h2>多策略运行与历史回测</h2>
        <span class="subtle" id="multiStatus">待命</span>
      </div>
      <div class="runner-grid">
        <div class="form">
          <div class="notice">
            适合同时跑多个币种或多个策略，也可以用指定时间段做历史回测。这里勾选的策略包含你在策略中心上传的自定义/NOFX 策略。
          </div>
          <label>多币种
            <input id="multiSymbols" value="BTC/USDT, ETH/USDT" autocomplete="off">
          </label>
          <label>市场类型
            <select id="multiMarketType">
              <option value="spot">现货 spot</option>
              <option value="swap">永续合约 swap</option>
              <option value="future">交割合约 future</option>
            </select>
          </label>
          <label>策略开关
            <div class="strategy-list" id="multiStrategyList">
              {strategy_toggles}
            </div>
          </label>
          <div class="inline-grid">
            <label>轮次
              <input id="multiRounds" type="number" min="1" max="100" value="1">
            </label>
            <label>运行时长 秒
              <input id="multiDuration" type="number" min="0" max="86400" value="0">
            </label>
          </div>
          <div class="inline-grid">
            <label>模拟初始资金
              <input id="simInitialQuote" type="number" min="1" step="1" value="1000">
            </label>
            <label>模拟 K 线上限
              <input id="simLookback" type="number" min="60" max="10000" value="120">
            </label>
          </div>
          <div class="inline-grid">
            <label>模拟开始时间
              <input id="simStartTime" type="datetime-local">
            </label>
            <label>模拟结束时间
              <input id="simEndTime" type="datetime-local">
            </label>
          </div>
          <label class="toggle">
            <span>批量运行时执行订单</span>
            <input id="multiExecute" type="checkbox">
          </label>
          <div class="actions">
            <button type="button" class="primary" id="batchRunButton">批量单次决策</button>
            <button type="button" id="simulateButton">历史回测</button>
            <button type="button" id="startRunButton">启动多策略运行</button>
          </div>
        </div>
        <div class="content">
          <div class="chart-wrap">
            <canvas id="strategyChart" width="900" height="280"></canvas>
            <div class="notice" id="chartCaption">模拟后显示价格和权益曲线。</div>
          </div>
          <div style="overflow:auto;">
            <table>
              <thead>
                <tr><th>标的</th><th>策略</th><th>动作/收益</th><th>状态</th></tr>
              </thead>
              <tbody id="multiTable"><tr><td colspan="4">暂无结果</td></tr></tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
    <section class="wide page-section" data-page="grid">
      <div class="section-head">
        <h2>网格机器人</h2>
        <span class="subtle" id="gridStatus">待命</span>
      </div>
      <div class="dashboard-grid">
        <div class="form">
          <label>网格币种
            <input id="gridSymbol" value="{settings.default_symbol}" autocomplete="off">
          </label>
          <div class="inline-grid">
            <label>市场类型
              <select id="gridMarketType">
                <option value="spot">现货 spot</option>
                <option value="swap">合约 swap</option>
                <option value="future">交割合约 future</option>
              </select>
            </label>
            <label>间距模式
              <select id="gridSpacingMode">
                <option value="arithmetic">等差间距</option>
                <option value="geometric">等比间距</option>
              </select>
            </label>
          </div>
          <div class="inline-grid">
            <label>网格策略
              <select id="gridStrategy">
                <option value="neutral_range">中性区间</option>
                <option value="trend_follow">趋势跟随</option>
                <option value="volatility_adaptive">波动自适应</option>
                <option value="ai_adaptive">AI 自适应</option>
              </select>
            </label>
            <label>轮询间隔 秒
              <input id="gridInterval" type="number" min="0" max="3600" step="0.5" value="2">
            </label>
          </div>
          <div class="inline-grid">
            <label>下边界
              <input id="gridLower" type="number" min="0" step="0.0001" placeholder="留空自动">
            </label>
            <label>上边界
              <input id="gridUpper" type="number" min="0" step="0.0001" placeholder="留空自动">
            </label>
          </div>
          <div class="inline-grid">
            <label>网格数量
              <input id="gridCount" type="number" min="3" max="100" value="12">
            </label>
            <label>投入金额
              <input id="gridInvestment" type="number" min="1" step="1" value="500">
            </label>
          </div>
          <div class="inline-grid">
            <label>杠杆
              <input id="gridLeverage" type="number" min="1" max="125" step="1" value="1">
            </label>
            <label>总结间隔
              <input value="2 小时" disabled>
            </label>
          </div>
          <label>AI 策略说明
            <textarea id="gridAiGuidance" spellcheck="false" placeholder="例如：偏保守，只做当前价格上下 5% 区间，避免追高。"></textarea>
          </label>
          <label class="toggle">
            <span>触发网格时执行订单</span>
            <input id="gridExecute" type="checkbox">
          </label>
          <label class="toggle">
            <span>
              每 2 小时推送交易总结
              <span class="subtle">使用方糖 SendKey</span>
            </span>
            <input id="gridSummaryPush" type="checkbox">
          </label>
          <div class="actions">
            <button type="button" class="primary" id="gridPlanButton">生成网格</button>
            <button type="button" id="gridStartButton">启动机器人</button>
          </div>
        </div>
        <div class="content">
          <div class="status-grid">
            <div class="metric"><span>Bot ID</span><strong id="gridBotId">-</strong></div>
            <div class="metric"><span>状态</span><strong id="gridBotState">-</strong></div>
            <div class="metric"><span>当前价格</span><strong id="gridCurrentPrice">-</strong></div>
            <div class="metric"><span>行情来源/触发</span><strong id="gridEventCount">-</strong></div>
          </div>
          <div class="grid-levels">
            <table>
              <thead><tr><th>#</th><th>价格</th><th>动作</th><th>金额</th><th>状态</th></tr></thead>
              <tbody id="gridLevelsTable"><tr><td colspan="5">生成网格后显示价位。</td></tr></tbody>
            </table>
          </div>
          <table>
            <thead><tr><th>轮次</th><th>标的</th><th>动作</th><th>价位/市价</th><th>订单</th></tr></thead>
            <tbody id="gridEventsTable"><tr><td colspan="5">启动机器人后显示触发事件。</td></tr></tbody>
          </table>
          <pre id="gridOutput">等待生成网格...</pre>
        </div>
      </div>
    </section>
    <section class="wide page-section" data-page="strategies">
      <div class="section-head">
        <h2>策略模板与上传</h2>
        <span class="subtle" id="strategyUploadStatus">待命</span>
      </div>
      <div class="dashboard-grid">
        <div class="form">
          <div class="notice">
            这里负责导入和管理策略，不负责执行。上传成功后，到“单策略运行”选择一个策略运行；需要多币种、多策略或历史验证时，到“多策略与回测”勾选。
          </div>
          <div class="actions">
            <button type="button" id="loadTemplateButton">载入模板</button>
            <button type="button" id="loadNofxTemplateButton">载入 NOFX 模板</button>
            <button type="button" class="primary" id="uploadStrategyButton">上传策略</button>
          </div>
          <label>上传 JSON 文件
            <input id="strategyFileInput" type="file" accept="application/json,.json">
          </label>
          <label>策略 JSON
            <textarea id="strategyJson" spellcheck="false"></textarea>
          </label>
        </div>
        <div class="content">
          <div class="notice">
            支持 JSON 规则策略和 NOFX StrategyConfig 兼容导入，不执行上传代码。可用指标：rsi、sma_cross、price_vs_sma、breakout。NOFX 配置会转换为本系统可运行的规则策略，并保留原始配置。上传后无需重启，运行页会自动出现该策略。
          </div>
          <pre id="strategyUploadOutput">等待策略模板或上传...</pre>
        </div>
      </div>
    </section>
    <section class="wide page-section" data-page="dashboard">
      <div class="section-head">
        <h2>实时运行看板</h2>
        <span class="subtle" id="runStatus">未开始</span>
      </div>
      <div class="status-grid">
        <div class="metric"><span>Run ID</span><strong id="runIdValue">-</strong></div>
        <div class="metric"><span>状态</span><strong id="runStateValue">-</strong></div>
        <div class="metric"><span>轮次</span><strong id="runRoundValue">-</strong></div>
        <div class="metric"><span>事件数</span><strong id="runEventCountValue">-</strong></div>
      </div>
      <div class="content">
        <div class="section-head compact-head">
          <h3>交易所可用余额</h3>
          <button type="button" id="refreshBalancesButton">刷新余额</button>
        </div>
        <div class="status-grid">
          <div class="metric"><span>交易所</span><strong id="balanceExchangeValue">-</strong></div>
          <div class="metric"><span>市场类型</span><strong id="balanceMarketTypeValue">-</strong></div>
          <div class="metric"><span>更新时间</span><strong id="balanceUpdatedAtValue">-</strong></div>
          <div class="metric"><span>状态</span><strong id="balanceStatus">未刷新</strong></div>
        </div>
        <div class="notice" id="balanceMessage">点击刷新余额会读取当前配置的交易所账户；密钥不会在页面显示。</div>
        <table>
          <thead>
            <tr><th>资产</th><th>总余额</th></tr>
          </thead>
          <tbody id="balanceTable"><tr><td colspan="2">暂无余额，请点击刷新。</td></tr></tbody>
        </table>
      </div>
      <div class="content">
        <table>
          <thead>
            <tr><th>类型</th><th>标的</th><th>策略</th><th>轮次/置信度</th><th>状态</th></tr>
          </thead>
          <tbody id="runEventTable"><tr><td colspan="5">策略开始执行后这里会显示实时开仓和平仓。</td></tr></tbody>
        </table>
      </div>
    </section>
    <section class="wide page-section" data-page="config">
      <div class="section-head">
        <h2>运行配置</h2>
        <span class="subtle" id="settingsStatus">未加载</span>
      </div>
      <form id="settingsForm">
        <div class="settings-grid">
          <label>AI Provider
            <select id="cfg_ai_provider">
              <option value="openrouter">openrouter</option>
              <option value="openai">openai</option>
              <option value="minimax">minimax</option>
              <option value="mimo">mimo</option>
              <option value="custom_openai_compatible">custom_openai_compatible</option>
            </select>
          </label>
          <label>AI Model
            <select id="cfg_ai_model"></select>
          </label>
          <label>自定义 Model
            <input id="cfg_ai_model_custom" autocomplete="off" placeholder="选择自定义时填写">
          </label>
          <label>AI Base URL
            <select id="cfg_ai_base_url"></select>
          </label>
          <label>自定义 Base URL
            <input id="cfg_ai_base_url_custom" autocomplete="off" placeholder="选择自定义时填写">
          </label>
          <label>AI API Key
            <input id="cfg_ai_api_key" type="password" autocomplete="new-password" placeholder="留空不修改">
          </label>
          <label>默认交易所
            <select id="cfg_default_exchange">
              <option value="paper">paper</option>
              <option value="binance">binance</option>
              <option value="bybit">bybit</option>
              <option value="okx">okx</option>
              <option value="bitget">bitget</option>
              <option value="kucoin">kucoin</option>
              <option value="gateio">gateio</option>
              <option value="hyperliquid">hyperliquid</option>
            </select>
          </label>
          <label>交易所 ID
            <input id="cfg_exchange_id" autocomplete="off">
          </label>
          <label>交易所 API Key
            <input id="cfg_exchange_api_key" type="password" autocomplete="new-password" placeholder="留空不修改">
          </label>
          <label>交易所 Secret
            <input id="cfg_exchange_secret" type="password" autocomplete="new-password" placeholder="留空不修改">
          </label>
          <label>交易所 Password
            <input id="cfg_exchange_password" type="password" autocomplete="new-password" placeholder="留空不修改">
          </label>
          <label>默认标的
            <input id="cfg_default_symbol" autocomplete="off">
          </label>
          <label>默认周期
            <select id="cfg_default_timeframe">
              <option value="5m">5m</option>
              <option value="15m">15m</option>
              <option value="1h">1h</option>
              <option value="4h">4h</option>
              <option value="1d">1d</option>
            </select>
          </label>
          <label>单次交易金额
            <input id="cfg_trade_quote_size" type="number" min="0" step="0.01">
          </label>
          <label>最大持仓金额
            <input id="cfg_max_position_quote" type="number" min="0" step="0.01">
          </label>
          <label>资金风控优先级
            <select id="cfg_risk_limit_priority">
              <option value="global">全局限制优先</option>
              <option value="strategy">策略内部限制优先</option>
            </select>
          </label>
          <label>最低 AI 置信度
            <input id="cfg_min_ai_confidence" type="number" min="0" max="1" step="0.01">
          </label>
          <label>方糖 SendKey
            <input id="cfg_serverchan_sendkey" type="password" autocomplete="new-password" placeholder="留空不修改">
          </label>
          <label class="toggle">
            <span>启用实盘交易</span>
            <input id="cfg_live_trading_enabled" type="checkbox">
          </label>
          <label class="toggle">
            <span>交易所 Sandbox</span>
            <input id="cfg_exchange_sandbox" type="checkbox">
          </label>
          <label class="toggle">
            <span>成交后推送方糖</span>
            <input id="cfg_notify_trade_success" type="checkbox">
          </label>
          <label class="toggle">
            <span>dry-run 也推送</span>
            <input id="cfg_notify_dry_run" type="checkbox">
          </label>
        </div>
        <div class="settings-actions">
          <div class="notice" id="secretNotice">密钥字段保存后会加密落盘；再次打开只显示已配置状态，留空不会覆盖原值。</div>
          <span class="subtle" id="serverChanTestStatus">未测试</span>
          <button type="button" id="testServerChanButton">测试方糖推送</button>
          <button type="submit" class="primary" id="saveSettingsButton">保存配置</button>
        </div>
      </form>
    </section>
  </main>
  <script>
    const currentPage = "{page}";
    let strategyLabels = {strategy_labels_json};
    const gridStrategyLabels = {{
      neutral_range: '中性区间网格',
      trend_follow: '趋势跟随网格',
      volatility_adaptive: '波动自适应网格',
      ai_adaptive: 'AI 自适应网格'
    }};
    const actionLabels = {{ buy: '买入', sell: '卖出', hold: '观望' }};
    const form = document.getElementById('runForm');
    const settingsForm = document.getElementById('settingsForm');
    const runButton = document.getElementById('runButton');
    const saveSettingsButton = document.getElementById('saveSettingsButton');
    const testServerChanButton = document.getElementById('testServerChanButton');
    const refreshBalancesButton = document.getElementById('refreshBalancesButton');
    const clearButton = document.getElementById('clearButton');
    const batchRunButton = document.getElementById('batchRunButton');
    const simulateButton = document.getElementById('simulateButton');
    const startRunButton = document.getElementById('startRunButton');
    const startTradeRunButton = document.getElementById('startTradeRunButton');
    const stopTradeRunButton = document.getElementById('stopTradeRunButton');
    const gridPlanButton = document.getElementById('gridPlanButton');
    const gridStartButton = document.getElementById('gridStartButton');
    const loadTemplateButton = document.getElementById('loadTemplateButton');
    const loadNofxTemplateButton = document.getElementById('loadNofxTemplateButton');
    const uploadStrategyButton = document.getElementById('uploadStrategyButton');
    const strategyFileInput = document.getElementById('strategyFileInput');
    const strategyJson = document.getElementById('strategyJson');
    const strategyUploadOutput = document.getElementById('strategyUploadOutput');
    const rawOutput = document.getElementById('rawOutput');
    const multiTable = document.getElementById('multiTable');
    const gridLevelsTable = document.getElementById('gridLevelsTable');
    const gridEventsTable = document.getElementById('gridEventsTable');
    const gridOutput = document.getElementById('gridOutput');
    const runEventTable = document.getElementById('runEventTable');
    const balanceTable = document.getElementById('balanceTable');
    const chartCanvas = document.getElementById('strategyChart');
    let currentRunId = null;
    let runPollTimer = null;
    let currentGridBotId = null;
    let gridPollTimer = null;
    const fields = {{
      action: document.getElementById('actionValue'),
      confidence: document.getElementById('confidenceValue'),
      mode: document.getElementById('modeValue'),
      order: document.getElementById('orderStatus'),
      symbol: document.getElementById('resultSymbol'),
      quote: document.getElementById('quoteSize'),
      block: document.getElementById('blockReason'),
      aiDecisionSummary: document.getElementById('aiDecisionSummary'),
      strategySignal: document.getElementById('strategySignalValue'),
      aiDecision: document.getElementById('aiDecisionValue'),
      aiRiskNotes: document.getElementById('aiRiskNotesValue'),
      lastRun: document.getElementById('lastRun'),
      service: document.getElementById('serviceStatus'),
      settingsStatus: document.getElementById('settingsStatus'),
      multiStatus: document.getElementById('multiStatus'),
      chartCaption: document.getElementById('chartCaption'),
      strategyUploadStatus: document.getElementById('strategyUploadStatus'),
      runStatus: document.getElementById('runStatus'),
      runId: document.getElementById('runIdValue'),
      runState: document.getElementById('runStateValue'),
      runRound: document.getElementById('runRoundValue'),
      runEventCount: document.getElementById('runEventCountValue'),
      balanceStatus: document.getElementById('balanceStatus'),
      balanceExchange: document.getElementById('balanceExchangeValue'),
      balanceMarketType: document.getElementById('balanceMarketTypeValue'),
      balanceUpdatedAt: document.getElementById('balanceUpdatedAtValue'),
      balanceMessage: document.getElementById('balanceMessage'),
      serverChanTestStatus: document.getElementById('serverChanTestStatus'),
      gridStatus: document.getElementById('gridStatus'),
      gridBotId: document.getElementById('gridBotId'),
      gridBotState: document.getElementById('gridBotState'),
      gridCurrentPrice: document.getElementById('gridCurrentPrice'),
      gridEventCount: document.getElementById('gridEventCount')
    }};
    const configIds = [
      'ai_provider', 'ai_model', 'ai_base_url', 'ai_api_key',
      'default_exchange', 'exchange_id', 'exchange_api_key', 'exchange_secret', 'exchange_password',
      'default_symbol', 'default_timeframe', 'trade_quote_size', 'max_position_quote', 'risk_limit_priority', 'min_ai_confidence',
      'serverchan_sendkey', 'live_trading_enabled', 'exchange_sandbox', 'notify_trade_success', 'notify_dry_run'
    ];
    const secretIds = new Set(['ai_api_key', 'exchange_api_key', 'exchange_secret', 'exchange_password', 'serverchan_sendkey']);
    const providerOptions = {{
      openrouter: {{
        models: [
          ['openai/gpt-4o-mini', 'OpenAI GPT-4o mini'],
          ['openai/gpt-4.1-mini', 'OpenAI GPT-4.1 mini'],
          ['anthropic/claude-3.5-sonnet', 'Claude 3.5 Sonnet'],
          ['google/gemini-2.0-flash-001', 'Gemini 2.0 Flash']
        ],
        bases: [['https://openrouter.ai/api/v1', 'OpenRouter']]
      }},
      openai: {{
        models: [
          ['gpt-4o-mini', 'GPT-4o mini'],
          ['gpt-4.1-mini', 'GPT-4.1 mini'],
          ['gpt-4.1', 'GPT-4.1']
        ],
        bases: [['https://api.openai.com/v1', 'OpenAI']]
      }},
      minimax: {{
        models: [
          ['MiniMax-M2.7', 'MiniMax-M2.7'],
          ['MiniMax-M2.7-highspeed', 'MiniMax-M2.7 highspeed'],
          ['MiniMax-M2.5', 'MiniMax-M2.5']
        ],
        bases: [
          ['https://api.minimaxi.com/v1', 'MiniMax 中国区'],
          ['https://api.minimax.io/v1', 'MiniMax 国际区']
        ]
      }},
      mimo: {{
        models: [
          ['mimo-vl-7b', 'MiMo VL 7B'],
          ['mimo-7b', 'MiMo 7B']
        ],
        bases: [['https://api.mimo-v2.com/v1', '小米 MiMo']]
      }},
      custom_openai_compatible: {{
        models: [],
        bases: []
      }}
    }};

    function showCurrentPage() {{
      for (const section of document.querySelectorAll('.page-section')) {{
        section.classList.toggle('active', section.dataset.page === currentPage);
      }}
    }}

    async function checkHealth() {{
      try {{
        const response = await fetch('/health');
        fields.service.textContent = response.ok ? '在线' : '异常';
      }} catch (error) {{
        fields.service.textContent = '离线';
      }}
    }}

    function setActionBadge(action) {{
      fields.action.textContent = actionLabel(action);
      fields.action.className = '';
      if (action) fields.action.classList.add('badge', action);
    }}

    function render(data) {{
      const plan = data.plan || {{}};
      const result = data.result || null;
      setActionBadge(plan.action);
      fields.confidence.textContent = plan.confidence == null ? '-' : Number(plan.confidence).toFixed(2);
      fields.mode.textContent = plan.dry_run ? 'dry-run' : 'live';
      fields.order.textContent = `${{data.market_type || '-'}} / ${{result ? result.status : '未执行'}}`;
      fields.symbol.textContent = plan.symbol || '-';
      fields.quote.textContent = plan.quote_size == null ? '-' : plan.quote_size;
      fields.block.textContent = plan.block_reason || '-';
      renderAiDecision(plan);
      fields.lastRun.textContent = new Date().toLocaleString();
      rawOutput.textContent = JSON.stringify(data, null, 2);
    }}

    function reset() {{
      setActionBadge(null);
      fields.confidence.textContent = '-';
      fields.mode.textContent = '-';
      fields.order.textContent = '-';
      fields.symbol.textContent = '-';
      fields.quote.textContent = '-';
      fields.block.textContent = '-';
      fields.aiDecisionSummary.textContent = 'AI 决策过程会在运行后显示。';
      fields.strategySignal.textContent = '-';
      fields.aiDecision.textContent = '-';
      fields.aiRiskNotes.textContent = '-';
      fields.lastRun.textContent = '尚未运行';
      rawOutput.textContent = '等待运行...';
    }}

    function renderAiDecision(plan) {{
      const signal = plan.strategy_signal || {{}};
      const ai = plan.ai_decision || {{}};
      fields.strategySignal.textContent = signal.action
        ? `${{actionLabel(signal.action)}} / ${{Number(signal.confidence || 0).toFixed(2)}}`
        : '-';
      fields.aiDecision.textContent = ai.action
        ? `${{actionLabel(ai.action)}} / ${{Number(ai.confidence || 0).toFixed(2)}}`
        : '-';
      const riskNotes = ai.risk_notes || [];
      fields.aiRiskNotes.textContent = riskNotes.length ? riskNotes.join('；') : '无';
      const steps = plan.decision_steps || [];
      fields.aiDecisionSummary.innerHTML = steps.length
        ? steps.map((step) => `<div>${{escapeHtml(step)}}</div>`).join('')
        : escapeHtml(plan.reason || '暂无 AI 决策信息');
    }}

    function cfgElement(id) {{
      return document.getElementById('cfg_' + id);
    }}

    function setSelectOptions(select, options, customLabel) {{
      select.innerHTML = options.map(([value, label]) => `<option value="${{escapeHtml(value)}}">${{escapeHtml(label)}} · ${{escapeHtml(value)}}</option>`).join('')
        + `<option value="__custom">${{escapeHtml(customLabel)}}</option>`;
    }}

    function selectKnownOrCustom(id, value) {{
      const select = cfgElement(id);
      const custom = cfgElement(id + '_custom');
      const hasOption = [...select.options].some((option) => option.value === value);
      if (value && hasOption) {{
        select.value = value;
        if (custom) custom.value = '';
      }} else if (value) {{
        select.value = '__custom';
        if (custom) custom.value = value;
      }} else {{
        select.selectedIndex = 0;
        if (custom) custom.value = '';
      }}
      toggleCustomConfigInputs();
    }}

    function populateAiMenus(provider, currentModel, currentBaseUrl) {{
      const options = providerOptions[provider] || providerOptions.custom_openai_compatible;
      setSelectOptions(cfgElement('ai_model'), options.models, '自定义模型');
      setSelectOptions(cfgElement('ai_base_url'), options.bases, '自定义 Base URL');
      selectKnownOrCustom('ai_model', currentModel || (options.models[0] && options.models[0][0]) || '');
      selectKnownOrCustom('ai_base_url', currentBaseUrl || (options.bases[0] && options.bases[0][0]) || '');
    }}

    function toggleCustomConfigInputs() {{
      for (const id of ['ai_model', 'ai_base_url']) {{
        const custom = cfgElement(id + '_custom');
        if (custom) custom.disabled = cfgElement(id).value !== '__custom';
      }}
    }}

    function applySettings(settings) {{
      populateAiMenus(settings.ai_provider || 'openrouter', settings.ai_model || '', settings.ai_base_url || '');
      for (const id of configIds) {{
        const element = cfgElement(id);
        const value = settings[id];
        if (!element) continue;
        if (element.type === 'checkbox') {{
          element.checked = Boolean(value);
        }} else if (secretIds.has(id)) {{
          element.value = '';
          element.placeholder = value && value.configured ? '已配置：' + value.masked + '，留空不修改' : '留空不修改';
        }} else if (id === 'ai_model' || id === 'ai_base_url') {{
          selectKnownOrCustom(id, value || '');
        }} else {{
          element.value = value ?? '';
        }}
      }}
      document.getElementById('exchange').value = settings.default_exchange || 'paper';
      document.getElementById('symbol').value = settings.default_symbol || 'BTC/USDT';
      document.getElementById('gridSymbol').value = settings.default_symbol || 'BTC/USDT';
      document.getElementById('timeframe').value = settings.default_timeframe || '1h';
      fields.settingsStatus.textContent = '已加载';
    }}

    async function loadSettings() {{
      try {{
        const response = await fetch('/settings');
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || '配置读取失败');
        applySettings(data.settings);
      }} catch (error) {{
        fields.settingsStatus.textContent = '读取失败';
      }}
    }}

    function collectSettingsPayload() {{
      const payload = {{}};
      for (const id of configIds) {{
        const element = cfgElement(id);
        if (!element) continue;
        if (element.type === 'checkbox') {{
          payload[id] = element.checked;
        }} else if (element.type === 'number') {{
          payload[id] = element.value === '' ? null : Number(element.value);
        }} else if (id === 'ai_model' || id === 'ai_base_url') {{
          payload[id] = element.value === '__custom' ? cfgElement(id + '_custom').value.trim() : element.value;
        }} else {{
          payload[id] = element.value.trim();
        }}
      }}
      return payload;
    }}

    function selectedStrategies() {{
      const values = [...document.querySelectorAll('.multiStrategy:checked')].map((item) => item.value);
      return values.length ? values : ['strategy_ensemble'];
    }}

    async function refreshStrategies() {{
      const response = await fetch('/strategies');
      const data = await readJsonOrText(response);
      if (!response.ok) throw new Error(data.detail || '策略读取失败');
      const selected = new Set(selectedStrategies());
      const strategies = data.strategies || [];
      strategyLabels = data.labels || strategyLabels;
      document.getElementById('strategy').innerHTML = strategies.map((name) => `<option value="${{escapeHtml(name)}}">${{escapeHtml(strategyLabel(name))}}</option>`).join('');
      document.getElementById('multiStrategyList').innerHTML = strategies.map((name) => {{
        const checked = selected.has(name) || (!selected.size && name === 'strategy_ensemble') ? 'checked' : '';
        return `<label class="toggle compact"><span>${{escapeHtml(strategyLabel(name))}}</span><input class="multiStrategy" type="checkbox" value="${{escapeHtml(name)}}" ${{checked}}></label>`;
      }}).join('');
    }}

    function selectedSymbols() {{
      return document.getElementById('multiSymbols').value
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean);
    }}

    function multiPayload() {{
      return {{
        exchange: document.getElementById('exchange').value,
        symbols: selectedSymbols(),
        market_type: document.getElementById('multiMarketType').value,
        timeframe: document.getElementById('timeframe').value,
        strategies: selectedStrategies(),
        execute: document.getElementById('multiExecute').checked,
        rounds: Number(document.getElementById('multiRounds').value || 1),
        duration_seconds: Number(document.getElementById('multiDuration').value || 0),
        interval_seconds: 0,
        until_stopped: false
      }};
    }}

    function simulationPayload() {{
      return {{
        exchange: document.getElementById('exchange').value,
        symbols: selectedSymbols(),
        market_type: document.getElementById('multiMarketType').value,
        timeframe: document.getElementById('timeframe').value,
        strategies: selectedStrategies(),
        initial_quote: Number(document.getElementById('simInitialQuote').value || 1000),
        trade_quote_size: Number(cfgElement('trade_quote_size').value || 50),
        lookback: Number(document.getElementById('simLookback').value || 120),
        start_time: datetimeLocalToMs(document.getElementById('simStartTime').value),
        end_time: datetimeLocalToMs(document.getElementById('simEndTime').value)
      }};
    }}

    function tradeRunPayload() {{
      return {{
        exchange: document.getElementById('exchange').value,
        symbols: [document.getElementById('symbol').value.trim()].filter(Boolean),
        market_type: document.getElementById('tradeMarketType').value,
        timeframe: document.getElementById('timeframe').value,
        strategies: [document.getElementById('strategy').value],
        execute: document.getElementById('execute').checked,
        rounds: Number(document.getElementById('tradeRounds').value || 100),
        duration_seconds: secondsFromUnit('tradeDuration', 'tradeDurationUnit'),
        interval_seconds: secondsFromUnit('tradeInterval', 'tradeIntervalUnit') || 60,
        until_stopped: document.getElementById('tradeUntilStopped').checked
      }};
    }}

    function secondsFromUnit(inputId, unitId) {{
      return Number(document.getElementById(inputId).value || 0) * Number(document.getElementById(unitId).value || 1);
    }}

    function datetimeLocalToMs(value) {{
      if (!value) return null;
      const time = new Date(value).getTime();
      return Number.isFinite(time) ? time : null;
    }}

    function gridPayload() {{
      const lower = document.getElementById('gridLower').value;
      const upper = document.getElementById('gridUpper').value;
      return {{
        exchange: document.getElementById('exchange').value,
        symbol: document.getElementById('gridSymbol').value.trim(),
        market_type: document.getElementById('gridMarketType').value,
        strategy: document.getElementById('gridStrategy').value,
        spacing_mode: document.getElementById('gridSpacingMode').value,
        lower_price: lower ? Number(lower) : null,
        upper_price: upper ? Number(upper) : null,
        grid_count: Number(document.getElementById('gridCount').value || 12),
        investment_quote: Number(document.getElementById('gridInvestment').value || 500),
        leverage: Number(document.getElementById('gridLeverage').value || 1),
        ai_guidance: document.getElementById('gridAiGuidance').value.trim(),
        timeframe: document.getElementById('timeframe').value,
        execute: document.getElementById('gridExecute').checked,
        rounds: Number(document.getElementById('multiRounds').value || 10),
        interval_seconds: Number(document.getElementById('gridInterval').value || 2),
        summary_push_enabled: document.getElementById('gridSummaryPush').checked,
        summary_interval_seconds: 7200
      }};
    }}

    function renderBatch(data) {{
      const rows = [];
      for (const round of data.rounds || []) {{
        for (const item of round.items || []) {{
          if (item.error) {{
            rows.push(`<tr><td colspan="4">${{escapeHtml(item.error)}}</td></tr>`);
            continue;
          }}
          const plan = item.plan || {{}};
          const result = item.result || {{}};
          rows.push(`<tr><td>${{escapeHtml(item.symbol)}}</td><td>${{escapeHtml(item.strategy_label || strategyLabel(item.strategy))}}</td><td>${{escapeHtml(actionLabel(plan.action))}} / ${{Number(plan.confidence || 0).toFixed(2)}}</td><td>${{escapeHtml(result.status || '未执行')}}</td></tr>`);
        }}
      }}
      multiTable.innerHTML = rows.length ? rows.join('') : '<tr><td colspan="4">暂无结果</td></tr>';
      rawOutput.textContent = JSON.stringify(data, null, 2);
    }}

    function renderSimulation(data) {{
      const rows = [];
      for (const item of data.simulations || []) {{
        rows.push(`<tr><td>${{escapeHtml(item.symbol)}}</td><td>${{escapeHtml(item.strategy_label || strategyLabel(item.strategy))}}</td><td>${{Number(item.profit_pct || 0).toFixed(2)}}%</td><td>${{Number(item.final_equity || 0).toFixed(2)}}</td></tr>`);
      }}
      multiTable.innerHTML = rows.length ? rows.join('') : '<tr><td colspan="4">暂无结果</td></tr>';
      rawOutput.textContent = JSON.stringify(data, null, 2);
      if (data.simulations && data.simulations[0]) drawChart(data.simulations[0]);
      if (data.simulations && data.simulations[0]) {{
        const first = data.simulations[0];
        fields.chartCaption.textContent += `；时间段 ${{formatDate(first.start_time)}} - ${{formatDate(first.end_time)}}`;
      }}
    }}

    function formatDate(timestamp) {{
      return timestamp ? new Date(timestamp).toLocaleString() : '-';
    }}

    function renderRun(run) {{
      fields.runId.textContent = run.id || '-';
      fields.runState.textContent = run.status || '-';
      fields.runRound.textContent = run.round || 0;
      fields.runEventCount.textContent = (run.events || []).length;
      const coinSelection = run.coin_selection || {{}};
      const selectedSymbols = coinSelection.symbols || [];
      const coinText = selectedSymbols.length ? `；选币 ${{selectedSymbols.join(', ')}}` : '';
      fields.runStatus.textContent = (run.error || (run.status === 'running' ? '运行中' : run.status || '未开始')) + coinText;
      const events = run.events || [];
      runEventTable.innerHTML = events.length ? events.slice().reverse().map((event) => {{
        const cls = event.type === 'open' ? 'event-open' : event.type === 'close' ? 'event-close' : '';
        const type = event.type === 'open' ? '开仓' : event.type === 'close' ? '平仓' : '错误';
        return `<tr><td class="${{cls}}">${{escapeHtml(type)}}</td><td>${{escapeHtml(event.symbol || '-')}}</td><td>${{escapeHtml(event.strategy_label || strategyLabel(event.strategy))}}</td><td>${{escapeHtml(event.round || '-')}} / ${{Number(event.confidence || 0).toFixed(2)}}</td><td>${{escapeHtml(event.status || event.message || '-')}}</td></tr>`;
      }}).join('') : '<tr><td colspan="5">暂无开仓或平仓事件。</td></tr>';
      if (run.status !== 'running' && runPollTimer) {{
        clearInterval(runPollTimer);
        runPollTimer = null;
      }}
      stopTradeRunButton.disabled = !['running', 'stopping'].includes(run.status);
      startTradeRunButton.disabled = ['running', 'stopping'].includes(run.status);
    }}

    function renderBalances(data) {{
      fields.balanceExchange.textContent = data.exchange || '-';
      fields.balanceMarketType.textContent = data.market_type || '-';
      fields.balanceUpdatedAt.textContent = data.updated_at ? new Date(data.updated_at * 1000).toLocaleString() : '-';
      fields.balanceStatus.textContent = data.available ? '可用' : '不可用';
      fields.balanceMessage.textContent = data.message || '-';
      const rows = (data.assets || []).map((item) => {{
        const amount = Number(item.amount || 0).toLocaleString(undefined, {{ maximumFractionDigits: 8 }});
        return `<tr><td>${{escapeHtml(item.asset)}}</td><td>${{amount}}</td></tr>`;
      }});
      balanceTable.innerHTML = rows.length ? rows.join('') : '<tr><td colspan="2">没有读取到余额。</td></tr>';
    }}

    async function refreshBalances() {{
      refreshBalancesButton.disabled = true;
      refreshBalancesButton.textContent = '刷新中';
      fields.balanceStatus.textContent = '刷新中';
      try {{
        const exchange = document.getElementById('exchange').value || cfgElement('default_exchange').value || 'paper';
        const marketType = document.getElementById('tradeMarketType').value || document.getElementById('multiMarketType').value || 'spot';
        const params = new URLSearchParams({{ exchange, market_type: marketType }});
        const response = await fetch('/balances?' + params.toString());
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '余额刷新失败');
        renderBalances(data);
      }} catch (error) {{
        fields.balanceStatus.textContent = '刷新失败';
        fields.balanceMessage.textContent = String(error.message || error);
        balanceTable.innerHTML = '<tr><td colspan="2">余额读取失败，请检查交易所配置。</td></tr>';
      }} finally {{
        refreshBalancesButton.disabled = false;
        refreshBalancesButton.textContent = '刷新余额';
      }}
    }}

    function renderGridPlan(plan) {{
      fields.gridCurrentPrice.textContent = Number(plan.current_price || 0).toFixed(4);
      fields.gridEventCount.textContent = plan.price_source || '-';
      gridLevelsTable.innerHTML = (plan.levels || []).map((level) => {{
        const cls = level.action === 'buy' ? 'event-open' : 'event-close';
        return `<tr><td>${{level.index}}</td><td>${{Number(level.price).toFixed(4)}}</td><td class="${{cls}}">${{escapeHtml(actionLabel(level.action))}}</td><td>${{Number(level.quote_size).toFixed(2)}}</td><td>${{level.triggered ? '已触发' : '等待'}}</td></tr>`;
      }}).join('') || '<tr><td colspan="5">暂无网格。</td></tr>';
    }}

    function renderGridBot(bot) {{
      fields.gridBotId.textContent = bot.id || '-';
      fields.gridBotState.textContent = bot.status || '-';
      fields.gridStatus.textContent = bot.error || (bot.status === 'running' ? '运行中' : bot.status || '待命');
      if (bot.plan) renderGridPlan(bot.plan);
      fields.gridEventCount.textContent = `${{(bot.plan && bot.plan.price_source) || '-'}} / ${{(bot.events || []).length}}`;
      const events = bot.events || [];
      gridEventsTable.innerHTML = events.length ? events.slice().reverse().map((event) => {{
        const cls = event.action === 'buy' ? 'event-open' : 'event-close';
        const status = event.result ? event.result.status : 'planned';
        return `<tr><td>${{event.round}}</td><td>${{escapeHtml(event.symbol)}}</td><td class="${{cls}}">${{escapeHtml(actionLabel(event.action))}}</td><td>${{Number(event.level_price).toFixed(4)}} / ${{Number(event.market_price).toFixed(4)}}</td><td>${{escapeHtml(status)}}</td></tr>`;
      }}).join('') : '<tr><td colspan="5">暂无触发事件。</td></tr>';
      gridOutput.textContent = JSON.stringify(bot, null, 2);
      if (bot.status !== 'running' && gridPollTimer) {{
        clearInterval(gridPollTimer);
        gridPollTimer = null;
      }}
    }}

    async function pollGridBot() {{
      if (!currentGridBotId) return;
      const response = await fetch('/grid/bots/' + encodeURIComponent(currentGridBotId));
      const data = await readJsonOrText(response);
      if (!response.ok) throw new Error(data.detail || '网格状态读取失败');
      renderGridBot(data.bot);
    }}

    async function pollRun() {{
      if (!currentRunId) return;
      const response = await fetch('/runs/' + encodeURIComponent(currentRunId));
      const data = await readJsonOrText(response);
      if (!response.ok) throw new Error(data.detail || '运行状态读取失败');
      renderRun(data.run);
    }}

    function drawChart(simulation) {{
      const ctx = chartCanvas.getContext('2d');
      const width = chartCanvas.width;
      const height = chartCanvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      const candles = simulation.candles || [];
      const equity = simulation.equity_curve || [];
      if (!candles.length || !equity.length) return;
      drawSeries(ctx, candles.map((item) => item.close), '#0f766e', width, height, 24);
      drawSeries(ctx, equity.map((item) => item.equity), '#b45309', width, height, 24);
      ctx.fillStyle = '#17202a';
      ctx.font = '13px sans-serif';
      ctx.fillText(`${{simulation.symbol}} / ${{simulation.strategy_label || strategyLabel(simulation.strategy)}}`, 14, 20);
      ctx.fillStyle = '#0f766e';
      ctx.fillText('价格', width - 90, 20);
      ctx.fillStyle = '#b45309';
      ctx.fillText('权益', width - 45, 20);
      fields.chartCaption.textContent = `收益 ${{Number(simulation.profit || 0).toFixed(2)}}，收益率 ${{Number(simulation.profit_pct || 0).toFixed(2)}}%，交易 ${{simulation.trades.length}} 次`;
    }}

    function drawSeries(ctx, values, color, width, height, pad) {{
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min || 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((value, index) => {{
        const x = pad + (index / Math.max(values.length - 1, 1)) * (width - pad * 2);
        const y = height - pad - ((value - min) / span) * (height - pad * 2);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }}

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, (char) => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[char]));
    }}

    function strategyLabel(name) {{
      return strategyLabels[name] || name || '-';
    }}

    function gridStrategyLabel(name) {{
      return gridStrategyLabels[name] || name || '-';
    }}

    function actionLabel(action) {{
      return actionLabels[action] || action || '-';
    }}

    function updateProviderHint(offerFill) {{
      const provider = cfgElement('ai_provider').value;
      if (offerFill) {{
        populateAiMenus(provider, '', '');
      }} else {{
        toggleCustomConfigInputs();
      }}
    }}

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      runButton.disabled = true;
      runButton.textContent = '运行中';
      rawOutput.textContent = '正在请求 /run-once ...';
      const payload = {{
        exchange: document.getElementById('exchange').value,
        symbol: document.getElementById('symbol').value.trim(),
        market_type: document.getElementById('tradeMarketType').value,
        timeframe: document.getElementById('timeframe').value,
        strategy: document.getElementById('strategy').value,
        execute: document.getElementById('execute').checked
      }};
      try {{
        const response = await fetch('/run-once', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '请求失败');
        render(data);
      }} catch (error) {{
        rawOutput.textContent = String(error.message || error);
      }} finally {{
        runButton.disabled = false;
        runButton.textContent = '单次决策';
      }}
    }});

    clearButton.addEventListener('click', reset);
    refreshBalancesButton.addEventListener('click', refreshBalances);
    startTradeRunButton.addEventListener('click', async () => {{
      startTradeRunButton.disabled = true;
      stopTradeRunButton.disabled = false;
      fields.runStatus.textContent = '策略启动中';
      try {{
        const response = await fetch('/runs/start', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(tradeRunPayload())
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '启动策略运行失败');
        currentRunId = data.run_id;
        renderRun(data.run);
        if (runPollTimer) clearInterval(runPollTimer);
        runPollTimer = setInterval(() => pollRun().catch((error) => {{
          fields.runStatus.textContent = String(error.message || error);
        }}), 1200);
      }} catch (error) {{
        fields.runStatus.textContent = String(error.message || error);
        startTradeRunButton.disabled = false;
        stopTradeRunButton.disabled = true;
      }}
    }});
    stopTradeRunButton.addEventListener('click', async () => {{
      if (!currentRunId) return;
      stopTradeRunButton.disabled = true;
      fields.runStatus.textContent = '正在停止';
      try {{
        const response = await fetch('/runs/' + encodeURIComponent(currentRunId) + '/stop', {{ method: 'POST' }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '停止失败');
        renderRun(data.run);
      }} catch (error) {{
        fields.runStatus.textContent = String(error.message || error);
        stopTradeRunButton.disabled = false;
      }}
    }});
    batchRunButton.addEventListener('click', async () => {{
      batchRunButton.disabled = true;
      fields.multiStatus.textContent = '批量运行中';
      try {{
        const response = await fetch('/batch-run', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(multiPayload())
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '批量运行失败');
        renderBatch(data);
        fields.multiStatus.textContent = '批量运行完成';
      }} catch (error) {{
        fields.multiStatus.textContent = '批量运行失败';
        rawOutput.textContent = String(error.message || error);
      }} finally {{
        batchRunButton.disabled = false;
      }}
    }});
    startRunButton.addEventListener('click', async () => {{
      startRunButton.disabled = true;
      fields.runStatus.textContent = '启动中';
      try {{
        const response = await fetch('/runs/start', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(multiPayload())
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '启动失败');
        currentRunId = data.run_id;
        renderRun(data.run);
        if (runPollTimer) clearInterval(runPollTimer);
        runPollTimer = setInterval(() => pollRun().catch((error) => {{
          fields.runStatus.textContent = String(error.message || error);
        }}), 1200);
      }} catch (error) {{
        fields.runStatus.textContent = String(error.message || error);
      }} finally {{
        startRunButton.disabled = false;
      }}
    }});
    simulateButton.addEventListener('click', async () => {{
      simulateButton.disabled = true;
      fields.multiStatus.textContent = '模拟运行中';
      try {{
        const response = await fetch('/simulate', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(simulationPayload())
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '模拟失败');
        renderSimulation(data);
        fields.multiStatus.textContent = '模拟完成';
      }} catch (error) {{
        fields.multiStatus.textContent = '模拟失败';
        const message = String(error.message || error);
        multiTable.innerHTML = `<tr><td colspan="4">${{escapeHtml(message)}}</td></tr>`;
        fields.chartCaption.textContent = message;
        rawOutput.textContent = message;
      }} finally {{
        simulateButton.disabled = false;
      }}
    }});
    gridPlanButton.addEventListener('click', async () => {{
      gridPlanButton.disabled = true;
      fields.gridStatus.textContent = '生成中';
      try {{
        const response = await fetch('/grid/plan', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(gridPayload())
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '生成网格失败');
        renderGridPlan(data.plan);
        gridOutput.textContent = JSON.stringify(data, null, 2);
        fields.gridStatus.textContent = '网格已生成';
      }} catch (error) {{
        fields.gridStatus.textContent = String(error.message || error);
      }} finally {{
        gridPlanButton.disabled = false;
      }}
    }});
    gridStartButton.addEventListener('click', async () => {{
      gridStartButton.disabled = true;
      fields.gridStatus.textContent = '启动中';
      try {{
        const response = await fetch('/grid/start', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(gridPayload())
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '启动网格失败');
        currentGridBotId = data.bot_id;
        renderGridBot(data.bot);
        if (gridPollTimer) clearInterval(gridPollTimer);
        gridPollTimer = setInterval(() => pollGridBot().catch((error) => {{
          fields.gridStatus.textContent = String(error.message || error);
        }}), 1200);
      }} catch (error) {{
        fields.gridStatus.textContent = String(error.message || error);
      }} finally {{
        gridStartButton.disabled = false;
      }}
    }});
    loadTemplateButton.addEventListener('click', async () => {{
      fields.strategyUploadStatus.textContent = '加载模板中';
      try {{
        const response = await fetch('/strategy-template');
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '模板加载失败');
        strategyJson.value = JSON.stringify(data.template, null, 2);
        strategyUploadOutput.textContent = strategyJson.value;
        fields.strategyUploadStatus.textContent = '模板已载入';
      }} catch (error) {{
        fields.strategyUploadStatus.textContent = String(error.message || error);
      }}
    }});
    loadNofxTemplateButton.addEventListener('click', async () => {{
      fields.strategyUploadStatus.textContent = '加载 NOFX 模板中';
      try {{
        const response = await fetch('/strategy-template/nofx');
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || 'NOFX 模板加载失败');
        strategyJson.value = JSON.stringify(data.template, null, 2);
        strategyUploadOutput.textContent = strategyJson.value;
        fields.strategyUploadStatus.textContent = 'NOFX 模板已载入';
      }} catch (error) {{
        fields.strategyUploadStatus.textContent = String(error.message || error);
      }}
    }});
    strategyFileInput.addEventListener('change', async () => {{
      const file = strategyFileInput.files && strategyFileInput.files[0];
      if (!file) return;
      strategyJson.value = await file.text();
      strategyUploadOutput.textContent = strategyJson.value;
      fields.strategyUploadStatus.textContent = '文件已读取';
    }});
    uploadStrategyButton.addEventListener('click', async () => {{
      uploadStrategyButton.disabled = true;
      fields.strategyUploadStatus.textContent = '上传中';
      try {{
        const definition = JSON.parse(strategyJson.value);
        const response = await fetch('/strategies/upload', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ definition }})
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '上传失败');
        strategyLabels = data.labels || strategyLabels;
        strategyUploadOutput.textContent = JSON.stringify(data, null, 2);
        fields.strategyUploadStatus.textContent = '上传成功：' + strategyLabel(data.strategy);
        await refreshStrategies();
      }} catch (error) {{
        fields.strategyUploadStatus.textContent = String(error.message || error);
      }} finally {{
        uploadStrategyButton.disabled = false;
      }}
    }});
    cfgElement('ai_provider').addEventListener('change', () => updateProviderHint(true));
    testServerChanButton.addEventListener('click', async () => {{
      testServerChanButton.disabled = true;
      fields.serverChanTestStatus.textContent = '发送中';
      try {{
        const typedSendkey = cfgElement('serverchan_sendkey').value.trim();
        const payload = typedSendkey ? {{ sendkey: typedSendkey }} : {{}};
        const response = await fetch('/notifications/serverchan/test', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '方糖测试失败');
        fields.serverChanTestStatus.textContent = data.message || '测试已发送';
      }} catch (error) {{
        fields.serverChanTestStatus.textContent = String(error.message || error);
      }} finally {{
        testServerChanButton.disabled = false;
      }}
    }});
    settingsForm.addEventListener('submit', async (event) => {{
      event.preventDefault();
      saveSettingsButton.disabled = true;
      saveSettingsButton.textContent = '保存中';
      fields.settingsStatus.textContent = '保存中';
      try {{
        const response = await fetch('/settings', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(collectSettingsPayload())
        }});
        const data = await readJsonOrText(response);
        if (!response.ok) throw new Error(data.detail || '保存失败');
        applySettings(data.settings);
        fields.settingsStatus.textContent = '已保存并重载';
      }} catch (error) {{
        fields.settingsStatus.textContent = String(error.message || error);
      }} finally {{
        saveSettingsButton.disabled = false;
        saveSettingsButton.textContent = '保存配置';
      }}
    }});
    async function readJsonOrText(response) {{
      const text = await response.text();
      if (!text) return {{}};
      try {{
        return JSON.parse(text);
      }} catch (error) {{
        return {{ detail: text }};
      }}
    }}
    checkHealth();
    showCurrentPage();
    loadSettings();
    refreshStrategies().catch(() => {{}});
  </script>
</body>
</html>"""
