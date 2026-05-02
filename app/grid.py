from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.ai.providers import AIClient
from app.models import Candle, TradeAction

GridStrategyName = Literal["neutral_range", "trend_follow", "volatility_adaptive", "ai_adaptive"]
MarketType = Literal["spot", "swap", "future"]
GridSpacingMode = Literal["arithmetic", "geometric"]


class GridPlanRequest(BaseModel):
    exchange: Optional[str] = None
    symbol: str = "BTC/USDT"
    market_type: MarketType = "spot"
    strategy: GridStrategyName = "neutral_range"
    spacing_mode: GridSpacingMode = "arithmetic"
    lower_price: Optional[float] = Field(default=None, gt=0)
    upper_price: Optional[float] = Field(default=None, gt=0)
    grid_count: int = Field(default=12, ge=3, le=100)
    investment_quote: float = Field(default=500.0, gt=0)
    leverage: float = Field(default=1.0, ge=1, le=125)
    ai_guidance: str = ""
    timeframe: str = "1h"
    execute: bool = False
    rounds: int = Field(default=10, ge=1, le=500)
    interval_seconds: float = Field(default=2.0, ge=0, le=3600)
    summary_push_enabled: bool = False
    summary_interval_seconds: int = Field(default=7200, ge=60, le=86_400)


class GridLevel(BaseModel):
    index: int
    price: float
    action: TradeAction
    quote_size: float
    triggered: bool = False


class GridPlan(BaseModel):
    symbol: str
    market_type: MarketType
    strategy: GridStrategyName
    spacing_mode: GridSpacingMode
    price_source: str = "exchange_ohlcv"
    price_warning: str = ""
    lower_price: float
    upper_price: float
    grid_count: int
    current_price: float
    investment_quote: float
    leverage: float
    quote_per_grid: float
    levels: list[GridLevel]
    reason: str
    ai_notes: list[str] = Field(default_factory=list)


GRID_STRATEGIES: dict[str, dict[str, str]] = {
    "neutral_range": {
        "name": "neutral_range",
        "label": "中性区间网格",
        "description": "围绕当前价格上下铺网格，适合震荡行情。",
    },
    "trend_follow": {
        "name": "trend_follow",
        "label": "趋势跟随网格",
        "description": "根据近端趋势轻微上移或下移区间，适合缓慢趋势行情。",
    },
    "volatility_adaptive": {
        "name": "volatility_adaptive",
        "label": "波动自适应网格",
        "description": "根据近期高低点扩大网格边界，适合波动变大的行情。",
    },
    "ai_adaptive": {
        "name": "ai_adaptive",
        "label": "AI 自适应网格",
        "description": "AI 根据行情和你的说明建议区间、格数和偏向，系统再做风控裁剪。",
    },
}


async def build_grid_plan(request: GridPlanRequest, candles: list[Candle], ai_client: AIClient) -> GridPlan:
    current = candles[-1].close
    lower, upper, count, reason, ai_notes = await _choose_bounds(request, candles, ai_client)
    lower = max(0.00000001, lower)
    if upper <= lower:
        upper = lower * 1.02
    quote_per_grid = request.investment_quote / count
    levels = []
    prices = _grid_prices(lower, upper, count, request.spacing_mode)
    for index, price in enumerate(prices):
        action = TradeAction.buy if price < current else TradeAction.sell
        levels.append(GridLevel(index=index + 1, price=price, action=action, quote_size=quote_per_grid))
    return GridPlan(
        symbol=request.symbol,
        market_type=request.market_type,
        strategy=request.strategy,
        spacing_mode=request.spacing_mode,
        price_source="paper_estimate" if request.exchange == "paper" else "exchange_ohlcv",
        price_warning="Paper prices are synthetic estimates; use a real exchange for live KAS pricing." if request.exchange == "paper" else "",
        lower_price=lower,
        upper_price=upper,
        grid_count=count,
        current_price=current,
        investment_quote=request.investment_quote,
        leverage=request.leverage,
        quote_per_grid=quote_per_grid,
        levels=levels,
        reason=reason,
        ai_notes=ai_notes,
    )


def _grid_prices(lower: float, upper: float, count: int, spacing_mode: GridSpacingMode) -> list[float]:
    if count <= 1:
        return [lower]
    if spacing_mode == "geometric" and lower > 0 and upper > 0:
        ratio = (upper / lower) ** (1 / (count - 1))
        return [lower * (ratio**index) for index in range(count)]
    step = (upper - lower) / (count - 1)
    return [lower + step * index for index in range(count)]


async def _choose_bounds(request: GridPlanRequest, candles: list[Candle], ai_client: AIClient) -> tuple[float, float, int, str, list[str]]:
    current = candles[-1].close
    closes = [c.close for c in candles]
    recent = candles[-48:] if len(candles) >= 48 else candles
    recent_high = max(c.high for c in recent)
    recent_low = min(c.low for c in recent)
    count = request.grid_count
    if request.lower_price and request.upper_price:
        return request.lower_price, request.upper_price, count, "Manual grid bounds.", []
    if request.strategy == "trend_follow":
        fast = sum(closes[-12:]) / min(len(closes), 12)
        slow = sum(closes[-36:]) / min(len(closes), 36)
        shift = 0.035 if fast >= slow else -0.035
        lower = current * (0.94 + shift)
        upper = current * (1.06 + shift)
        return lower, upper, count, f"Trend-follow grid: SMA12={fast:.4f}, SMA36={slow:.4f}.", []
    if request.strategy == "volatility_adaptive":
        padding = max((recent_high - recent_low) * 0.18, current * 0.015)
        return recent_low - padding, recent_high + padding, count, "Volatility-adaptive grid from recent high/low range.", []
    if request.strategy == "ai_adaptive":
        lower, upper, ai_count, notes = await _ask_ai_for_grid(request, candles, ai_client)
        return lower, upper, ai_count, "AI adaptive grid with bounded parameters.", notes
    return current * 0.94, current * 1.06, count, "Neutral range grid around current price.", []


async def _ask_ai_for_grid(request: GridPlanRequest, candles: list[Candle], ai_client: AIClient) -> tuple[float, float, int, list[str]]:
    current = candles[-1].close
    base_asset, quote_asset = _split_symbol(request.symbol)
    recent = candles[-48:] if len(candles) >= 48 else candles
    recent_high = max(c.high for c in recent)
    recent_low = min(c.low for c in recent)
    messages = [
        {
            "role": "system",
            "content": (
                "You design conservative crypto grid bot parameters. Return strict JSON with keys: "
                "lower_price, upper_price, grid_count, notes. Keep the range near the current market. "
                "Use only the symbol supplied by the user. Do not assume BTC or mention BTC unless the supplied symbol is BTC."
            ),
        },
        {
            "role": "user",
            "content": (
                f"symbol={request.symbol}, base_asset={base_asset}, quote_asset={quote_asset}, market_type={request.market_type}, current_{request.symbol}={current}, "
                f"recent_low={recent_low}, recent_high={recent_high}, requested_grids={request.grid_count}, "
                f"user_guidance={request.ai_guidance or 'none'}"
            ),
        },
    ]
    try:
        data = await ai_client.chat_json(messages, temperature=0.2)
    except Exception as exc:
        return current * 0.94, current * 1.06, request.grid_count, [f"AI grid guidance failed; fallback neutral grid used: {type(exc).__name__}: {exc}"]
    lower = _float_or(data.get("lower_price"), current * 0.94)
    upper = _float_or(data.get("upper_price"), current * 1.06)
    count = int(_float_or(data.get("grid_count"), request.grid_count))
    count = min(max(count, 3), 100)
    lower = min(max(lower, current * 0.75), current * 0.995)
    upper = max(min(upper, current * 1.25), current * 1.005)
    notes = data.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if not notes:
        notes = [str(data.get("reason", "AI guidance unavailable; fallback bounds applied."))]
    return lower, upper, count, list(notes)


def _float_or(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _split_symbol(symbol: str) -> tuple[str, str]:
    parts = symbol.split("/")
    if len(parts) >= 2:
        return parts[0].upper(), parts[1].upper()
    return symbol.upper(), ""
