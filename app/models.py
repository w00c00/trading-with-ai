from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TradeAction(str, Enum):
    buy = "buy"
    sell = "sell"
    hold = "hold"


class Candle(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketSnapshot(BaseModel):
    exchange: str
    symbol: str
    timeframe: str
    candles: list[Candle]
    last_price: float
    balances: dict[str, float] = Field(default_factory=dict)
    position_quote: float = 0.0


class StrategySignal(BaseModel):
    strategy: str
    action: TradeAction
    confidence: float = Field(ge=0, le=1)
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIDecision(BaseModel):
    action: TradeAction
    confidence: float = Field(ge=0, le=1)
    reason: str
    risk_notes: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class TradePlan(BaseModel):
    symbol: str
    action: TradeAction
    quote_size: float
    confidence: float
    reason: str
    dry_run: bool
    blocked: bool = False
    block_reason: Optional[str] = None
    strategy_signal: dict[str, Any] = Field(default_factory=dict)
    ai_decision: dict[str, Any] = Field(default_factory=dict)
    decision_steps: list[str] = Field(default_factory=list)
    execution_intent: dict[str, Any] = Field(default_factory=dict)


class OrderResult(BaseModel):
    exchange: str
    symbol: str
    action: TradeAction
    quote_size: float
    status: str
    order_id: Optional[str] = None
    detail: dict[str, Any] = Field(default_factory=dict)


class PositionUpdate(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    action: TradeAction
    filled_amount: float = 0.0
    average_price: float = 0.0
    trade_cost: float = 0.0
    position_amount: float = 0.0
    average_cost: float = 0.0
    position_cost: float = 0.0
    realized_profit: Optional[float] = None
    realized_profit_pct: Optional[float] = None
