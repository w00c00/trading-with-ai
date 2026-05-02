from __future__ import annotations

import logging
from typing import Optional

from app.ai.decision import AIDecisionMaker
from app.config import Settings
from app.exchanges import CCXTExchange, ExchangeClient, PaperExchange
from app.models import Candle, MarketSnapshot, OrderResult, TradeAction, TradePlan
from app.notifications import ServerChanNotifier
from app.positions import PositionBook
from app.strategies import STRATEGIES

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, settings: Settings, ai: AIDecisionMaker) -> None:
        self.settings = settings
        self.ai = ai
        self.positions = PositionBook()
        self.notifier = ServerChanNotifier(settings)

    def build_exchange(self, exchange_name: Optional[str] = None, market_type: str = "spot") -> ExchangeClient:
        selected = exchange_name or self.settings.default_exchange
        if selected == "paper":
            return PaperExchange()
        return CCXTExchange(
            exchange_id=selected or self.settings.exchange_id,
            api_key=self.settings.exchange_api_key,
            secret=self.settings.exchange_secret,
            password=self.settings.exchange_password,
            sandbox=self.settings.exchange_sandbox,
            market_type=market_type,
        )

    async def snapshot(self, exchange: ExchangeClient, symbol: str, timeframe: str) -> MarketSnapshot:
        candles = await exchange.fetch_ohlcv(symbol, timeframe, limit=120)
        balances = await exchange.fetch_balances()
        return self.snapshot_from_candles(exchange.id, symbol, timeframe, candles, balances)

    def snapshot_from_candles(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        candles: list[Candle],
        balances: Optional[dict[str, float]] = None,
    ) -> MarketSnapshot:
        balances = balances or {}
        base, quote = symbol.split("/")
        last_price = candles[-1].close
        position_quote = balances.get(base, 0.0) * last_price
        return MarketSnapshot(
            exchange=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            last_price=last_price,
            balances={base: balances.get(base, 0.0), quote: balances.get(quote, 0.0)},
            position_quote=position_quote,
        )

    async def plan_trade(
        self,
        strategy_name: str = "strategy_ensemble",
        exchange_name: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        market_type: str = "spot",
    ) -> TradePlan:
        exchange = self.build_exchange(exchange_name, market_type=market_type)
        trading_symbol = symbol or self.settings.default_symbol
        trading_timeframe = timeframe or self.settings.default_timeframe
        snapshot = await self.snapshot(exchange, trading_symbol, trading_timeframe)
        strategy = STRATEGIES[strategy_name]
        signal = strategy.evaluate(snapshot)
        supporting_signals = [signal]
        if "signals" in signal.metadata:
            supporting_signals = []
            for raw in signal.metadata["signals"]:
                supporting_signals.append(signal.__class__.model_validate(raw))
        ai_decision = await self.ai.decide(snapshot, supporting_signals)
        action = _combine(signal.action, signal.confidence, ai_decision.action, ai_decision.confidence)
        confidence = max(signal.confidence, ai_decision.confidence) if action != TradeAction.hold else min(signal.confidence, ai_decision.confidence)
        dry_run = exchange.id != "paper" and not self.settings.live_trading_enabled
        decision_steps = [
            f"策略信号：{signal.action.value}，置信度 {signal.confidence:.2f}，原因：{signal.reason}",
            f"AI 复核：{ai_decision.action.value}，置信度 {ai_decision.confidence:.2f}，原因：{ai_decision.reason}",
            f"合并结果：{action.value}，最终置信度 {confidence:.2f}",
        ]
        if ai_decision.risk_notes:
            decision_steps.append("AI 风险提示：" + "；".join(ai_decision.risk_notes))
        plan = TradePlan(
            symbol=trading_symbol,
            action=action,
            quote_size=self.settings.trade_quote_size,
            confidence=confidence,
            reason=f"strategy={signal.reason}; ai={ai_decision.reason}",
            dry_run=dry_run,
            strategy_signal=signal.model_dump(mode="json"),
            ai_decision=ai_decision.model_dump(mode="json"),
            decision_steps=decision_steps,
        )
        return self._apply_risk(plan, snapshot.position_quote)

    async def execute_plan(self, plan: TradePlan, exchange_name: Optional[str] = None, market_type: str = "spot") -> OrderResult:
        exchange = self.build_exchange(exchange_name, market_type=market_type)
        if plan.blocked or plan.action == TradeAction.hold:
            return OrderResult(
                exchange=exchange.id,
                symbol=plan.symbol,
                action=plan.action,
                quote_size=plan.quote_size,
                status="blocked",
                detail={"reason": plan.block_reason or "hold"},
            )
        if plan.dry_run:
            result = OrderResult(
                exchange=exchange.id,
                symbol=plan.symbol,
                action=plan.action,
                quote_size=plan.quote_size,
                status="dry_run",
                detail={"reason": plan.reason},
            )
            if self.settings.notify_dry_run:
                await self._notify_success(result)
            return result
        result = await exchange.create_market_order(plan.symbol, plan.action, plan.quote_size)
        if result.status not in {"blocked", "rejected", "canceled", "cancelled", "failed"}:
            await self._notify_success(result)
        return result

    async def _notify_success(self, result: OrderResult) -> None:
        position = self.positions.apply_order(result)
        try:
            await self.notifier.send_trade_success(result, position)
        except Exception as exc:
            logger.warning("trade notification failed: %s", exc)

    def _apply_risk(self, plan: TradePlan, position_quote: float) -> TradePlan:
        if plan.action != TradeAction.hold and plan.confidence < self.settings.min_ai_confidence:
            plan.blocked = True
            plan.block_reason = f"confidence {plan.confidence:.2f} below threshold {self.settings.min_ai_confidence:.2f}"
            plan.decision_steps.append(f"风控拦截：{plan.block_reason}")
        if plan.action == TradeAction.buy and position_quote + plan.quote_size > self.settings.max_position_quote:
            plan.blocked = True
            plan.block_reason = f"position limit would exceed {self.settings.max_position_quote:.2f} quote"
            plan.decision_steps.append(f"风控拦截：{plan.block_reason}")
        return plan


def _combine(strategy_action: TradeAction, strategy_confidence: float, ai_action: TradeAction, ai_confidence: float) -> TradeAction:
    if strategy_action == ai_action:
        return strategy_action
    if ai_action == TradeAction.hold or ai_confidence < 0.55:
        return TradeAction.hold
    if strategy_confidence >= 0.72 and ai_confidence >= 0.62:
        return strategy_action
    return TradeAction.hold
