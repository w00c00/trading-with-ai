from __future__ import annotations

import logging
from typing import Any, Optional

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
        execution_intent = signal.metadata.get("execution_intent", {}) if isinstance(signal.metadata, dict) else {}
        limits = self._effective_risk_limits(signal.metadata, snapshot)
        decision_steps = [
            f"策略信号：{signal.action.value}，置信度 {signal.confidence:.2f}，原因：{signal.reason}",
            f"AI 复核：{ai_decision.action.value}，置信度 {ai_decision.confidence:.2f}，原因：{ai_decision.reason}",
            f"合并结果：{action.value}，最终置信度 {confidence:.2f}",
            "资金风控："
            f"{'策略内部限制优先' if limits['priority'] == 'strategy' else '全局限制优先'}"
            f"，单次金额={limits['trade_quote_size']:.2f}"
            f"，最大持仓={limits['max_position_quote']:.2f}"
            f"，最低置信度={limits['min_confidence']:.2f}",
        ]
        if execution_intent:
            intent_name = execution_intent.get("intent") or execution_intent.get("mode") or "-"
            leverage = execution_intent.get("leverage")
            stop_loss = execution_intent.get("stop_loss")
            take_profit = execution_intent.get("take_profit")
            decision_steps.append(
                "执行意图："
                f"{intent_name}"
                f"，杠杆={leverage if leverage is not None else '-'}"
                f"，止损={stop_loss if stop_loss is not None else '-'}"
                f"，止盈={take_profit if take_profit is not None else '-'}"
            )
        if ai_decision.risk_notes:
            decision_steps.append("AI 风险提示：" + "；".join(ai_decision.risk_notes))
        plan = TradePlan(
            symbol=trading_symbol,
            action=action,
            quote_size=limits["trade_quote_size"],
            confidence=confidence,
            reason=f"strategy={signal.reason}; ai={ai_decision.reason}",
            dry_run=dry_run,
            strategy_signal=signal.model_dump(mode="json"),
            ai_decision=ai_decision.model_dump(mode="json"),
            decision_steps=decision_steps,
            execution_intent=execution_intent,
        )
        return self._apply_risk(plan, snapshot.position_quote, market_type, limits)

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
        execution_intent = plan.execution_intent or {}
        contract_intent = bool(execution_intent.get("requires_contract"))
        if market_type == "spot" and contract_intent:
            return OrderResult(
                exchange=exchange.id,
                symbol=plan.symbol,
                action=plan.action,
                quote_size=plan.quote_size,
                status="blocked",
                detail={"reason": "contract execution intent cannot be submitted in spot mode", "execution_intent": execution_intent},
            )
        if contract_intent and execution_intent.get("requires_hard_stop_loss") and not execution_intent.get("stop_loss"):
            return OrderResult(
                exchange=exchange.id,
                symbol=plan.symbol,
                action=plan.action,
                quote_size=plan.quote_size,
                status="blocked",
                detail={"reason": "contract execution requires a hard stop_loss before submitting live or paper order", "execution_intent": execution_intent},
            )
        preflight_error = _contract_preflight_error(plan)
        if preflight_error:
            return OrderResult(
                exchange=exchange.id,
                symbol=plan.symbol,
                action=plan.action,
                quote_size=plan.quote_size,
                status="blocked",
                detail={"reason": preflight_error, "execution_intent": execution_intent},
            )
        if plan.dry_run:
            result = OrderResult(
                exchange=exchange.id,
                symbol=plan.symbol,
                action=plan.action,
                quote_size=plan.quote_size,
                status="dry_run",
                detail={"reason": plan.reason, "execution_intent": execution_intent, "market_type": market_type},
            )
            if self.settings.notify_dry_run:
                await self._notify_success(result)
            return result
        result = await exchange.create_plan_order(plan.symbol, plan.action, plan.quote_size, execution_intent, market_type)
        if result.status not in {"blocked", "rejected", "canceled", "cancelled", "failed"}:
            await self._notify_success(result)
        return result

    async def _notify_success(self, result: OrderResult) -> None:
        position = self.positions.apply_order(result)
        try:
            await self.notifier.send_trade_success(result, position)
        except Exception as exc:
            logger.warning("trade notification failed: %s", exc)

    def _apply_risk(self, plan: TradePlan, position_quote: float, market_type: str = "spot", limits: Optional[dict[str, float]] = None) -> TradePlan:
        limits = limits or {
            "min_confidence": self.settings.min_ai_confidence,
            "max_position_quote": self.settings.max_position_quote,
        }
        min_confidence = limits["min_confidence"]
        max_position_quote = limits["max_position_quote"]
        if plan.action != TradeAction.hold and plan.confidence < min_confidence:
            plan.blocked = True
            plan.block_reason = f"confidence {plan.confidence:.2f} below threshold {min_confidence:.2f}"
            plan.decision_steps.append(f"风控拦截：{plan.block_reason}")
        if plan.action == TradeAction.buy and position_quote + plan.quote_size > max_position_quote:
            plan.blocked = True
            plan.block_reason = f"position limit would exceed {max_position_quote:.2f} quote"
            plan.decision_steps.append(f"风控拦截：{plan.block_reason}")
        if market_type == "spot" and plan.execution_intent.get("requires_contract"):
            plan.decision_steps.append("提示：该策略包含合约语义；现货模式下会降级为买入/卖出/观望信号。")
        return plan

    def _effective_risk_limits(self, metadata: dict[str, Any], snapshot: MarketSnapshot) -> dict[str, Any]:
        global_limits = {
            "priority": "global",
            "trade_quote_size": self.settings.trade_quote_size,
            "max_position_quote": self.settings.max_position_quote,
            "min_confidence": self.settings.min_ai_confidence,
        }
        if self.settings.risk_limit_priority != "strategy":
            return global_limits
        strategy_limits = _strategy_risk_limits(metadata, snapshot)
        return {
            "priority": "strategy",
            "trade_quote_size": strategy_limits.get("trade_quote_size") or global_limits["trade_quote_size"],
            "max_position_quote": strategy_limits.get("max_position_quote") or global_limits["max_position_quote"],
            "min_confidence": strategy_limits.get("min_confidence") or global_limits["min_confidence"],
        }


def _combine(strategy_action: TradeAction, strategy_confidence: float, ai_action: TradeAction, ai_confidence: float) -> TradeAction:
    if strategy_action == ai_action:
        return strategy_action
    if ai_action == TradeAction.hold or ai_confidence < 0.55:
        return TradeAction.hold
    if strategy_confidence >= 0.72 and ai_confidence >= 0.62:
        return strategy_action
    return TradeAction.hold


def _contract_preflight_error(plan: TradePlan) -> Optional[str]:
    intent = plan.execution_intent or {}
    if not intent.get("requires_contract"):
        return None
    side = str(intent.get("position_side", "")).lower()
    stop_loss = _float_or_none(intent.get("stop_loss"))
    take_profit = _float_or_none(intent.get("take_profit"))
    reference_price = _float_or_none(intent.get("reference_price"))
    leverage = _float_or_none(intent.get("leverage"))
    max_leverage = _float_or_none(intent.get("max_leverage"))
    if max_leverage is not None and leverage is not None and leverage > max_leverage:
        return f"leverage {leverage:g} exceeds strategy max {max_leverage:g}"
    if reference_price is None:
        return None
    if side == "long":
        if stop_loss is not None and stop_loss >= reference_price:
            return "long stop_loss must be below reference price"
        if take_profit is not None and take_profit <= reference_price:
            return "long take_profit must be above reference price"
    if side == "short":
        if stop_loss is not None and stop_loss <= reference_price:
            return "short stop_loss must be above reference price"
        if take_profit is not None and take_profit >= reference_price:
            return "short take_profit must be below reference price"
    return None


def _strategy_risk_limits(metadata: dict[str, Any], snapshot: MarketSnapshot) -> dict[str, float]:
    if not isinstance(metadata, dict):
        return {}
    risk = metadata.get("nofx_risk_control") if isinstance(metadata.get("nofx_risk_control"), dict) else {}
    execution = metadata.get("execution_intent") if isinstance(metadata.get("execution_intent"), dict) else {}
    matched_rule = metadata.get("matched_rule") if isinstance(metadata.get("matched_rule"), dict) else {}
    sources = [matched_rule, execution, risk]
    trade_quote_size = _first_positive(
        sources,
        "trade_quote_size",
        "quote_size",
        "order_quote_size",
        "position_size_usd",
        "base_order_size",
        "min_position_size",
    )
    max_position_quote = _first_positive(
        sources,
        "max_position_quote",
        "max_position_usd",
        "max_position_size_usd",
        "max_position_value",
    )
    if max_position_quote is None:
        max_position_quote = _ratio_position_limit(snapshot, risk)
    min_confidence = _normalize_confidence(_first_value(sources, "min_confidence", "minConfidence", "minimum_confidence"))
    limits: dict[str, float] = {}
    if trade_quote_size is not None:
        limits["trade_quote_size"] = trade_quote_size
    if max_position_quote is not None:
        limits["max_position_quote"] = max_position_quote
    if min_confidence is not None:
        limits["min_confidence"] = min_confidence
    return limits


def _ratio_position_limit(snapshot: MarketSnapshot, risk: dict[str, Any]) -> Optional[float]:
    base, quote = snapshot.symbol.split("/", 1)
    ratio_key = "btc_eth_max_position_value_ratio" if base.upper() in {"BTC", "ETH"} else "altcoin_max_position_value_ratio"
    ratio = _float_or_none(risk.get(ratio_key))
    quote_balance = _float_or_none(snapshot.balances.get(quote))
    if ratio is None or quote_balance is None or quote_balance <= 0:
        return None
    fraction = ratio if ratio <= 1 else ratio / 100
    return quote_balance * fraction


def _first_positive(sources: list[dict[str, Any]], *keys: str) -> Optional[float]:
    value = _first_value(sources, *keys)
    parsed = _float_or_none(value)
    return parsed if parsed is not None and parsed > 0 else None


def _first_value(sources: list[dict[str, Any]], *keys: str) -> Any:
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
        lowered = {str(key).lower(): value for key, value in source.items()}
        for key in keys:
            if key.lower() in lowered:
                return lowered[key.lower()]
    return None


def _normalize_confidence(value: Any) -> Optional[float]:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    if parsed > 1:
        parsed /= 100
    return min(max(parsed, 0.0), 1.0)


def _float_or_none(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
