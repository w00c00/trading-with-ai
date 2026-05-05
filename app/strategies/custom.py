from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Any, Optional

from app.models import MarketSnapshot, StrategySignal, TradeAction
from app.strategies.base import Strategy
from app.strategies.indicators import rsi, sma

CUSTOM_STRATEGY_DIR = Path(os.getenv("TRADAI_STRATEGY_DIR", ".config/strategies"))


class RuleStrategy(Strategy):
    def __init__(self, definition: dict[str, Any]) -> None:
        self.definition = validate_strategy_definition(normalize_strategy_definition(definition))
        self.name = self.definition["name"]

    def evaluate(self, snapshot: MarketSnapshot) -> StrategySignal:
        common_metadata = _definition_metadata(self.definition)
        for rule in self.definition["rules"]:
            matched, reason = _rule_matches(rule, snapshot)
            if matched:
                metadata = dict(common_metadata)
                metadata["matched_rule"] = rule
                metadata["execution_intent"] = _execution_intent(self.definition, rule, snapshot)
                return StrategySignal(
                    strategy=self.name,
                    action=TradeAction(rule["action"]),
                    confidence=float(rule.get("confidence", 0.6)),
                    reason=reason,
                    metadata=metadata,
                )
        metadata = dict(common_metadata)
        metadata["execution_intent"] = _execution_intent(self.definition, None, snapshot)
        return StrategySignal(
            strategy=self.name,
            action=TradeAction(self.definition.get("default_action", "hold")),
            confidence=float(self.definition.get("default_confidence", 0.45)),
            reason="No custom rule matched.",
            metadata=metadata,
        )


def strategy_template() -> dict[str, Any]:
    return {
        "name": "custom_rsi_sma_template",
        "display_name": "自定义 RSI 均线策略",
        "description": "RSI 超买超卖结合 SMA 趋势的规则策略模板。",
        "default_action": "hold",
        "default_confidence": 0.45,
        "rules": [
            {"indicator": "rsi", "period": 14, "operator": "<", "value": 30, "action": "buy", "confidence": 0.72},
            {"indicator": "rsi", "period": 14, "operator": ">", "value": 70, "action": "sell", "confidence": 0.72},
            {"indicator": "sma_cross", "fast": 12, "slow": 36, "direction": "above", "action": "buy", "confidence": 0.66},
            {"indicator": "sma_cross", "fast": 12, "slow": 36, "direction": "below", "action": "sell", "confidence": 0.66},
        ],
    }


def nofx_strategy_template() -> dict[str, Any]:
    return {
        "strategyName": "nofx_rsi_ema_template",
        "displayName": "NOFX RSI EMA 策略",
        "description": "NOFX StrategyConfig 兼容模板，导入后会转换为本系统可运行的规则策略。",
        "coinSource": {"type": "manual", "symbols": ["BTC/USDT"]},
        "indicators": {
            "enableRSI": True,
            "rsiPeriods": [14],
            "enableEMA": True,
            "emaPeriods": [12, 36],
            "enableATR": False,
        },
        "riskControl": {
            "minConfidence": 0.62,
            "maxPositionPercent": 20,
            "stopLossPercent": 3,
            "takeProfitPercent": 6,
        },
        "promptSections": {
            "market": True,
            "technical": True,
            "risk": True,
        },
    }


def normalize_strategy_definition(definition: dict[str, Any]) -> dict[str, Any]:
    if isinstance(definition, dict) and "rules" in definition:
        return _ensure_safe_strategy_name(definition)
    if _looks_like_nofx_strategy(definition):
        definition = _convert_nofx_strategy(definition)
    return _ensure_safe_strategy_name(definition)


def validate_strategy_definition(definition: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(definition, dict):
        raise ValueError("Strategy definition must be a JSON object.")
    name = str(definition.get("name", "")).strip()
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{2,48}", name):
        raise ValueError("Strategy name must be 3-49 chars: letters, numbers, underscore, starting with a letter.")
    rules = definition.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("Strategy rules must be a non-empty array.")
    clean_rules = []
    for rule in rules:
        clean_rules.append(_validate_rule(rule))
    return {
        "name": name,
        "display_name": str(definition.get("display_name", ""))[:80],
        "description": str(definition.get("description", ""))[:300],
        "default_action": _validate_action(definition.get("default_action", "hold")),
        "default_confidence": _validate_confidence(definition.get("default_confidence", 0.45)),
        "rules": clean_rules,
        "source_format": str(definition.get("source_format", "tradai"))[:40],
        "nofx_config": _redact_sensitive(definition.get("nofx_config", {})),
        "nofx_prompt_sections": _redact_sensitive(definition.get("nofx_prompt_sections", {})),
        "nofx_coin_source": _redact_sensitive(definition.get("nofx_coin_source", {})),
        "nofx_risk_control": _redact_sensitive(definition.get("nofx_risk_control", {})),
        "nofx_execution": _redact_sensitive(definition.get("nofx_execution", {})),
        "nofx_indicators": _redact_sensitive(definition.get("nofx_indicators", {})),
    }


def _looks_like_nofx_strategy(definition: dict[str, Any]) -> bool:
    lowered = {str(key).lower() for key in definition}
    config = definition.get("config") if isinstance(definition.get("config"), dict) else {}
    config_lowered = {str(key).lower() for key in config}
    nofx_keys = {
        "coin_source",
        "coinsource",
        "coin_config",
        "coinconfig",
        "indicators",
        "risk_control",
        "riskcontrol",
        "risk",
        "prompt_sections",
        "promptsections",
        "prompts",
        "strategy_config",
        "strategyconfig",
        "strategyname",
        "strategy_name",
        "source_format",
        "sourceformat",
    }
    source_format = str(_get_any(definition, "source_format", "sourceFormat", "format") or "").lower()
    return source_format == "nofx" or bool(lowered & nofx_keys) or bool(config_lowered & nofx_keys)


def _ensure_safe_strategy_name(definition: dict[str, Any]) -> dict[str, Any]:
    clean = dict(definition)
    raw_name = str(_get_any(clean, "name", "strategyName", "strategy_name", "id") or "").strip()
    display_name = str(_get_any(clean, "display_name", "displayName", "title") or "").strip()
    if raw_name and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{2,48}", raw_name):
        clean["name"] = raw_name
    else:
        clean["name"] = _slugify(raw_name or display_name or "custom_strategy")
        if not display_name and raw_name:
            display_name = raw_name
    if display_name and not clean.get("display_name"):
        clean["display_name"] = display_name
    return clean


def _convert_nofx_strategy(definition: dict[str, Any]) -> dict[str, Any]:
    config = definition.get("config") if isinstance(definition.get("config"), dict) else {}
    source = config or definition
    coin_source = _get_any(source, "coinSource", "coin_source", "coinConfig", "coin_config") or {}
    indicators = _get_any(source, "indicators", "Indicators") or {}
    risk = _get_any(source, "riskControl", "risk_control", "RiskControl") or {}
    prompts = _get_any(source, "promptSections", "prompt_sections", "prompts") or {}
    strategy_type = str(_get_any(source, "strategy_type", "strategyType") or "ai_trading")
    raw_name = str(_get_any(definition, "name", "strategyName", "strategy_name", "id") or "nofx_imported_strategy")
    name = _slugify(raw_name)
    display_name = str(_get_any(definition, "display_name", "displayName", "title", "strategyName", "name") or f"NOFX 策略：{name}")[:80]
    default_confidence = _normalize_confidence(_get_any(risk, "minConfidence", "min_confidence", "MinConfidence") or 0.55)
    rules = []

    if _boolish(_get_any(indicators, "enableRSI", "enable_rsi", "RSI", "rsi"), default=True):
        rsi_periods = _as_number_list(_get_any(indicators, "rsiPeriods", "rsi_periods", "RSIPeriods"), [14])
        period = int(rsi_periods[0])
        rules.extend(
            [
                {
                    "indicator": "rsi",
                    "period": period,
                    "operator": "<",
                    "value": 30,
                    "action": "buy",
                    "confidence": max(default_confidence, 0.68),
                    "intent": "open_long_or_dca_long",
                    "position_side": "long",
                },
                {
                    "indicator": "rsi",
                    "period": period,
                    "operator": ">",
                    "value": 70,
                    "action": "sell",
                    "confidence": max(default_confidence, 0.68),
                    "intent": "open_short_or_dca_short",
                    "position_side": "short",
                },
            ]
        )

    if _boolish(_get_any(indicators, "enableEMA", "enable_ema", "EMA", "ema"), default=True):
        ema_periods = sorted(_as_number_list(_get_any(indicators, "emaPeriods", "ema_periods", "EMAPeriods"), [12, 36]))
        if len(ema_periods) >= 2:
            fast, slow = int(ema_periods[0]), int(ema_periods[-1])
            rules.extend(
                [
                    {
                        "indicator": "sma_cross",
                        "fast": fast,
                        "slow": slow,
                        "direction": "above",
                        "action": "buy",
                        "confidence": default_confidence,
                        "intent": "open_long",
                        "position_side": "long",
                    },
                    {
                        "indicator": "sma_cross",
                        "fast": fast,
                        "slow": slow,
                        "direction": "below",
                        "action": "sell",
                        "confidence": default_confidence,
                        "intent": "open_short",
                        "position_side": "short",
                    },
                ]
            )

    if _boolish(_get_any(indicators, "enableBOLL", "enable_boll", "BOLL", "boll"), default=False):
        boll_periods = _as_number_list(_get_any(indicators, "bollPeriods", "boll_periods", "BOLLPeriods"), [20])
        period = int(boll_periods[0])
        rules.extend(
            [
                {
                    "indicator": "price_vs_sma",
                    "period": period,
                    "operator": ">",
                    "multiplier": 1.01,
                    "action": "buy",
                    "confidence": default_confidence,
                    "intent": "breakout_follow_long",
                    "position_side": "long",
                },
                {
                    "indicator": "price_vs_sma",
                    "period": period,
                    "operator": "<",
                    "multiplier": 0.99,
                    "action": "sell",
                    "confidence": default_confidence,
                    "intent": "breakout_follow_short",
                    "position_side": "short",
                },
            ]
        )

    if not rules:
        rules.append(
            {
                "indicator": "sma_cross",
                "fast": 12,
                "slow": 36,
                "direction": "above",
                "action": "buy",
                "confidence": default_confidence,
                "intent": "open_long",
                "position_side": "long",
            }
        )

    execution = _nofx_execution_profile(strategy_type, indicators, risk, definition, prompts)

    return {
        "name": name,
        "display_name": display_name,
        "description": str(definition.get("description", "NOFX StrategyConfig converted for Trading with AI."))[:300],
        "default_action": "hold",
        "default_confidence": min(default_confidence, 0.5),
        "rules": rules,
        "source_format": "nofx",
        "nofx_config": _redact_sensitive(definition),
        "nofx_prompt_sections": prompts,
        "nofx_coin_source": coin_source,
        "nofx_risk_control": risk,
        "nofx_execution": execution,
        "nofx_indicators": indicators,
    }


def _get_any(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    lowered = {str(key).lower(): value for key, value in source.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _definition_metadata(definition: dict[str, Any]) -> dict[str, Any]:
    return {
        "custom": True,
        "description": definition.get("description", ""),
        "source_format": definition.get("source_format", "tradai"),
        "nofx_config": definition.get("nofx_config", {}),
        "nofx_prompt_sections": definition.get("nofx_prompt_sections", {}),
        "nofx_coin_source": definition.get("nofx_coin_source", {}),
        "nofx_risk_control": definition.get("nofx_risk_control", {}),
        "nofx_execution": definition.get("nofx_execution", {}),
        "nofx_indicators": definition.get("nofx_indicators", {}),
    }


def _execution_intent(definition: dict[str, Any], rule: Optional[dict[str, Any]], snapshot: MarketSnapshot) -> dict[str, Any]:
    execution = definition.get("nofx_execution") or {}
    risk = definition.get("nofx_risk_control") or {}
    rule = rule or {}
    side = str(rule.get("position_side", "none"))
    action = str(rule.get("action", definition.get("default_action", "hold")))
    intent = str(rule.get("intent") or _default_intent(action, side))
    atr_value = _atr(snapshot.candles, int(_first_number(execution.get("atr_periods"), 14)))
    stop_distance = atr_value * float(execution.get("stop_loss_atr_multiple", 1.5) or 1.5)
    take_distance = stop_distance * float(execution.get("min_risk_reward_ratio", 1.8) or 1.8)
    price = snapshot.last_price
    if side == "short":
        stop_loss = price + stop_distance if stop_distance else None
        take_profit = price - take_distance if take_distance else None
    elif side == "long":
        stop_loss = price - stop_distance if stop_distance else None
        take_profit = price + take_distance if take_distance else None
    else:
        stop_loss = None
        take_profit = None
    return {
        "intent": intent,
        "position_side": side,
        "requires_contract": bool(execution.get("requires_contract", False)),
        "allowed_actions": execution.get("allowed_actions", []),
        "leverage": _leverage_for_symbol(snapshot.symbol, risk),
        "max_positions": risk.get("max_positions"),
        "max_dca_layers": execution.get("max_dca_layers"),
        "dca_size_multiplier_max": execution.get("dca_size_multiplier_max"),
        "hedge_max_ratio": execution.get("hedge_max_ratio"),
        "requires_hard_stop_loss": execution.get("requires_hard_stop_loss", False),
        "trailing_take_profit": execution.get("trailing_take_profit", False),
        "cooldown_after_close_candles": execution.get("cooldown_after_close_candles"),
        "cooldown_after_stop_candles": execution.get("cooldown_after_stop_candles"),
        "min_risk_reward_ratio": execution.get("min_risk_reward_ratio"),
        "stop_loss": round(stop_loss, 8) if stop_loss is not None else None,
        "take_profit": round(take_profit, 8) if take_profit is not None else None,
        "risk_model": "atr_structure",
    }


def _default_intent(action: str, side: str) -> str:
    if action == "buy":
        return "open_long" if side != "short" else "close_short"
    if action == "sell":
        return "open_short" if side != "long" else "close_long"
    return "wait"


def _nofx_execution_profile(
    strategy_type: str,
    indicators: dict[str, Any],
    risk: dict[str, Any],
    definition: dict[str, Any],
    prompts: dict[str, Any],
) -> dict[str, Any]:
    source_text = " ".join([strategy_type, str(definition.get("name", "")), str(definition.get("description", "")), " ".join(str(value) for value in prompts.values())])
    is_dca = "dca" in source_text.lower()
    return {
        "strategy_type": strategy_type,
        "requires_contract": True,
        "allowed_actions": ["open_long", "open_short", "close_long", "close_short", "hold", "wait", "dca_long", "dca_short", "hedge"],
        "max_positions": risk.get("max_positions", 2),
        "max_dca_layers": 2 if is_dca else 1,
        "dca_size_multiplier_max": 1.2,
        "hedge_max_ratio": 0.5,
        "requires_hard_stop_loss": True,
        "trailing_take_profit": True,
        "cooldown_after_close_candles": 2,
        "cooldown_after_stop_candles": 4,
        "min_risk_reward_ratio": _normalize_ratio(risk.get("min_risk_reward_ratio"), 1.8),
        "stop_loss_atr_multiple": 1.5,
        "atr_periods": _as_number_list(_get_any(indicators, "atrPeriods", "atr_periods"), [14]),
        "primary_timeframe": _get_any(_get_any(indicators, "klines") or {}, "primary_timeframe", "primaryTimeframe") or "5m",
        "selected_timeframes": _get_any(_get_any(indicators, "klines") or {}, "selected_timeframes", "selectedTimeframes") or [],
    }


def _leverage_for_symbol(symbol: str, risk: dict[str, Any]) -> Optional[float]:
    base = symbol.split("/", 1)[0].upper()
    if base in {"BTC", "ETH"}:
        return _float_or_none(risk.get("btc_eth_max_leverage")) or 3
    return _float_or_none(risk.get("altcoin_max_leverage")) or 2


def _atr(candles: list, period: int) -> float:
    if len(candles) <= 1:
        return 0.0
    ranges = []
    for previous, current in zip(candles[-period - 1 : -1], candles[-period:]):
        ranges.append(max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close)))
    return sum(ranges) / max(len(ranges), 1)


def _first_number(value: Any, fallback: float) -> float:
    values = _as_number_list(value, [fallback])
    return values[0] if values else fallback


def _normalize_ratio(value: Any, fallback: float) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None and parsed > 0 else fallback


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slugify(value: str) -> str:
    raw = value.strip()
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", raw.lower()).strip("_")
    if not slug:
        slug = f"nofx_strategy_{digest}"
    elif not slug[0].isalpha():
        slug = f"nofx_{slug}"
    if len(slug) < 3:
        slug = f"nofx_{slug}_{digest}"
    if len(slug) > 49:
        slug = f"{slug[:40].rstrip('_')}_{digest}"
    return slug


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        enabled = _get_any(value, "enabled", "enable")
        return default if enabled is None else _boolish(enabled, default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _as_number_list(value: Any, fallback: list[int]) -> list[float]:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if isinstance(value, list):
        numbers = []
        for item in value:
            try:
                numbers.append(float(item))
            except (TypeError, ValueError):
                continue
        return numbers or fallback
    return fallback


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.55
    if confidence > 1:
        confidence /= 100
    return min(max(confidence, 0.0), 1.0)


def _redact_sensitive(value: Any) -> Any:
    sensitive_tokens = ("api_key", "apikey", "secret", "password", "token", "sendkey")
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            key_text = str(key).lower()
            clean[key] = "***REDACTED***" if any(token in key_text for token in sensitive_tokens) else _redact_sensitive(item)
        return clean
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def save_custom_strategy(definition: dict[str, Any]) -> RuleStrategy:
    strategy = RuleStrategy(definition)
    CUSTOM_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    path = CUSTOM_STRATEGY_DIR / f"{strategy.name}.json"
    path.write_text(json.dumps(strategy.definition, ensure_ascii=False, indent=2), encoding="utf-8")
    return strategy


def load_custom_strategies() -> dict[str, RuleStrategy]:
    strategies: dict[str, RuleStrategy] = {}
    if not CUSTOM_STRATEGY_DIR.exists():
        return strategies
    for path in sorted(CUSTOM_STRATEGY_DIR.glob("*.json")):
        try:
            definition = json.loads(path.read_text(encoding="utf-8"))
            strategy = RuleStrategy(definition)
            strategies[strategy.name] = strategy
        except Exception:
            continue
    return strategies


def _validate_rule(rule: Any) -> dict[str, Any]:
    if not isinstance(rule, dict):
        raise ValueError("Each rule must be an object.")
    indicator = str(rule.get("indicator", "")).strip()
    if indicator not in {"rsi", "sma_cross", "price_vs_sma", "breakout"}:
        raise ValueError(f"Unsupported indicator: {indicator}")
    clean = dict(rule)
    clean["indicator"] = indicator
    clean["action"] = _validate_action(rule.get("action"))
    clean["confidence"] = _validate_confidence(rule.get("confidence", 0.6))
    return clean


def _validate_action(value: Any) -> str:
    action = str(value).strip().lower()
    if action not in {item.value for item in TradeAction}:
        raise ValueError(f"Unsupported action: {value}")
    return action


def _validate_confidence(value: Any) -> float:
    confidence = float(value)
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1.")
    return confidence


def _rule_matches(rule: dict[str, Any], snapshot: MarketSnapshot) -> tuple[bool, str]:
    indicator = rule["indicator"]
    closes = [c.close for c in snapshot.candles]
    if indicator == "rsi":
        value = rsi(snapshot.candles, int(rule.get("period", 14)))
        matched = _compare(value, str(rule.get("operator", "<")), float(rule.get("value", 30)))
        return matched, f"RSI={value:.2f} {rule.get('operator')} {rule.get('value')}"
    if indicator == "sma_cross":
        fast = sma(closes, int(rule.get("fast", 12)))
        slow = sma(closes, int(rule.get("slow", 36)))
        direction = str(rule.get("direction", "above"))
        matched = fast > slow if direction == "above" else fast < slow
        return matched, f"SMA{rule.get('fast', 12)}={fast:.4f}, SMA{rule.get('slow', 36)}={slow:.4f}, direction={direction}"
    if indicator == "price_vs_sma":
        price = closes[-1]
        average = sma(closes, int(rule.get("period", 20)))
        matched = _compare(price, str(rule.get("operator", ">")), average * float(rule.get("multiplier", 1)))
        return matched, f"price={price:.4f}, SMA{rule.get('period', 20)}={average:.4f}"
    if indicator == "breakout":
        lookback = int(rule.get("lookback", 24))
        prior = snapshot.candles[-lookback - 1 : -1]
        if not prior:
            return False, "not enough candles for breakout"
        high = max(c.high for c in prior)
        low = min(c.low for c in prior)
        price = closes[-1]
        direction = str(rule.get("direction", "up"))
        matched = price > high if direction == "up" else price < low
        return matched, f"price={price:.4f}, channel=({low:.4f}, {high:.4f}), direction={direction}"
    return False, "unknown rule"


def _compare(left: float, operator: str, right: float) -> bool:
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "==":
        return left == right
    return False
