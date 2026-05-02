from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Any

from app.models import MarketSnapshot, StrategySignal, TradeAction
from app.strategies.base import Strategy
from app.strategies.indicators import rsi, sma

CUSTOM_STRATEGY_DIR = Path(os.getenv("TRADAI_STRATEGY_DIR", ".config/strategies"))


class RuleStrategy(Strategy):
    def __init__(self, definition: dict[str, Any]) -> None:
        self.definition = validate_strategy_definition(normalize_strategy_definition(definition))
        self.name = self.definition["name"]

    def evaluate(self, snapshot: MarketSnapshot) -> StrategySignal:
        for rule in self.definition["rules"]:
            matched, reason = _rule_matches(rule, snapshot)
            if matched:
                return StrategySignal(
                    strategy=self.name,
                    action=TradeAction(rule["action"]),
                    confidence=float(rule.get("confidence", 0.6)),
                    reason=reason,
                    metadata={
                        "custom": True,
                        "description": self.definition.get("description", ""),
                        "source_format": self.definition.get("source_format", "tradai"),
                        "nofx_config": self.definition.get("nofx_config", {}),
                        "nofx_prompt_sections": self.definition.get("nofx_prompt_sections", {}),
                    },
                )
        return StrategySignal(
            strategy=self.name,
            action=TradeAction(self.definition.get("default_action", "hold")),
            confidence=float(self.definition.get("default_confidence", 0.45)),
            reason="No custom rule matched.",
            metadata={
                "custom": True,
                "description": self.definition.get("description", ""),
                "source_format": self.definition.get("source_format", "tradai"),
                "nofx_config": self.definition.get("nofx_config", {}),
                "nofx_prompt_sections": self.definition.get("nofx_prompt_sections", {}),
            },
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
        "nofx_config": definition.get("nofx_config", {}),
        "nofx_prompt_sections": definition.get("nofx_prompt_sections", {}),
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
    indicators = _get_any(source, "indicators", "Indicators") or {}
    risk = _get_any(source, "riskControl", "risk_control", "RiskControl") or {}
    prompts = _get_any(source, "promptSections", "prompt_sections", "prompts") or {}
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
                {"indicator": "rsi", "period": period, "operator": "<", "value": 30, "action": "buy", "confidence": max(default_confidence, 0.68)},
                {"indicator": "rsi", "period": period, "operator": ">", "value": 70, "action": "sell", "confidence": max(default_confidence, 0.68)},
            ]
        )

    if _boolish(_get_any(indicators, "enableEMA", "enable_ema", "EMA", "ema"), default=True):
        ema_periods = sorted(_as_number_list(_get_any(indicators, "emaPeriods", "ema_periods", "EMAPeriods"), [12, 36]))
        if len(ema_periods) >= 2:
            fast, slow = int(ema_periods[0]), int(ema_periods[-1])
            rules.extend(
                [
                    {"indicator": "sma_cross", "fast": fast, "slow": slow, "direction": "above", "action": "buy", "confidence": default_confidence},
                    {"indicator": "sma_cross", "fast": fast, "slow": slow, "direction": "below", "action": "sell", "confidence": default_confidence},
                ]
            )

    if _boolish(_get_any(indicators, "enableBOLL", "enable_boll", "BOLL", "boll"), default=False):
        boll_periods = _as_number_list(_get_any(indicators, "bollPeriods", "boll_periods", "BOLLPeriods"), [20])
        period = int(boll_periods[0])
        rules.extend(
            [
                {"indicator": "price_vs_sma", "period": period, "operator": ">", "multiplier": 1.01, "action": "buy", "confidence": default_confidence},
                {"indicator": "price_vs_sma", "period": period, "operator": "<", "multiplier": 0.99, "action": "sell", "confidence": default_confidence},
            ]
        )

    if not rules:
        rules.append({"indicator": "sma_cross", "fast": 12, "slow": 36, "direction": "above", "action": "buy", "confidence": default_confidence})

    return {
        "name": name,
        "display_name": display_name,
        "description": str(definition.get("description", "NOFX StrategyConfig converted for Trading with AI."))[:300],
        "default_action": "hold",
        "default_confidence": min(default_confidence, 0.5),
        "rules": rules,
        "source_format": "nofx",
        "nofx_config": definition,
        "nofx_prompt_sections": prompts,
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
