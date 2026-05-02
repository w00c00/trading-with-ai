from __future__ import annotations

from app.models import OrderResult, PositionUpdate, TradeAction


class PositionBook:
    def __init__(self) -> None:
        self._positions: dict[str, dict[str, float]] = {}

    def apply_order(self, result: OrderResult) -> PositionUpdate:
        base, quote = result.symbol.split("/")
        filled = _extract_float(result.detail, "filled", "amount", "executedQty") or _safe_div(result.quote_size, _extract_price(result))
        average_price = _extract_price(result)
        trade_cost = _extract_float(result.detail, "cost", "cummulativeQuoteQty", "quoteQty") or filled * average_price
        position = self._positions.setdefault(result.symbol, {"amount": 0.0, "avg_cost": 0.0})

        realized_profit = None
        realized_profit_pct = None
        if result.action == TradeAction.buy:
            old_cost = position["amount"] * position["avg_cost"]
            new_amount = position["amount"] + filled
            if new_amount > 0:
                position["avg_cost"] = (old_cost + trade_cost) / new_amount
            position["amount"] = new_amount
        elif result.action == TradeAction.sell:
            sell_amount = min(filled, position["amount"]) if position["amount"] > 0 else filled
            cost_basis = sell_amount * position["avg_cost"]
            proceeds = sell_amount * average_price
            if cost_basis > 0:
                realized_profit = proceeds - cost_basis
                realized_profit_pct = realized_profit / cost_basis * 100
            position["amount"] = max(0.0, position["amount"] - filled)
            if position["amount"] == 0:
                position["avg_cost"] = 0.0

        return PositionUpdate(
            symbol=result.symbol,
            base_asset=base,
            quote_asset=quote,
            action=result.action,
            filled_amount=filled,
            average_price=average_price,
            trade_cost=trade_cost,
            position_amount=position["amount"],
            average_cost=position["avg_cost"],
            position_cost=position["amount"] * position["avg_cost"],
            realized_profit=realized_profit,
            realized_profit_pct=realized_profit_pct,
        )


def _extract_price(result: OrderResult) -> float:
    return _extract_float(result.detail, "average", "avgPrice", "price", "estimated_price") or _safe_div(result.quote_size, 1.0)


def _extract_float(data: dict, *keys: str) -> float:
    for key in keys:
        value = data.get(key)
        if value is None and isinstance(data.get("info"), dict):
            value = data["info"].get(key)
        try:
            if value not in (None, ""):
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
