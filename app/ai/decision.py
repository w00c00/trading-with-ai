from __future__ import annotations

from app.ai.providers import AIClient
from app.models import AIDecision, MarketSnapshot, StrategySignal, TradeAction


class AIDecisionMaker:
    def __init__(self, client: AIClient) -> None:
        self.client = client

    async def decide(self, snapshot: MarketSnapshot, signals: list[StrategySignal]) -> AIDecision:
        closes = [round(c.close, 4) for c in snapshot.candles[-20:]]
        signal_text = [signal.model_dump() for signal in signals]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conservative trading risk assistant. Return strict JSON only with keys: "
                    "action one of buy/sell/hold, confidence 0..1, reason string, risk_notes string array. "
                    "Prefer hold when signals conflict, data is thin, or confidence is low."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Market: {snapshot.exchange} {snapshot.symbol} {snapshot.timeframe}\n"
                    f"Last price: {snapshot.last_price}\n"
                    f"Recent closes: {closes}\n"
                    f"Strategy signals: {signal_text}\n"
                    f"Balances: {snapshot.balances}\n"
                    "Decide the next action."
                ),
            },
        ]
        data = await self.client.chat_json(messages)
        action = data.get("action", "hold")
        if action not in {item.value for item in TradeAction}:
            action = "hold"
        return AIDecision(
            action=TradeAction(action),
            confidence=float(data.get("confidence", 0.0)),
            reason=str(data.get("reason", "")),
            risk_notes=list(data.get("risk_notes", [])),
            raw=data,
        )
