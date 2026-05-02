from __future__ import annotations

import httpx

from app.config import Settings
from app.models import OrderResult, PositionUpdate


class ServerChanNotifier:
    def __init__(self, settings: Settings) -> None:
        self.sendkey = settings.serverchan_sendkey
        self.enabled = settings.notify_trade_success and bool(settings.serverchan_sendkey)

    async def send_trade_success(self, result: OrderResult, position: PositionUpdate) -> bool:
        if not self.enabled:
            return False
        title = f"交易成功 {result.action.value.upper()} {result.symbol}"
        desp = _format_trade_message(result, position)
        return await self.send(title, desp)

    async def send(self, title: str, desp: str) -> bool:
        if not self.sendkey:
            return False
        url = f"https://sctapi.ftqq.com/{self.sendkey}.send"
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, data={"title": title, "desp": desp})
            response.raise_for_status()
        return True


def _format_trade_message(result: OrderResult, position: PositionUpdate) -> str:
    profit = "暂无，买入或缺少成本基础"
    if position.realized_profit is not None:
        profit = f"{position.realized_profit:.4f} {position.quote_asset}"
        if position.realized_profit_pct is not None:
            profit += f" ({position.realized_profit_pct:.2f}%)"

    rows = [
        "## 成交信息",
        "",
        f"- 交易所：{result.exchange}",
        f"- 标的：{result.symbol}",
        f"- 方向：{result.action.value}",
        f"- 状态：{result.status}",
        f"- 订单ID：{result.order_id or '-'}",
        f"- 成交数量：{position.filled_amount:.8f} {position.base_asset}",
        f"- 成交均价：{position.average_price:.8f} {position.quote_asset}",
        f"- 成本/成交额：{position.trade_cost:.4f} {position.quote_asset}",
        f"- 已实现利润：{profit}",
        "",
        "## 当前持仓",
        "",
        f"- 持仓数量：{position.position_amount:.8f} {position.base_asset}",
        f"- 平均成本：{position.average_cost:.8f} {position.quote_asset}",
        f"- 持仓成本：{position.position_cost:.4f} {position.quote_asset}",
        "",
        "## 成交详情",
        "",
        "```json",
        result.model_dump_json(indent=2),
        "```",
    ]
    return "\n".join(rows)
