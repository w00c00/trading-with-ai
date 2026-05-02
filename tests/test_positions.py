from app.models import OrderResult, TradeAction
from app.positions import PositionBook


def test_position_book_tracks_cost_and_realized_profit() -> None:
    book = PositionBook()
    buy = OrderResult(
        exchange="paper",
        symbol="BTC/USDT",
        action=TradeAction.buy,
        quote_size=100,
        status="paper_filled",
        detail={"filled": 0.01, "average": 10_000, "cost": 100},
    )
    buy_position = book.apply_order(buy)
    assert buy_position.position_amount == 0.01
    assert buy_position.average_cost == 10_000
    assert buy_position.realized_profit is None

    sell = OrderResult(
        exchange="paper",
        symbol="BTC/USDT",
        action=TradeAction.sell,
        quote_size=120,
        status="paper_filled",
        detail={"filled": 0.01, "average": 12_000, "cost": 120},
    )
    sell_position = book.apply_order(sell)
    assert sell_position.position_amount == 0
    assert sell_position.realized_profit == 20
    assert sell_position.realized_profit_pct == 20
