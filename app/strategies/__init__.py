from app.strategies.builtin import STRATEGIES
from app.strategies.custom import load_custom_strategies

STRATEGIES.update(load_custom_strategies())

__all__ = ["STRATEGIES"]
