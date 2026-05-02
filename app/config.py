from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.secure_config import load_encrypted_config


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Trading with AI"
    host: str = "0.0.0.0"
    port: int = 3471

    live_trading_enabled: bool = False
    default_exchange: str = "paper"
    default_symbol: str = "BTC/USDT"
    default_timeframe: str = "1h"
    trade_quote_size: float = Field(default=50.0, gt=0)
    max_position_quote: float = Field(default=200.0, gt=0)
    min_ai_confidence: float = Field(default=0.55, ge=0, le=1)

    exchange_id: str = "binance"
    exchange_api_key: str = ""
    exchange_secret: str = ""
    exchange_password: str = ""
    exchange_sandbox: bool = True

    ai_provider: Literal["openrouter", "openai", "minimax", "mimo", "custom_openai_compatible"] = "openrouter"
    ai_model: str = "openai/gpt-4o-mini"
    ai_api_key: str = ""
    ai_base_url: str = ""

    serverchan_sendkey: str = ""
    notify_trade_success: bool = True
    notify_dry_run: bool = False


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    overrides = load_encrypted_config()
    if overrides:
        settings = settings.model_copy(update=overrides)
    return settings
