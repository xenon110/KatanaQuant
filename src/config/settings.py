"""
System settings and configuration management using Pydantic.
"""
from typing import List, Optional, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.core.enums import TradingMode, AccountType


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General Operation
    trading_mode: TradingMode = Field(default=TradingMode.DRY_RUN, alias="TRADING_MODE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    database_url: str = Field(default="sqlite+aiosqlite:///./trading_system.db", alias="DATABASE_URL")

    # Alpaca Broker
    alpaca_api_key: str = Field(default="placeholder_key", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="placeholder_secret", alias="ALPACA_SECRET_KEY")
    alpaca_paper: bool = Field(default=True, alias="ALPACA_PAPER")
    alpaca_base_url: Optional[str] = Field(default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL")

    # Finnhub Real-Time Market Data & News
    finnhub_api_key: Optional[str] = Field(default=None, alias="FINNHUB_API_KEY")

    # Symbols & Universe
    watchlist_symbols: Any = Field(
        default=["AAPL", "MSFT", "NVDA", "SPY", "QQQ"],
        alias="WATCHLIST_SYMBOLS"
    )

    @field_validator("watchlist_symbols", mode="before")
    @classmethod
    def parse_watchlist(cls, v):
        if isinstance(v, str):
            # If it's a JSON array string
            if v.startswith("[") and v.endswith("]"):
                try:
                    import json
                    return json.loads(v)
                except Exception:
                    pass
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return v

    # Account Mechanics
    account_type: AccountType = Field(default=AccountType.MARGIN, alias="ACCOUNT_TYPE")
    initial_account_equity: float = Field(default=30000.00, alias="INITIAL_ACCOUNT_EQUITY")
    enforce_pdt: bool = Field(default=True, alias="ENFORCE_PDT")

    # Deterministic Risk Limits
    max_position_size_usd: float = Field(default=5000.00, alias="MAX_POSITION_SIZE_USD")
    max_position_size_pct_nav: float = Field(default=0.10, alias="MAX_POSITION_SIZE_PCT_NAV")
    max_concurrent_positions: int = Field(default=4, alias="MAX_CONCURRENT_POSITIONS")
    max_daily_loss_usd: float = Field(default=600.00, alias="MAX_DAILY_LOSS_USD")
    max_daily_loss_pct: float = Field(default=0.02, alias="MAX_DAILY_LOSS_PCT")
    max_staleness_seconds: int = Field(default=60, alias="MAX_STALENESS_SECONDS")

    # LLM Settings (Supports Free & Paid Providers)
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gemini-flash-latest", alias="LLM_MODEL")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, alias="HUGGINGFACE_API_KEY")
    # Mobile & Remote Alerts (Optional Telegram Bot)
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")

    # Supabase Cloud Database Configuration
    supabase_url: Optional[str] = Field(default=None, alias="SUPABASE_URL")
    supabase_anon_key: Optional[str] = Field(default=None, alias="SUPABASE_ANON_KEY")
    supabase_service_role_key: Optional[str] = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")


# Global singleton settings instance
settings = Settings()

