"""Application configuration loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    SHOPIFY_STORE_URL: str
    SHOPIFY_ACCESS_TOKEN: str
    GEMINI_API_KEY: str = ""
    OUTPUT_DIR: str = "./generated_videos"


settings = Settings()
