"""In-memory runtime overrides for settings a user can change from the app UI.

The Gemini API key can be supplied via .env at startup, or entered later
through the /settings page without restarting the process. A runtime value
always takes priority over the .env value.
"""
from config import settings

_gemini_api_key: str | None = None


def set_gemini_api_key(api_key: str) -> None:
    global _gemini_api_key
    _gemini_api_key = api_key.strip()


def clear_gemini_api_key() -> None:
    global _gemini_api_key
    _gemini_api_key = None


def get_gemini_api_key() -> str | None:
    if _gemini_api_key:
        return _gemini_api_key
    return settings.GEMINI_API_KEY or None


def has_gemini_api_key() -> bool:
    return bool(get_gemini_api_key())
