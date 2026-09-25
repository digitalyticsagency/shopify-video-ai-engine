"""In-memory runtime overrides for settings a user can change from the app UI.

Both the Gemini API key and the Shopify store credentials can be supplied via
.env at startup, or entered later through the app's settings pages without
restarting the process. A runtime value always takes priority over .env.
"""
from config import settings

_gemini_api_key: str | None = None
_shopify_store_url: str | None = None
_shopify_access_token: str | None = None


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


def set_shopify_credentials(store_url: str, access_token: str) -> None:
    global _shopify_store_url, _shopify_access_token
    _shopify_store_url = store_url.strip().rstrip("/")
    _shopify_access_token = access_token.strip()


def clear_shopify_credentials() -> None:
    global _shopify_store_url, _shopify_access_token
    _shopify_store_url = None
    _shopify_access_token = None


def get_shopify_store_url() -> str | None:
    if _shopify_store_url:
        return _shopify_store_url
    return settings.SHOPIFY_STORE_URL or None


def get_shopify_access_token() -> str | None:
    if _shopify_access_token:
        return _shopify_access_token
    return settings.SHOPIFY_ACCESS_TOKEN or None


def has_shopify_credentials() -> bool:
    return bool(get_shopify_store_url() and get_shopify_access_token())
