"""Shopify OAuth: the install/callback flow for a Partners app (client id + secret).

Used when the merchant clicks "Connect with Shopify" instead of pasting a
Custom App access token directly. Needs SHOPIFY_CLIENT_ID, SHOPIFY_CLIENT_SECRET
and SHOPIFY_APP_URL (this app's own public base URL) set in .env — the redirect
URI built here (SHOPIFY_APP_URL + /auth/shopify/callback) must be whitelisted
in the Shopify Partners app's configuration.
"""
import hashlib
import hmac as hmac_lib
import logging
import re
import secrets
import time
from urllib.parse import urlencode

import httpx

from config import settings

logger = logging.getLogger(__name__)

_SHOP_DOMAIN_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*\.myshopify\.com$")
_STATE_TTL_SECONDS = 600

# state -> expiry timestamp; single-use, short-lived CSRF token for the install flow.
_pending_states: dict[str, float] = {}


class ShopifyOAuthError(Exception):
    """Raised when the OAuth install or callback flow fails or looks tampered with."""


def is_configured() -> bool:
    return bool(settings.SHOPIFY_CLIENT_ID and settings.SHOPIFY_CLIENT_SECRET and settings.SHOPIFY_APP_URL)


def validate_shop_domain(shop: str) -> str:
    """Validate a `shop` query param is a real *.myshopify.com domain. Returns it, or raises."""
    shop = (shop or "").strip().lower()
    if not _SHOP_DOMAIN_RE.match(shop):
        raise ShopifyOAuthError(f"Invalid shop domain: {shop!r}")
    return shop


def _redirect_uri() -> str:
    return settings.SHOPIFY_APP_URL.rstrip("/") + "/auth/shopify/callback"


def generate_state() -> str:
    """Create a single-use CSRF token for one install attempt, expiring after 10 minutes."""
    _prune_expired_states()
    state = secrets.token_urlsafe(24)
    _pending_states[state] = time.time() + _STATE_TTL_SECONDS
    return state


def consume_state(state: str) -> bool:
    """Check a state token was one we issued and hasn't expired, then invalidate it."""
    _prune_expired_states()
    expiry = _pending_states.pop(state, None)
    return expiry is not None


def _prune_expired_states() -> None:
    now = time.time()
    expired = [s for s, exp in _pending_states.items() if exp < now]
    for s in expired:
        _pending_states.pop(s, None)


def build_authorize_url(shop: str, state: str) -> str:
    if not is_configured():
        raise ShopifyOAuthError(
            "Shopify OAuth not configured. Set SHOPIFY_CLIENT_ID, SHOPIFY_CLIENT_SECRET, "
            "and SHOPIFY_APP_URL in .env."
        )
    params = {
        "client_id": settings.SHOPIFY_CLIENT_ID,
        "scope": settings.SHOPIFY_SCOPES,
        "redirect_uri": _redirect_uri(),
        "state": state,
    }
    return f"https://{shop}/admin/oauth/authorize?{urlencode(params)}"


def verify_hmac(params: dict[str, str]) -> bool:
    """Verify the callback's `hmac` param against SHOPIFY_CLIENT_SECRET, per Shopify's spec."""
    received = params.get("hmac", "")
    if not received:
        return False

    message_params = {k: v for k, v in params.items() if k not in ("hmac", "signature")}
    message = "&".join(f"{k}={v}" for k, v in sorted(message_params.items()))

    computed = hmac_lib.new(
        settings.SHOPIFY_CLIENT_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac_lib.compare_digest(computed, received)


async def exchange_code_for_token(shop: str, code: str) -> str:
    """Exchange the callback's one-time code for a permanent Admin API access token."""
    payload = {
        "client_id": settings.SHOPIFY_CLIENT_ID,
        "client_secret": settings.SHOPIFY_CLIENT_SECRET,
        "code": code,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"https://{shop}/admin/oauth/access_token", json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Shopify token exchange failed: HTTP %s: %s", exc.response.status_code, exc.response.text)
        raise ShopifyOAuthError(f"Token exchange failed: HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        logger.error("Shopify token exchange request failed: %s", exc)
        raise ShopifyOAuthError(f"Token exchange request failed: {exc}") from exc

    body = response.json()
    access_token = body.get("access_token")
    if not access_token:
        logger.error("Shopify token exchange response missing access_token: %s", body)
        raise ShopifyOAuthError("Shopify token exchange response missing access_token")

    return access_token
