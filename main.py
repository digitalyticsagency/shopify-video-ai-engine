"""FastAPI entry point for the daily Shopify-to-video generation pipeline."""
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import runtime_settings
import shopify_oauth
from ai_orchestrator import AIOrchestrationError, AIOrchestrator
from config import settings
from shopify_client import ShopifyClient, ShopifyClientError, ShopifyProduct
from shopify_oauth import ShopifyOAuthError
from video_engine import VideoRenderError, render_product_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shopify Video AI Engine",
    description="Automated daily worker that turns Shopify products into 10-second vertical promo videos.",
    version="1.0.0",
)

# Rendered MP4s are served back to the dashboard from here.
Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=settings.OUTPUT_DIR), name="videos")

# In-memory run log; a real deployment would persist this to a database.
_run_history: list[dict] = []


class VideoResult(BaseModel):
    product_title: str
    status: str
    output_path: str | None = None
    video_url: str | None = None
    error: str | None = None


class GenerationSummary(BaseModel):
    triggered_at: str
    products_processed: int
    videos_created: int
    results: list[VideoResult]


class ProductSummary(BaseModel):
    id: str
    title: str
    price: str
    currency_code: str
    image_url: str | None = None


def _process_product(client_products: list[ShopifyProduct]) -> GenerationSummary:
    orchestrator = AIOrchestrator()
    results: list[VideoResult] = []

    for product in client_products:
        try:
            script = orchestrator.generate_script(product)
            output_path = render_product_video(product, script)
            video_url = f"/videos/{Path(output_path).name}"
            results.append(
                VideoResult(
                    product_title=product.title,
                    status="success",
                    output_path=output_path,
                    video_url=video_url,
                )
            )
        except (AIOrchestrationError, VideoRenderError) as exc:
            logger.error("Failed to process product '%s': %s", product.title, exc)
            results.append(VideoResult(product_title=product.title, status="failed", error=str(exc)))
        except Exception as exc:  # noqa: BLE001 - guard the background task from crashing the worker
            logger.exception("Unexpected error processing product '%s'", product.title)
            results.append(VideoResult(product_title=product.title, status="failed", error=str(exc)))

    summary = GenerationSummary(
        triggered_at=datetime.now(timezone.utc).isoformat(),
        products_processed=len(client_products),
        videos_created=sum(1 for r in results if r.status == "success"),
        results=results,
    )
    _run_history.append(summary.model_dump())
    return summary


async def run_daily_video_generation(product_ids: list[str] | None = None) -> GenerationSummary:
    """Render a promo video for each given product, or the top 3 most recently updated if none given."""
    shopify_client = ShopifyClient()

    try:
        if product_ids:
            products = await shopify_client.fetch_products_by_ids(product_ids)
        else:
            products = await shopify_client.fetch_top_products(count=3)
    except ShopifyClientError as exc:
        logger.error("Shopify fetch failed: %s", exc)
        raise

    if not products:
        logger.info("No products returned from Shopify; skipping this run.")
        summary = GenerationSummary(
            triggered_at=datetime.now(timezone.utc).isoformat(),
            products_processed=0,
            videos_created=0,
            results=[],
        )
        _run_history.append(summary.model_dump())
        return summary

    return _process_product(products)


@app.get("/api/status")
async def status() -> dict:
    """Operational system summary."""
    return {
        "service": "Shopify Video AI Engine",
        "status": "running",
        "gemini_api_key_configured": runtime_settings.has_gemini_api_key(),
        "shopify_configured": runtime_settings.has_shopify_credentials(),
        "runs_completed": len(_run_history),
        "last_run": _run_history[-1] if _run_history else None,
    }


class ShopifyCredentialsUpdate(BaseModel):
    store_url: str = Field(..., min_length=1)
    access_token: str = Field(..., min_length=1)


class ShopifyStatus(BaseModel):
    shopify_configured: bool
    store_url: str | None = None
    oauth_available: bool = False


def _shopify_status() -> ShopifyStatus:
    return ShopifyStatus(
        shopify_configured=runtime_settings.has_shopify_credentials(),
        store_url=runtime_settings.get_shopify_store_url(),
        oauth_available=shopify_oauth.is_configured(),
    )


@app.get("/api/settings/shopify", response_model=ShopifyStatus)
async def get_shopify_status() -> ShopifyStatus:
    """Report whether Shopify credentials are configured (store URL only, never the token)."""
    return _shopify_status()


@app.post("/api/settings/shopify", response_model=ShopifyStatus)
async def set_shopify_credentials(payload: ShopifyCredentialsUpdate) -> ShopifyStatus:
    """Connect a Shopify store: store URL + Admin API access token, in memory, for this process."""
    runtime_settings.set_shopify_credentials(payload.store_url, payload.access_token)
    return _shopify_status()


@app.delete("/api/settings/shopify", response_model=ShopifyStatus)
async def clear_shopify_credentials() -> ShopifyStatus:
    """Disconnect the Shopify store (falls back to .env values, if set)."""
    runtime_settings.clear_shopify_credentials()
    return _shopify_status()


@app.get("/auth/shopify/install")
async def shopify_oauth_install(shop: str) -> RedirectResponse:
    """Start the OAuth flow: redirect the merchant to Shopify's own authorize screen."""
    try:
        validated_shop = shopify_oauth.validate_shop_domain(shop)
        state = shopify_oauth.generate_state()
        authorize_url = shopify_oauth.build_authorize_url(validated_shop, state)
    except ShopifyOAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(authorize_url)


@app.get("/auth/shopify/callback")
async def shopify_oauth_callback(request: Request) -> RedirectResponse:
    """Handle Shopify's redirect back: verify it, exchange the code, save the access token."""
    params = dict(request.query_params)
    shop = params.get("shop", "")
    state = params.get("state", "")
    code = params.get("code", "")

    try:
        validated_shop = shopify_oauth.validate_shop_domain(shop)
    except ShopifyOAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not shopify_oauth.consume_state(state):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")
    if not shopify_oauth.verify_hmac(params):
        raise HTTPException(status_code=400, detail="Shopify callback signature verification failed.")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    try:
        access_token = await shopify_oauth.exchange_code_for_token(validated_shop, code)
    except ShopifyOAuthError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    runtime_settings.set_shopify_credentials(validated_shop, access_token)
    logger.info("Shopify OAuth connected: %s", validated_shop)
    return RedirectResponse("/settings?connected=shopify")


@app.get("/api/products", response_model=list[ProductSummary])
async def list_products() -> list[ProductSummary]:
    """List recent products from the connected Shopify store, for picking which ones to render."""
    try:
        shopify_client = ShopifyClient()
        products = await shopify_client.fetch_products(count=20)
    except ShopifyClientError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return [
        ProductSummary(
            id=p.id,
            title=p.title,
            price=p.price,
            currency_code=p.currency_code,
            image_url=p.image_urls[0] if p.image_urls else None,
        )
        for p in products
    ]


class GeminiKeyUpdate(BaseModel):
    api_key: str = Field(..., min_length=1)


class GeminiKeyStatus(BaseModel):
    gemini_api_key_configured: bool


@app.get("/api/settings", response_model=GeminiKeyStatus)
async def get_settings_status() -> GeminiKeyStatus:
    """Report whether a Gemini API key is currently configured (never returns the value)."""
    return GeminiKeyStatus(gemini_api_key_configured=runtime_settings.has_gemini_api_key())


@app.post("/api/settings/gemini-key", response_model=GeminiKeyStatus)
async def set_gemini_key(payload: GeminiKeyUpdate) -> GeminiKeyStatus:
    """Set the Gemini API key used for script generation, in memory, for this running process."""
    runtime_settings.set_gemini_api_key(payload.api_key)
    return GeminiKeyStatus(gemini_api_key_configured=True)


@app.delete("/api/settings/gemini-key", response_model=GeminiKeyStatus)
async def clear_gemini_key() -> GeminiKeyStatus:
    """Clear the runtime Gemini API key (falls back to GEMINI_API_KEY from .env, if set)."""
    runtime_settings.clear_gemini_api_key()
    return GeminiKeyStatus(gemini_api_key_configured=runtime_settings.has_gemini_api_key())


_BASE_STYLE = """
  :root {
    --color-primary: #7C3AED;
    --color-on-primary: #FFFFFF;
    --color-secondary: #6366F1;
    --color-accent: #EC4899;
    --color-background: #FAF5FF;
    --color-foreground: #0F172A;
    --color-muted: #F7F3FD;
    --color-muted-foreground: #5B5470;
    --color-border: #EFE7FC;
    --color-destructive: #DC2626;
    --color-success: #16A34A;
    --color-ring: #7C3AED;
    --radius: 10px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --color-background: #120C22;
      --color-foreground: #F3F0FF;
      --color-muted: #1D1533;
      --color-muted-foreground: #B4ABCC;
      --color-border: #2C2247;
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--color-background);
    color: var(--color-foreground);
    margin: 0;
    min-height: 100vh;
  }
  code, .mono { font-family: 'Fira Code', ui-monospace, monospace; }
  a { color: var(--color-primary); }
  .nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid var(--color-border);
  }
  .nav-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
    font-size: 15px;
    text-decoration: none;
    color: var(--color-foreground);
  }
  .nav-link {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    font-weight: 500;
    text-decoration: none;
    color: var(--color-muted-foreground);
    padding: 8px 12px;
    border-radius: var(--radius);
    transition: background-color 200ms ease, color 200ms ease;
    cursor: pointer;
  }
  .nav-link:hover { background: var(--color-muted); color: var(--color-foreground); }
  .nav-link:focus-visible, button:focus-visible, input:focus-visible {
    outline: 2px solid var(--color-ring);
    outline-offset: 2px;
  }
  main { max-width: 760px; margin: 0 auto; padding: 32px 20px 80px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 0 0 12px; }
  .subtitle { color: var(--color-muted-foreground); font-size: 14px; margin: 0 0 28px; }
  .card {
    background: var(--color-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 20px;
  }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .stat-card {
    background: var(--color-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    padding: 16px;
  }
  .stat-label { font-size: 12px; color: var(--color-muted-foreground); margin-bottom: 6px; }
  .stat-value { font-size: 20px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
  }
  .badge-success { background: rgba(22, 163, 74, 0.12); color: var(--color-success); }
  .badge-error { background: rgba(220, 38, 38, 0.12); color: var(--color-destructive); }
  .badge-muted { background: rgba(91, 84, 112, 0.12); color: var(--color-muted-foreground); }
  button {
    font-family: inherit;
    font-size: 14px;
    font-weight: 600;
    padding: 11px 20px;
    border: none;
    border-radius: var(--radius);
    background: var(--color-primary);
    color: var(--color-on-primary);
    cursor: pointer;
    transition: background-color 200ms ease, opacity 200ms ease, transform 150ms ease;
  }
  button:hover { background: #6D28D9; }
  button:active { transform: scale(0.98); }
  button:disabled { opacity: 0.6; cursor: not-allowed; }
  button.secondary { background: transparent; color: var(--color-foreground); border: 1px solid var(--color-border); }
  button.secondary:hover { background: var(--color-muted); }
  input[type="password"], input[type="text"] {
    width: 100%;
    font-family: inherit;
    padding: 11px 12px;
    font-size: 14px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    background: var(--color-background);
    color: var(--color-foreground);
  }
  label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
  .run { border-top: 1px solid var(--color-border); padding: 14px 0; }
  .run:first-child { border-top: none; }
  .run-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
  .run-time { font-size: 12px; color: var(--color-muted-foreground); }
  .product-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 6px 0; font-size: 13px; }
  .product-title { flex: 1; }
  .empty-state { color: var(--color-muted-foreground); font-size: 14px; text-align: center; padding: 24px 0; }
  .icon { width: 16px; height: 16px; flex-shrink: 0; }
  #status-msg { margin-top: 14px; font-size: 13px; }
  .ok { color: var(--color-success); }
  .err { color: var(--color-destructive); }
  video.preview { width: 100%; max-width: 220px; border-radius: var(--radius); margin-top: 6px; display: block; }
  .divider { border: none; border-top: 1px solid var(--color-border); margin: 18px 0; }
  .product-picker { display: flex; flex-direction: column; gap: 2px; max-height: 340px; overflow-y: auto; margin: 4px 0 16px; }
  .product-row-select {
    display: flex; align-items: center; gap: 12px; padding: 8px 6px; border-radius: 8px; cursor: pointer;
  }
  .product-row-select:hover { background: var(--color-background); }
  .product-row-select input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--color-primary); cursor: pointer; }
  .product-thumb {
    width: 36px; height: 36px; border-radius: 6px; object-fit: cover; background: var(--color-border); flex-shrink: 0;
  }
  .product-row-select .product-name { flex: 1; font-size: 13px; }
  .product-row-select .product-price { font-family: 'Fira Code', monospace; font-size: 12px; color: var(--color-muted-foreground); }
  .link-btn {
    background: none; border: none; padding: 0; color: var(--color-primary); font-family: inherit; font-size: 13px;
    font-weight: 600; cursor: pointer; text-decoration: underline;
  }
  @media (max-width: 640px) {
    .stat-grid { grid-template-columns: 1fr; }
    main { padding: 24px 16px 60px; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
  }
"""

_ICON_CHECK = (
    '<svg class="icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M4 10l4 4 8-8" /></svg>'
)
_ICON_KEY = (
    '<svg class="icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">'
    '<circle cx="7" cy="13" r="3" stroke-linecap="round" stroke-linejoin="round" />'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M9.5 10.5L16 4m0 0h-3m3 0v3" /></svg>'
)
_ICON_CLOCK = (
    '<svg class="icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">'
    '<circle cx="10" cy="10" r="7.5" /><path stroke-linecap="round" d="M10 5.5V10l3 2" /></svg>'
)
_ICON_GEAR = (
    '<svg class="icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8">'
    '<circle cx="10" cy="10" r="2.6" />'
    '<path stroke-linecap="round" d="M10 3.5v1.8M10 14.7v1.8M16.5 10h-1.8M5.3 10H3.5'
    'M14.8 5.2l-1.3 1.3M6.5 13.5l-1.3 1.3M14.8 14.8l-1.3-1.3M6.5 6.5L5.2 5.2" /></svg>'
)
_ICON_STORE = (
    '<svg class="icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M4 6l6-3 6 3v8l-6 3-6-3V6z" />'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M4 6l6 3 6-3M10 9v8" /></svg>'
)
_ICON_HOME = (
    '<svg class="icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M3.5 9.5L10 4l6.5 5.5" />'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M5.5 8.5V16h9V8.5" /></svg>'
)

_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&'
    'family=Fira+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">'
)


_NAV = f"""
  <div class="nav">
    <a class="nav-title" href="/">{_ICON_HOME} Shopify Video AI Engine</a>
    <a class="nav-link" href="/settings">{_ICON_GEAR} Settings</a>
  </div>
"""


@app.get("/settings", response_class=HTMLResponse)
async def settings_page() -> str:
    """Connect the Shopify store and set the Gemini API key, without editing .env or restarting."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Settings — Shopify Video AI Engine</title>
{_FONT_LINK}
<style>{_BASE_STYLE}</style>
</head>
<body>
{_NAV}
<main>
  <h1>Settings</h1>
  <p class="subtitle">Connect your Shopify store and Gemini API key. Both are kept in server memory only, never written to disk; either falls back to its .env value if unset here.</p>

  <div class="card">
    <h2>{_ICON_STORE} Shopify store</h2>
    <div id="shopify-current-status" class="stat-value" style="margin-bottom: 18px; font-size: 14px;"></div>

    <div id="oauth-block" hidden>
      <label for="oauth-shop">Store domain</label>
      <input type="text" id="oauth-shop" placeholder="your-shop.myshopify.com" autocomplete="off">
      <div style="margin-top: 14px;">
        <button id="oauth-connect-btn">Connect with Shopify</button>
      </div>
      <p class="subtitle" style="margin: 10px 0 0; font-size: 12px;">
        Redirects to Shopify's own login — you approve the requested access there, no token copy-paste.
      </p>
      <hr class="divider">
      <button id="show-manual-btn" class="link-btn">Or paste an access token directly</button>
    </div>

    <div id="manual-block" hidden>
      <label for="shop-url">Store URL</label>
      <input type="text" id="shop-url" placeholder="your-shop.myshopify.com" autocomplete="off" style="margin-bottom: 14px;">

      <label for="shop-token">Admin API access token</label>
      <input type="password" id="shop-token" placeholder="shpat_..." autocomplete="off">
      <p class="subtitle" style="margin: 8px 0 0; font-size: 12px;">
        From your Shopify admin: Settings → Apps and sales channels → Develop apps → your app →
        API credentials → Admin API access token.
      </p>
      <div style="margin-top: 14px;">
        <button id="shopify-save-btn">Connect store</button>
        <button id="shopify-clear-btn" class="secondary">Disconnect</button>
      </div>
    </div>

    <div id="shopify-status-msg"></div>
  </div>

  <div class="card">
    <h2>Gemini API key</h2>
    <p class="subtitle">Used to generate the 3-scene video script for each product.</p>
    <div id="current-status" class="stat-value" style="margin-bottom: 18px; font-size: 14px;"></div>

    <label for="api-key">Gemini API key</label>
    <input type="password" id="api-key" placeholder="AIzaSy..." autocomplete="off">
    <div style="margin-top: 14px;">
      <button id="save-btn">Save key</button>
      <button id="clear-btn" class="secondary">Clear</button>
    </div>
    <div id="status-msg"></div>
  </div>
</main>

<script>
  const checkIcon = '{_ICON_CHECK}';

  // Shopify store
  const oauthBlock = document.getElementById('oauth-block');
  const manualBlock = document.getElementById('manual-block');
  const oauthShopInput = document.getElementById('oauth-shop');
  const shopUrlInput = document.getElementById('shop-url');
  const shopTokenInput = document.getElementById('shop-token');
  const shopifyStatusEl = document.getElementById('shopify-current-status');
  const shopifyStatusMsg = document.getElementById('shopify-status-msg');

  if (new URLSearchParams(location.search).get('connected') === 'shopify') {{
    shopifyStatusMsg.textContent = 'Connected via Shopify.';
    shopifyStatusMsg.className = 'ok';
  }}

  async function refreshShopifyStatus() {{
    const res = await fetch('/api/settings/shopify');
    const data = await res.json();
    shopifyStatusEl.innerHTML = data.shopify_configured
      ? checkIcon + ' <span style="color: var(--color-success);">Connected to ' + data.store_url + '</span>'
      : '<span style="color: var(--color-muted-foreground);">No Shopify store connected yet</span>';

    if (data.oauth_available) {{
      oauthBlock.hidden = false;
      manualBlock.hidden = true;
    }} else {{
      oauthBlock.hidden = true;
      manualBlock.hidden = false;
    }}
  }}

  document.getElementById('oauth-connect-btn').addEventListener('click', () => {{
    const shop = oauthShopInput.value.trim();
    if (!shop) {{
      shopifyStatusMsg.textContent = 'Enter your store domain first.';
      shopifyStatusMsg.className = 'err';
      return;
    }}
    location.href = '/auth/shopify/install?shop=' + encodeURIComponent(shop);
  }});

  document.getElementById('show-manual-btn').addEventListener('click', () => {{
    oauthBlock.hidden = true;
    manualBlock.hidden = false;
  }});

  document.getElementById('shopify-save-btn').addEventListener('click', async () => {{
    const storeUrl = shopUrlInput.value.trim();
    const accessToken = shopTokenInput.value.trim();
    if (!storeUrl || !accessToken) {{
      shopifyStatusMsg.textContent = 'Enter both the store URL and access token.';
      shopifyStatusMsg.className = 'err';
      return;
    }}
    const res = await fetch('/api/settings/shopify', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ store_url: storeUrl, access_token: accessToken }}),
    }});
    if (res.ok) {{
      shopifyStatusMsg.textContent = 'Connected.';
      shopifyStatusMsg.className = 'ok';
      shopUrlInput.value = '';
      shopTokenInput.value = '';
      refreshShopifyStatus();
    }} else {{
      shopifyStatusMsg.textContent = 'Failed to connect.';
      shopifyStatusMsg.className = 'err';
    }}
  }});

  document.getElementById('shopify-clear-btn').addEventListener('click', async () => {{
    await fetch('/api/settings/shopify', {{ method: 'DELETE' }});
    shopifyStatusMsg.textContent = 'Disconnected.';
    shopifyStatusMsg.className = 'ok';
    refreshShopifyStatus();
  }});

  refreshShopifyStatus();

  // Gemini key
  const statusMsg = document.getElementById('status-msg');
  const currentStatusEl = document.getElementById('current-status');
  const input = document.getElementById('api-key');

  async function refreshStatus() {{
    const res = await fetch('/api/settings');
    const data = await res.json();
    currentStatusEl.innerHTML = data.gemini_api_key_configured
      ? checkIcon + ' <span style="color: var(--color-success);">Gemini API key configured</span>'
      : '<span style="color: var(--color-muted-foreground);">No Gemini API key configured yet</span>';
  }}

  document.getElementById('save-btn').addEventListener('click', async () => {{
    const apiKey = input.value.trim();
    if (!apiKey) {{
      statusMsg.textContent = 'Enter a key first.';
      statusMsg.className = 'err';
      return;
    }}
    const res = await fetch('/api/settings/gemini-key', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ api_key: apiKey }}),
    }});
    if (res.ok) {{
      statusMsg.textContent = 'Saved.';
      statusMsg.className = 'ok';
      input.value = '';
      refreshStatus();
    }} else {{
      statusMsg.textContent = 'Failed to save key.';
      statusMsg.className = 'err';
    }}
  }});

  document.getElementById('clear-btn').addEventListener('click', async () => {{
    await fetch('/api/settings/gemini-key', {{ method: 'DELETE' }});
    statusMsg.textContent = 'Cleared.';
    statusMsg.className = 'ok';
    refreshStatus();
  }});

  refreshStatus();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """The main dashboard: trigger a generation run, watch status, browse recent runs."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shopify Video AI Engine</title>
{_FONT_LINK}
<style>{_BASE_STYLE}</style>
</head>
<body>
{_NAV}
<main>
  <h1>Shopify Video AI Engine</h1>
  <p class="subtitle">Fetches your top Shopify products and renders a 10-second vertical promo video for each one.</p>

  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-label">Shopify store</div>
      <div class="stat-value" id="shopify-status">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Gemini API key</div>
      <div class="stat-value" id="key-status">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Runs completed</div>
      <div class="stat-value" id="runs-count">—</div>
    </div>
  </div>

  <div class="card">
    <h2>Generate videos</h2>
    <p class="subtitle" style="margin-bottom: 16px;">
      Pulls your 3 most recently updated products, writes a script with Gemini, and renders each video.
    </p>
    <button id="generate-btn">Generate daily videos</button>
    <button id="load-products-btn" class="link-btn" style="margin-left: 16px;">Or choose specific products</button>

    <div id="product-picker-wrap" hidden>
      <hr class="divider">
      <div id="product-picker" class="product-picker"><div class="empty-state">Loading products…</div></div>
      <button id="generate-selected-btn">Generate selected</button>
    </div>

    <div id="status-msg"></div>
  </div>

  <div class="card">
    <h2>Recent runs</h2>
    <div id="runs-list"><div class="empty-state">Loading…</div></div>
  </div>
</main>

<script>
  const checkIcon = '{_ICON_CHECK}';
  const keyIcon = '{_ICON_KEY}';
  const storeIcon = '{_ICON_STORE}';
  const clockIcon = '{_ICON_CLOCK}';
  const shopifyStatusEl = document.getElementById('shopify-status');
  const keyStatusEl = document.getElementById('key-status');
  const runsCountEl = document.getElementById('runs-count');
  const runsListEl = document.getElementById('runs-list');
  const statusMsg = document.getElementById('status-msg');
  const generateBtn = document.getElementById('generate-btn');
  const loadProductsBtn = document.getElementById('load-products-btn');
  const productPickerWrap = document.getElementById('product-picker-wrap');
  const productPickerEl = document.getElementById('product-picker');
  const generateSelectedBtn = document.getElementById('generate-selected-btn');

  function escapeHtml(str) {{
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }}

  function renderRun(run) {{
    const time = new Date(run.triggered_at).toLocaleString();
    const rows = run.results.map(r => {{
      const badge = r.status === 'success'
        ? '<span class="badge badge-success">' + checkIcon + ' success</span>'
        : '<span class="badge badge-error">failed</span>';
      const preview = r.status === 'success' && r.video_url
        ? '<video class="preview" src="' + r.video_url + '" controls muted playsinline></video>'
        : (r.error ? '<div class="subtitle" style="margin:4px 0 0;font-size:12px;">' + escapeHtml(r.error) + '</div>' : '');
      return '<div class="product-row"><span class="product-title">' + escapeHtml(r.product_title) + '</span>' + badge + '</div>' + preview;
    }}).join('');

    return '<div class="run">' +
      '<div class="run-head">' +
        '<strong>' + run.videos_created + ' / ' + run.products_processed + ' videos created</strong>' +
        '<span class="run-time">' + clockIcon + ' ' + time + '</span>' +
      '</div>' + rows +
    '</div>';
  }}

  async function refreshStatus() {{
    const res = await fetch('/api/status');
    const data = await res.json();
    keyStatusEl.innerHTML = data.gemini_api_key_configured
      ? checkIcon + ' <span style="color: var(--color-success);">Configured</span>'
      : keyIcon + ' <a href="/settings" style="font-size:14px;">Set key</a>';
    shopifyStatusEl.innerHTML = data.shopify_configured
      ? checkIcon + ' <span style="color: var(--color-success);">Connected</span>'
      : storeIcon + ' <a href="/settings" style="font-size:14px;">Connect store</a>';
    runsCountEl.textContent = data.runs_completed;
  }}

  async function refreshRuns() {{
    const res = await fetch('/api/runs');
    const runs = await res.json();
    if (runs.length === 0) {{
      runsListEl.innerHTML = '<div class="empty-state">No runs yet. Generate your first batch above.</div>';
      return;
    }}
    runsListEl.innerHTML = runs.slice().reverse().map(renderRun).join('');
  }}

  async function runGeneration(productIds, triggerBtn, triggerLabel) {{
    triggerBtn.disabled = true;
    triggerBtn.textContent = 'Generating…';
    statusMsg.className = '';
    statusMsg.textContent = 'Fetching products, writing scripts, rendering video — this can take a minute.';
    try {{
      const res = await fetch('/api/generate-daily-videos', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ product_ids: productIds }}),
      }});
      const data = await res.json();
      if (res.ok) {{
        statusMsg.className = 'ok';
        statusMsg.textContent = data.videos_created + ' of ' + data.products_processed + ' videos created.';
        refreshStatus();
        refreshRuns();
      }} else {{
        statusMsg.className = 'err';
        statusMsg.textContent = data.detail || 'Generation failed.';
      }}
    }} catch (e) {{
      statusMsg.className = 'err';
      statusMsg.textContent = 'Request failed: ' + e;
    }} finally {{
      triggerBtn.disabled = false;
      triggerBtn.textContent = triggerLabel;
    }}
  }}

  generateBtn.addEventListener('click', () => runGeneration(null, generateBtn, 'Generate daily videos'));

  loadProductsBtn.addEventListener('click', async () => {{
    productPickerWrap.hidden = false;
    productPickerEl.innerHTML = '<div class="empty-state">Loading products…</div>';
    try {{
      const res = await fetch('/api/products');
      const data = await res.json();
      if (!res.ok) {{
        productPickerEl.innerHTML = '<div class="empty-state">' + escapeHtml(data.detail || 'Could not load products.') + '</div>';
        return;
      }}
      if (data.length === 0) {{
        productPickerEl.innerHTML = '<div class="empty-state">No products found in this store.</div>';
        return;
      }}
      productPickerEl.innerHTML = data.map(p => {{
        const thumb = p.image_url
          ? '<img class="product-thumb" src="' + p.image_url + '" alt="">'
          : '<div class="product-thumb"></div>';
        return '<label class="product-row-select">' +
          '<input type="checkbox" value="' + p.id + '">' +
          thumb +
          '<span class="product-name">' + escapeHtml(p.title) + '</span>' +
          '<span class="product-price">' + escapeHtml(p.price) + ' ' + escapeHtml(p.currency_code) + '</span>' +
        '</label>';
      }}).join('');
    }} catch (e) {{
      productPickerEl.innerHTML = '<div class="empty-state">Request failed: ' + escapeHtml(String(e)) + '</div>';
    }}
  }});

  generateSelectedBtn.addEventListener('click', () => {{
    const ids = Array.from(productPickerEl.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
    if (ids.length === 0) {{
      statusMsg.className = 'err';
      statusMsg.textContent = 'Select at least one product first.';
      return;
    }}
    runGeneration(ids, generateSelectedBtn, 'Generate selected');
  }});

  refreshStatus();
  refreshRuns();
</script>
</body>
</html>"""


class GenerateRequest(BaseModel):
    product_ids: list[str] | None = None


@app.post("/api/generate-daily-videos", response_model=GenerationSummary)
async def generate_daily_videos(
    background_tasks: BackgroundTasks, payload: GenerateRequest | None = None
) -> GenerationSummary:
    """Trigger video generation synchronously for immediate feedback.

    With no body (or an empty product_ids), renders the top 3 most recently updated products.
    With product_ids set, renders exactly those products instead.

    A background task also records the run so repeated automated triggers (e.g. a scheduler)
    do not block on the full render before returning a response.
    """
    product_ids = payload.product_ids if payload else None
    try:
        summary = await run_daily_video_generation(product_ids)
    except ShopifyClientError as exc:
        raise HTTPException(status_code=502, detail=f"Shopify API error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures as a 500
        logger.exception("Unhandled error during daily video generation")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {exc}") from exc

    background_tasks.add_task(logger.info, "Daily video generation run recorded: %s videos created", summary.videos_created)
    return summary


@app.get("/api/runs", response_model=list[GenerationSummary])
async def list_runs() -> list[GenerationSummary]:
    """Return the history of generation runs for this process's lifetime."""
    return _run_history
