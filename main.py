"""FastAPI entry point for the daily Shopify-to-video generation pipeline."""
import logging
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import runtime_settings
from ai_orchestrator import AIOrchestrationError, AIOrchestrator
from shopify_client import ShopifyClient, ShopifyClientError, ShopifyProduct
from video_engine import VideoRenderError, render_product_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shopify Video AI Engine",
    description="Automated daily worker that turns Shopify products into 10-second vertical promo videos.",
    version="1.0.0",
)

# In-memory run log; a real deployment would persist this to a database.
_run_history: list[dict] = []


class VideoResult(BaseModel):
    product_title: str
    status: str
    output_path: str | None = None
    error: str | None = None


class GenerationSummary(BaseModel):
    triggered_at: str
    products_processed: int
    videos_created: int
    results: list[VideoResult]


def _process_product(client_products: list[ShopifyProduct]) -> GenerationSummary:
    orchestrator = AIOrchestrator()
    results: list[VideoResult] = []

    for product in client_products:
        try:
            script = orchestrator.generate_script(product)
            output_path = render_product_video(product, script)
            results.append(VideoResult(product_title=product.title, status="success", output_path=output_path))
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


async def run_daily_video_generation() -> GenerationSummary:
    """Fetch top Shopify products and render a promo video for each one."""
    shopify_client = ShopifyClient()

    try:
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


@app.get("/")
async def root() -> dict:
    """Operational system summary."""
    return {
        "service": "Shopify Video AI Engine",
        "status": "running",
        "gemini_api_key_configured": runtime_settings.has_gemini_api_key(),
        "runs_completed": len(_run_history),
        "last_run": _run_history[-1] if _run_history else None,
    }


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


@app.get("/settings", response_class=HTMLResponse)
async def settings_page() -> str:
    """A minimal page to enter the Gemini API key without editing .env or restarting the server."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shopify Video AI Engine — Settings</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 480px; margin: 64px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { font-size: 20px; }
  label { display: block; font-size: 14px; font-weight: 600; margin-top: 24px; margin-bottom: 6px; }
  input[type="password"], input[type="text"] { width: 100%; padding: 10px 12px; font-size: 14px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }
  button { margin-top: 16px; padding: 10px 18px; font-size: 14px; border: none; border-radius: 6px; background: #111; color: #fff; cursor: pointer; }
  button:hover { background: #333; }
  button.secondary { background: #eee; color: #111; margin-left: 8px; }
  #status { margin-top: 16px; font-size: 13px; }
  .ok { color: #0a7d2c; }
  .err { color: #c0392b; }
</style>
</head>
<body>
  <h1>Gemini API Key</h1>
  <p style="font-size: 13px; color: #555;">
    Used to generate the 3-scene video script for each product. Stored in memory only
    (not written to disk); falls back to <code>GEMINI_API_KEY</code> in <code>.env</code> if unset.
  </p>
  <div id="current-status" style="font-size: 13px;"></div>

  <label for="api-key">Gemini API key</label>
  <input type="password" id="api-key" placeholder="AIzaSy..." autocomplete="off">
  <div>
    <button id="save-btn">Save</button>
    <button id="clear-btn" class="secondary">Clear</button>
  </div>
  <div id="status"></div>

<script>
  const statusEl = document.getElementById('status');
  const currentStatusEl = document.getElementById('current-status');
  const input = document.getElementById('api-key');

  async function refreshStatus() {
    const res = await fetch('/api/settings');
    const data = await res.json();
    currentStatusEl.textContent = data.gemini_api_key_configured
      ? '✓ A Gemini API key is currently configured.'
      : 'No Gemini API key configured yet.';
  }

  document.getElementById('save-btn').addEventListener('click', async () => {
    const apiKey = input.value.trim();
    if (!apiKey) {
      statusEl.textContent = 'Enter a key first.';
      statusEl.className = 'err';
      return;
    }
    const res = await fetch('/api/settings/gemini-key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey }),
    });
    if (res.ok) {
      statusEl.textContent = 'Saved.';
      statusEl.className = 'ok';
      input.value = '';
      refreshStatus();
    } else {
      statusEl.textContent = 'Failed to save key.';
      statusEl.className = 'err';
    }
  });

  document.getElementById('clear-btn').addEventListener('click', async () => {
    await fetch('/api/settings/gemini-key', { method: 'DELETE' });
    statusEl.textContent = 'Cleared.';
    statusEl.className = 'ok';
    refreshStatus();
  });

  refreshStatus();
</script>
</body>
</html>"""


@app.post("/api/generate-daily-videos", response_model=GenerationSummary)
async def generate_daily_videos(background_tasks: BackgroundTasks) -> GenerationSummary:
    """Trigger the daily video generation pipeline synchronously for immediate feedback.

    A background task also records the run so repeated automated triggers (e.g. a scheduler)
    do not block on the full render before returning a response.
    """
    try:
        summary = await run_daily_video_generation()
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
