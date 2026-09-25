"""Gemini-powered orchestration that turns product data into a structured video script."""
import logging
from enum import Enum

from google import genai
from pydantic import BaseModel, Field

import runtime_settings
from shopify_client import ShopifyProduct

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.5-flash-lite"


class AnimationStyle(str, Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    STATIC = "static"


class Scene(BaseModel):
    text_overlay: str = Field(..., description="Ultra-punchy marketing copy, max 4-5 words.")
    image_index: int = Field(..., ge=0, le=2, description="Which of the 3 retrieved product images to use.")
    animation_style: AnimationStyle = Field(..., description="Ken Burns animation applied to this scene.")


class VideoScriptSchema(BaseModel):
    hook: Scene = Field(..., description="Scene 1: Hook. Duration 3.0s.")
    core_value: Scene = Field(..., description="Scene 2: Core Value/Feature. Duration 4.0s.")
    call_to_action: Scene = Field(..., description="Scene 3: Call to Action / Price. Duration 3.0s.")


class AIOrchestrationError(Exception):
    """Raised when Gemini fails to produce a usable video script."""


class AIOrchestrator:
    """Wraps the google-genai client to generate structured 3-scene video scripts."""

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or runtime_settings.get_gemini_api_key()
        if not resolved_key:
            raise AIOrchestrationError(
                "No Gemini API key configured. Set one at /settings or via GEMINI_API_KEY in .env."
            )
        self.client = genai.Client(api_key=resolved_key)

    def generate_script(self, product: ShopifyProduct) -> VideoScriptSchema:
        """Generate a strict 3-scene VideoScriptSchema for a single Shopify product."""
        num_images = len(product.image_urls)
        if num_images == 0:
            logger.warning("Product '%s' has no images; scenes will fall back to image_index 0", product.title)

        prompt = self._build_prompt(product, num_images)

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": VideoScriptSchema,
                },
            )
        except Exception as exc:
            logger.error("Gemini generation failed for product '%s': %s", product.title, exc)
            raise AIOrchestrationError(f"Gemini generation failed: {exc}") from exc

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, VideoScriptSchema):
            script = parsed
        else:
            try:
                script = VideoScriptSchema.model_validate_json(response.text)
            except Exception as exc:
                logger.error("Failed to parse Gemini response for product '%s': %s", product.title, exc)
                raise AIOrchestrationError(f"Failed to parse Gemini response: {exc}") from exc

        return self._clamp_image_indices(script, num_images)

    @staticmethod
    def _build_prompt(product: ShopifyProduct, num_images: int) -> str:
        max_index = max(num_images - 1, 0)
        return (
            "You are a senior short-form video copywriter creating a 10-second vertical "
            "product promo video for a Shopify store.\n\n"
            f"Product title: {product.title}\n"
            f"Product description: {product.description or 'No description available.'}\n"
            f"Price: {product.price} {product.currency_code}\n"
            f"Number of available product images: {num_images} (valid image_index values: 0 to {max_index}).\n\n"
            "Produce exactly 3 scenes:\n"
            "1. hook (3.0s): an attention-grabbing opener.\n"
            "2. core_value (4.0s): the standout feature or benefit.\n"
            "3. call_to_action (3.0s): a punchy CTA that includes the price.\n\n"
            "Each scene needs ultra-punchy text_overlay copy (max 4-5 words), a valid image_index, "
            "and an animation_style of 'zoom_in', 'zoom_out', or 'static'."
        )

    @staticmethod
    def _clamp_image_indices(script: VideoScriptSchema, num_images: int) -> VideoScriptSchema:
        max_index = max(num_images - 1, 0)
        for scene in (script.hook, script.core_value, script.call_to_action):
            if scene.image_index > max_index:
                scene.image_index = max_index
        return script
