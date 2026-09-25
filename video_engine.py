"""MoviePy rendering engine: turns a VideoScriptSchema + product images into a vertical MP4."""
import io
import logging
import uuid
from pathlib import Path

import httpx
import numpy as np
from moviepy import (
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
from PIL import Image

from ai_orchestrator import Scene, VideoScriptSchema
from config import settings
from shopify_client import ShopifyProduct

logger = logging.getLogger(__name__)

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30
ZOOM_FACTOR = 1.15  # Total scale change applied over a scene's duration for the Ken Burns effect.

_SCENE_DURATIONS = {
    "hook": 3.0,
    "core_value": 4.0,
    "call_to_action": 3.0,
}

# Bold sans-serif TTF/TTC candidates checked in order; MoviePy 2.x needs a real font FILE path.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # RHEL/Fedora
]


class VideoRenderError(Exception):
    """Raised when a video fails to render, from a bad image download to an encoding error."""


def _resolve_font() -> str:
    """Return the first available bold sans-serif font file path for text overlays."""
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise VideoRenderError(
        "No usable font file found. Install a bold sans-serif TTF "
        f"(checked: {', '.join(_FONT_CANDIDATES)})."
    )


def _slugify(title: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "-" for c in title).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "product"


def _download_image(url: str) -> Image.Image:
    """Download a remote image and return it as an RGB PIL Image."""
    try:
        response = httpx.get(url, timeout=20.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Failed to download image '%s': %s", url, exc)
        raise VideoRenderError(f"Failed to download image: {url}") from exc

    try:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        logger.error("Failed to decode image '%s': %s", url, exc)
        raise VideoRenderError(f"Failed to decode image: {url}") from exc

    return image


def _cover_resize(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize + center-crop an image to fully cover the target canvas (cover, not contain)."""
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _build_scene_clip(image: Image.Image, scene: Scene, duration: float) -> CompositeVideoClip:
    """Build one Ken Burns + text-overlay scene clip at the full canvas size."""
    base = _cover_resize(image, VIDEO_WIDTH, VIDEO_HEIGHT)
    base_array = np.array(base)
    image_clip = ImageClip(base_array).with_duration(duration)

    if scene.animation_style == "zoom_in":
        zoom_fn = lambda t: 1.0 + (ZOOM_FACTOR - 1.0) * (t / duration)
    elif scene.animation_style == "zoom_out":
        zoom_fn = lambda t: ZOOM_FACTOR - (ZOOM_FACTOR - 1.0) * (t / duration)
    else:
        zoom_fn = lambda t: 1.0

    animated_image = (
        image_clip.resized(zoom_fn)
        .with_position(("center", "center"))
    )

    text_clip = TextClip(
        font=_resolve_font(),
        text=scene.text_overlay,
        font_size=90,
        color="white",
        method="caption",
        size=(int(VIDEO_WIDTH * 0.85), None),
        text_align="center",
        stroke_color="black",
        stroke_width=3,
    )
    text_w, text_h = text_clip.size
    text_clip = text_clip.with_duration(duration).with_position(("center", "center"))

    # A translucent backdrop bar keeps the caption readable regardless of the underlying photo.
    backdrop = (
        ColorClip(size=(text_w + 80, text_h + 60), color=(0, 0, 0))
        .with_opacity(0.4)
        .with_duration(duration)
        .with_position(("center", "center"))
    )

    return CompositeVideoClip(
        [animated_image, backdrop, text_clip], size=(VIDEO_WIDTH, VIDEO_HEIGHT)
    ).with_duration(duration)


def render_product_video(product: ShopifyProduct, script: VideoScriptSchema) -> str:
    """Render the 3-scene, 10-second vertical MP4 for a product and return the output file path."""
    scenes = {
        "hook": script.hook,
        "core_value": script.core_value,
        "call_to_action": script.call_to_action,
    }

    if not product.image_urls:
        raise VideoRenderError(f"Product '{product.title}' has no images to render a video from")

    clips = []
    for scene_name, scene in scenes.items():
        index = min(scene.image_index, len(product.image_urls) - 1)
        image = _download_image(product.image_urls[index])
        duration = _SCENE_DURATIONS[scene_name]
        clips.append(_build_scene_clip(image, scene, duration))

    final_clip = concatenate_videoclips(clips, method="compose")

    output_dir = Path(settings.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(product.title)}-{uuid.uuid4().hex[:8]}.mp4"
    output_path = output_dir / filename

    try:
        final_clip.write_videofile(
            str(output_path),
            fps=FPS,
            codec="libx264",
            audio=False,
            bitrate="4000k",
            preset="medium",
            threads=4,
            logger=None,
        )
    except Exception as exc:
        logger.error("Failed to render video for product '%s': %s", product.title, exc)
        raise VideoRenderError(f"Failed to render video: {exc}") from exc
    finally:
        final_clip.close()
        for clip in clips:
            clip.close()

    logger.info("Rendered video for '%s' -> %s", product.title, output_path)
    return str(output_path)
