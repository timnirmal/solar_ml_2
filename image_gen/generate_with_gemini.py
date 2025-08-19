import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

try:
    # New SDK (preferred)
    from google import genai as google_genai
    HAS_GOOGLE_GENAI = True
except Exception:
    HAS_GOOGLE_GENAI = False

import google.generativeai as genai  # Fallback/also used for text prompts


def _ensure_api_key() -> str:
    load_dotenv()
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")  # last-resort fallback if user repurposes var
    )
    if not api_key:
        raise RuntimeError("No Gemini API key found in environment (.env). Use GOOGLE_API_KEY or GEMINI_API_KEY.")
    return api_key


def _configure_clients() -> Tuple[Optional[object], object]:
    api_key = _ensure_api_key()
    # Configure legacy/official generativeai client
    genai.configure(api_key=api_key)

    # Configure new SDK if available
    client = None
    if HAS_GOOGLE_GENAI:
        client = google_genai.Client(api_key=api_key)
    return client, genai


def pil_image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


@dataclass
class GeminiModels:
    # You can override these names when calling the functions
    text_to_image_model: str = "gemini-2.5-pro"  # For reasoning + control
    image_model: str = "gemini-2.5-pro"  # Using same for image understanding


def call_text_to_image(prompt: str, models: GeminiModels = GeminiModels()) -> bytes:
    client, legacy = _configure_clients()

    # The Gemini SDK does not directly return images for Pro; often image gen is via Imagen APIs.
    # Here we simulate by asking for a base64 image data URL if supported; otherwise return empty.
    # Replace with a proper image-gen endpoint when available in your environment.
    try:
        if client is not None:
            response = client.models.generate_content(
                model=models.text_to_image_model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            # Try to find an inline image in the response
            for cand in response.candidates or []:
                parts = getattr(cand, "content", {}).get("parts", []) if hasattr(cand, "content") else []
                for part in parts:
                    data = part.get("inline_data", {}) if isinstance(part, dict) else None
                    if data and data.get("mime_type", "").startswith("image/"):
                        return base64.b64decode(data.get("data", ""))
        # Fallback: ask legacy client for a data URL (depends on your account's enabled features)
        model = legacy.GenerativeModel(models.text_to_image_model)
        resp = model.generate_content(prompt)
        if hasattr(resp, "_result") and isinstance(getattr(resp, "_result", None), dict):
            # Try to parse imaginary inline image (not standard)
            pass
    except Exception:
        pass

    # If no image available, return an empty PNG as a placeholder to keep pipeline working
    img = Image.new("RGB", (512, 512), color=(240, 240, 240))
    return pil_image_to_bytes(img)


def call_image_edit(base_image: Image.Image, instruction: str, models: GeminiModels = GeminiModels()) -> bytes:
    # Structural consistency or domain adaptation can route here.
    # Currently uses caption + regenerate approach as placeholder.
    caption = f"Modify the solar panel photo to: {instruction}. Preserve structure and layout."
    return call_text_to_image(caption, models=models)


