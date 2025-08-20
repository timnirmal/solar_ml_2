import io
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from dotenv import load_dotenv


def _ensure_api_key() -> str:
    load_dotenv()
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("No Gemini API key found in environment (.env). Use GOOGLE_API_KEY or GEMINI_API_KEY.")
    return api_key


def _placeholder_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), color=(240, 240, 240)).save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class GeminiModels:
    # Default to image-capable Gemini with Imagen fallback
    text_to_image_model: str = "gemini-2.0-flash-preview-image-generation"
    imagen_model: str = "imagen-4.0-generate-001"


def _gemini_generate_image(prompt: str, model_name: str, debug: bool = False) -> Optional[bytes]:
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        if debug:
            print(f"[debug] google-genai not available: {e}")
        return None

    try:
        client = genai.Client(api_key=_ensure_api_key())
        if debug:
            print(f"[debug] gemini generate_content model={model_name}")
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    return inline.data
    except Exception as e:
        if debug:
            print(f"[debug] gemini error: {e}")
    return None


def _imagen_generate_image(prompt: str, model_name: str, debug: bool = False) -> Optional[bytes]:
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        if debug:
            print(f"[debug] google-genai not available: {e}")
        return None

    try:
        client = genai.Client(api_key=_ensure_api_key())
        if debug:
            print(f"[debug] imagen generate_images model={model_name}")
        resp = client.models.generate_images(
            model=model_name,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )
        if getattr(resp, "generated_images", None):
            img0 = resp.generated_images[0]
            if getattr(img0, "image", None) and getattr(img0.image, "image_bytes", None):
                return img0.image.image_bytes
    except Exception as e:
        if debug:
            print(f"[debug] imagen error: {e}")
    return None


def call_text_to_image(prompt: str, models: GeminiModels = GeminiModels(), debug: bool = False) -> bytes:
    img = _gemini_generate_image(prompt, models.text_to_image_model, debug=debug)
    if img:
        return img
    img = _imagen_generate_image(prompt, models.imagen_model, debug=debug)
    if img:
        return img
    return _placeholder_png()


def call_image_edit(base_image: Image.Image, instruction: str, models: GeminiModels = GeminiModels(), debug: bool = False) -> bytes:
    # Use descriptive edit prompt; SDK edit endpoints vary by access level
    prompt = f"{instruction}. Preserve solar panel layout and scene geometry; photorealistic."
    return call_text_to_image(prompt, models=models, debug=debug)


