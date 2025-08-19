import argparse
import io
import os
import time
from pathlib import Path
from typing import Optional

from PIL import Image
from dotenv import load_dotenv


def load_api_key() -> str:
    load_dotenv()
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Missing API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env")
    return api_key


def try_gemini_generate_image(prompt: str, model: str, api_key: str, debug: bool = False) -> Optional[bytes]:
    """Use new google-genai SDK to ask a Gemini image-capable model to return an image part."""
    try:
        from google import genai as google_genai  # type: ignore
    except Exception:
        if debug:
            print("[debug] google-genai package not available")
        return None

    try:
        client = google_genai.Client(api_key=api_key)
        if debug:
            print(f"[debug] trying models.generate_content model={model} (IMAGE modality)")
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"response_modalities": ["TEXT", "IMAGE"]},
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
            print(f"[debug] gemini generate_content error: {e}")
    return None


def try_imagen_generate_image(prompt: str, model: str, api_key: str, debug: bool = False) -> Optional[bytes]:
    """Use new google-genai SDK Imagen endpoint to generate an image."""
    try:
        from google import genai as google_genai  # type: ignore
    except Exception:
        if debug:
            print("[debug] google-genai package not available")
        return None

    try:
        client = google_genai.Client(api_key=api_key)
        if debug:
            print(f"[debug] trying models.generate_images model={model}")
        resp = client.models.generate_images(model=model, prompt=prompt)
        images = getattr(resp, "images", None)
        if images:
            img0 = images[0]
            if getattr(img0, "data", None):
                return img0.data
    except Exception as e:
        if debug:
            print(f"[debug] imagen generate_images error: {e}")
    return None


def save_bytes_as_image(img_bytes: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Validate and normalize by loading with PIL
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(out_path)
    except Exception:
        # If not decodable, write raw bytes with .bin extension
        out_path.with_suffix(".bin").write_bytes(img_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal Gemini image generation test script")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("generated_test/minimal.png"))
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--fallback-model", type=str, default="imagen-3.0-generate-001")
    parser.add_argument("--tries", type=int, default=3)
    parser.add_argument("--delay-seconds", type=int, default=60)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    api_key = load_api_key()
    last_err = None

    for attempt in range(1, args.tries + 1):
        if args.debug:
            print(f"[debug] attempt {attempt}/{args.tries} using model={args.model}")

        # Try Gemini image-capable model first
        img_bytes = try_gemini_generate_image(args.prompt, args.model, api_key, debug=args.debug)
        if img_bytes:
            save_bytes_as_image(img_bytes, args.out)
            print(f"Saved image to {args.out}")
            return 0

        # Try Imagen fallback
        img_bytes = try_imagen_generate_image(args.prompt, args.fallback_model, api_key, debug=args.debug)
        if img_bytes:
            save_bytes_as_image(img_bytes, args.out)
            print(f"Saved image to {args.out}")
            return 0

        last_err = f"no image returned on attempt {attempt}"
        if attempt < args.tries:
            if args.debug:
                print(f"[debug] {last_err}; sleeping {args.delay_seconds}s before retry")
            time.sleep(args.delay_seconds)

    print("Failed to generate an image. Ensure your account has access to an image-capable Gemini model or Imagen.")
    if args.debug and last_err:
        print(f"[debug] last_err: {last_err}")
    # Save a placeholder to inspect pipeline
    Image.new("RGB", (512, 512), color=(240, 240, 240)).save(args.out)
    print(f"Wrote placeholder image to {args.out}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


