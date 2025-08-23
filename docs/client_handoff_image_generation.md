## What to share to let the client generate images

### Files/folders to include
- `requirements.txt`
- `.env.example` (create from your `.env`, keep only the key name; do not commit secrets)
- `image_gen/`
  - `orchestrate_generation.py`
  - `generate_with_gemini.py`
  - `gemini_image_gen_minimal.py`
  - `generation_plan.md` (targets per class/method)
- `scripts/`
  - `run_full_generation.ps1` (Windows one-shot runner)
- `datasets/` (with subfolders: `bird_drop/`, `clean/`, `dusty/`, `electrical_damage/`, `physical_damage/`, `snow_covered/`)
  - Tip: SC uses images from `datasets/clean/` as structural refs; DA uses images from other classes.

### Prerequisites
- Python 3.10+
- Google API key with access to an image-capable Gemini model or Imagen
- `.env` file at repo root containing one of:
```
GOOGLE_API_KEY=YOUR_KEY_HERE
# or
GEMINI_API_KEY=YOUR_KEY_HERE
```

### Setup
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### One-command run (Windows)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_full_generation.ps1
```
- Writes images to `generated/<class>/<method>/...`
- Manifests: `runs/manifest_generated_{sc,da,t2i}_<timestamp>.json`
- Log: `runs/full_generation_<timestamp>.log`

### Manual run (any OS)
Run the three phases sequentially (adjust counts if needed):
```bash
python image_gen/orchestrate_generation.py --classes bird_drop clean dusty electrical_damage physical_damage snow_covered --ratio 100,0,0 --output-root generated --manifest runs/manifest_generated_sc.json --class-count bird_drop=180 --class-count clean=185 --class-count dusty=190 --class-count electrical_damage=246 --class-count physical_damage=260 --class-count snow_covered=231
python image_gen/orchestrate_generation.py --classes bird_drop clean dusty electrical_damage physical_damage snow_covered --ratio 0,100,0 --output-root generated --manifest runs/manifest_generated_da.json --class-count bird_drop=40 --class-count clean=41 --class-count dusty=42 --class-count electrical_damage=55 --class-count physical_damage=58 --class-count snow_covered=51
python image_gen/orchestrate_generation.py --classes bird_drop clean dusty electrical_damage physical_damage snow_covered --ratio 0,0,100 --output-root generated --manifest runs/manifest_generated_t2i.json --class-count bird_drop=79 --class-count clean=83 --class-count dusty=86 --class-count electrical_damage=109 --class-count physical_damage=116 --class-count snow_covered=104
```

### Models used (configurable)
- Default Gemini: `gemini-2.0-flash-preview-image-generation`
- Imagen fallback: `imagen-4.0-generate-001`
- Override via flags: `--models-text`, `--models-image`.

### Troubleshooting
- Only gray/blank images: your key likely lacks access to image generation; try the Imagen fallback (`--models-image imagen-4.0-generate-001`) or request access.
- Slow/occasional failures: scripts retry with a 60s backoff until counts are reached.




