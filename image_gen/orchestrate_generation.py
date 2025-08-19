import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from dotenv import load_dotenv

# Support running both as module (-m image_gen.orchestrate_generation) and as a script
try:
    from .generate_with_gemini import GeminiModels, call_image_edit, call_text_to_image
except Exception:
    # Fallback for direct execution
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from image_gen.generate_with_gemini import GeminiModels, call_image_edit, call_text_to_image


CLASS_NAMES = [
    "bird_drop",
    "clean",
    "dusty",
    "electrical_damage",
    "physical_damage",
    "snow_covered",
]


@dataclass
class MethodBudgets:
    structural_consistency: int
    domain_adaptation: int
    text_to_image: int


@dataclass
class MethodRatios:
    structural_consistency: float
    domain_adaptation: float
    text_to_image: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    load_dotenv()
    # Validate that an API key exists early
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Missing Gemini API key in .env (GOOGLE_API_KEY or GEMINI_API_KEY)")


def pick_reference_images(dataset_root: Path, class_name: str, max_images: int) -> List[Path]:
    class_dir = dataset_root / class_name
    imgs: List[Path] = []
    if class_dir.exists():
        for p in class_dir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                imgs.append(p)
                if len(imgs) >= max_images:
                    break
    return imgs


def generate_structural_consistency(class_name: str, out_dir: Path, num_images: int, models: GeminiModels) -> List[Path]:
    ensure_dir(out_dir)
    # Use base clean panels as structure reference if available
    reference_paths = list((out_dir.parent.parent / "clean").rglob("*.jpg"))[: min(10, num_images)]
    generated: List[Path] = []
    for i in range(num_images):
        instruction = f"apply '{class_name}' characteristics while preserving panel layout and scene geometry"
        # Placeholder: we are not passing image+mask because of SDK constraints in this scaffolding
        img_bytes = call_image_edit(Image.new("RGB", (1024, 1024), color=(255, 255, 255)), instruction, models=models)
        out_path = out_dir / f"struct_{i:05d}.png"
        out_path.write_bytes(img_bytes)
        generated.append(out_path)
    return generated


def generate_domain_adaptation(class_name: str, dataset_root: Path, out_dir: Path, num_images: int, models: GeminiModels) -> List[Path]:
    ensure_dir(out_dir)
    # Pick a few example images from the target class to adapt style
    ref_images = pick_reference_images(dataset_root, class_name, max_images=5)
    prompt_prefix = f"Domain adapt to '{class_name}' conditions using reference style cues. High realism, solar panels visible."
    generated: List[Path] = []
    for i in range(num_images):
        prompt = prompt_prefix
        img_bytes = call_text_to_image(prompt, models=models)
        out_path = out_dir / f"dom_{i:05d}.png"
        out_path.write_bytes(img_bytes)
        generated.append(out_path)
    return generated


def generate_text_to_image(class_name: str, out_dir: Path, num_images: int, models: GeminiModels) -> List[Path]:
    ensure_dir(out_dir)
    base_prompt = (
        "High-resolution aerial or close-up photo of photovoltaic solar panels, "
        "environment and lighting consistent with utility-scale plants, "
        "camera realism, physically plausible imperfections."
    )
    class_prompt_map: Dict[str, str] = {
        "bird_drop": "Panels with bird droppings causing localized soiling patterns",
        "clean": "Clean panels with no visible defects",
        "dusty": "Panels covered with fine dust haze, reducing gloss",
        "electrical_damage": "Panels showing electrical faults like hotspots or burn marks",
        "physical_damage": "Cracked or shattered glass on panels, broken cells",
        "snow_covered": "Panels partially covered by snow, cold lighting",
    }
    condition = class_prompt_map.get(class_name, class_name)
    generated: List[Path] = []
    for i in range(num_images):
        prompt = f"{base_prompt} Condition: {condition}."
        img_bytes = call_text_to_image(prompt, models=models)
        out_path = out_dir / f"t2i_{i:05d}.png"
        out_path.write_bytes(img_bytes)
        generated.append(out_path)
    return generated


def normalize_ratios(r: MethodRatios) -> MethodRatios:
    s = r.structural_consistency + r.domain_adaptation + r.text_to_image
    if s <= 0:
        return MethodRatios(1.0, 0.0, 0.0)
    return MethodRatios(r.structural_consistency / s, r.domain_adaptation / s, r.text_to_image / s)


def plan_counts(target_per_class: int, ratios: MethodRatios) -> MethodBudgets:
    ratios = normalize_ratios(ratios)
    # Priority via rounding: allocate SC first, then DA, remainder to T2I
    sc = int(target_per_class * ratios.structural_consistency)
    da = int(target_per_class * ratios.domain_adaptation)
    t2i = max(0, target_per_class - sc - da)
    return MethodBudgets(sc, da, t2i)


def orchestrate(dataset_root: Path, output_root: Path, targets: Dict[str, int], models: GeminiModels, ratios: MethodRatios) -> Dict[str, Dict[str, List[str]]]:
    load_env()
    ensure_dir(output_root)
    manifest: Dict[str, Dict[str, List[str]]] = {}
    for class_name, target in targets.items():
        budgets = plan_counts(target, ratios)
        class_out = output_root / class_name
        ensure_dir(class_out)

        # method-specific subfolders
        sc_dir = class_out / "structural_consistency"
        da_dir = class_out / "domain_adaptation"
        t2i_dir = class_out / "text_to_image"
        ensure_dir(sc_dir)
        ensure_dir(da_dir)
        ensure_dir(t2i_dir)

        manifest[class_name] = {"structural_consistency": [], "domain_adaptation": [], "text_to_image": []}

        sc_paths = generate_structural_consistency(class_name, sc_dir, budgets.structural_consistency, models)
        manifest[class_name]["structural_consistency"] = [str(p) for p in sc_paths]

        da_paths = generate_domain_adaptation(class_name, dataset_root, da_dir, budgets.domain_adaptation, models)
        manifest[class_name]["domain_adaptation"] = [str(p) for p in da_paths]

        t2i_paths = generate_text_to_image(class_name, t2i_dir, budgets.text_to_image, models)
        manifest[class_name]["text_to_image"] = [str(p) for p in t2i_paths]

    return manifest


def _parse_class_counts(values: List[str]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for item in values:
        if "=" not in item:
            continue
        name, num = item.split("=", 1)
        name = name.strip()
        try:
            overrides[name] = int(num)
        except ValueError:
            pass
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate images for dataset classes using Gemini with 3 methods.")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent.parent / "datasets")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent.parent / "generated")
    parser.add_argument("--target-per-class", type=int, default=100)
    parser.add_argument("--classes", type=str, nargs="*", default=CLASS_NAMES)
    parser.add_argument("--models-text", type=str, default="gemini-2.5-pro")
    parser.add_argument("--models-image", type=str, default="gemini-2.5-pro")
    parser.add_argument("--ratio", type=str, default="50,30,20", help="Ratios for SC,DA,T2I in percentages (priority order)")
    parser.add_argument("--class-count", type=str, action="append", default=[], help="Override per-class count like name=123 (can repeat)")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent.parent / "runs" / "manifest_generated.json")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, do not generate images")
    args = parser.parse_args()

    models = GeminiModels(text_to_image_model=args.models_text, image_model=args.models_image)

    # Parse ratios
    try:
        r_sc, r_da, r_t2i = [float(x.strip()) for x in args.ratio.split(",")]
    except Exception:
        r_sc, r_da, r_t2i = (50.0, 30.0, 20.0)
    ratios = MethodRatios(r_sc, r_da, r_t2i)

    # Targets with overrides
    targets: Dict[str, int] = {name: args.target_per_class for name in args.classes}
    overrides = _parse_class_counts(args.class_count)
    for k, v in overrides.items():
        if k in targets:
            targets[k] = v

    if args.dry_run:
        plan = {name: plan_counts(targets[name], ratios).__dict__ for name in targets}
        print(json.dumps({"targets": targets, "ratios": ratios.__dict__, "budgets": plan}, indent=2))
        return 0

    manifest = orchestrate(args.dataset_root, args.output_root, targets, models, ratios)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


