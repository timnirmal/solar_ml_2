import argparse
import datetime as dt
import os
from pathlib import Path

import fitz  # PyMuPDF


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    with fitz.open(pdf_path) as doc:
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
    return "\n".join(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDFs into timestamped run folder")
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Paths to PDF files. If omitted, defaults to all PDFs under ./docs",
    )
    parser.add_argument(
        "--runs_dir",
        default="runs",
        help="Base runs directory",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    docs_dir = project_root / "docs"
    runs_dir = project_root / args.runs_dir

    # Discover input PDFs
    input_files: list[Path]
    if args.input:
        input_files = [Path(p).resolve() for p in args.input]
    else:
        input_files = sorted(docs_dir.glob("*.pdf"))

    if not input_files:
        raise SystemExit("No PDF files found. Provide --input or add PDFs under ./docs")

    # Create timestamped run folder
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / f"run-{timestamp}"
    ensure_directory(run_dir)

    # Extract
    for pdf_path in input_files:
        try:
            text = extract_text_from_pdf(pdf_path)
            out_name = pdf_path.stem + ".txt"
            out_path = run_dir / out_name
            out_path.write_text(text, encoding="utf-8")
            print(f"Extracted: {pdf_path.name} -> {out_path.relative_to(project_root)}")
        except Exception as exc:
            print(f"Failed to extract {pdf_path}: {exc}")

    # Also write a simple manifest
    manifest_path = run_dir / "manifest.txt"
    manifest_lines = [
        f"created_at: {timestamp}",
        f"project_root: {project_root}",
        f"source_count: {len(input_files)}",
        "sources:",
    ]
    for p in input_files:
        try:
            size_bytes = os.path.getsize(p)
        except OSError:
            size_bytes = -1
        manifest_lines.append(f"- {p} ({size_bytes} bytes)")
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()



