import json
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "processed" / "pages.json"


def load_pdfs():
    pages = []
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        doc = pymupdf.open(pdf_path)
        kept = 0
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append(
                    {"source": pdf_path.name, "page": page_number, "text": text}
                )
                kept += 1
        print(f"{pdf_path.name}: {len(doc)} pages, {kept} with text")
    return pages


if __name__ == "__main__":
    pages = load_pdfs()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(pages)} pages to {OUT_FILE}")