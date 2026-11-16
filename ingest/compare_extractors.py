from pathlib import Path
from pypdf import PdfReader
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
pdf_path = ROOT / "data" / "raw" / "eu_ai_act.pdf"
page_no = 34  # same page as the sample chunk

pypdf_text = PdfReader(pdf_path).pages[page_no - 1].extract_text()
doc = pymupdf.open(pdf_path)
pymupdf_text = doc[page_no - 1].get_text()

print("=== PYPDF ===")
print(pypdf_text[:700])
print()
print("=== PYMUPDF ===")
print(pymupdf_text[:700])