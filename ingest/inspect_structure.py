import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pages = json.loads(
    (ROOT / "data" / "processed" / "pages_clean.json").read_text(encoding="utf-8")
)


def show(source, page_nos):
    for p in pages:
        if p["source"] == source and p["page"] in page_nos:
            lines = [l for l in p["text"].split("\n") if l.strip()]
            print(f"--- {source} p{p['page']} ({len(lines)} lines)")
            print("FIRST:", lines[:3])
            print("LAST: ", lines[-3:])


show("eu_ai_act.pdf", {40, 41})
show("nist_ai_rmf.pdf", {10, 11, 30})
show("nist_genai_profile.pdf", {10, 20})
show("oecd_ai_principles.pdf", {3, 6})

article = re.compile(r"^Article \d+\s*$", re.M)
eu = [p for p in pages if p["source"] == "eu_ai_act.pdf"]
print("\nEU lines that are just 'Article N':",
      sum(len(article.findall(p["text"])) for p in eu))

subcat = re.compile(r"^(GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+", re.M)
nist = [p for p in pages if p["source"] == "nist_ai_rmf.pdf"]
print("NIST subcategory starts:",
      sum(len(subcat.findall(p["text"])) for p in nist))