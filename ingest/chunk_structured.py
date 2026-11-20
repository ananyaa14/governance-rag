import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_FILE = ROOT / "data" / "processed" / "pages_clean.json"
OUT_FILE = ROOT / "data" / "processed" / "chunks_structured.json"

MAX_WORDS = 170
MIN_TAIL_WORDS = 40

ARTICLE = re.compile(r"^Article (\d+)([a-z]?)$")
ANNEX = re.compile(r"^ANNEX [IVXL]+$")
SUBCAT = re.compile(r"^(GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+")
DOTTED = re.compile(r"^\d+(?:\.\d+)+\.?\s+[A-Z].{2,70}$")
PAGE_ONLY = re.compile(r"^(Page\s+)?\d+$")


def page_lines(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines and lines[0] == "NIST AI 100-1":
        lines = lines[1:]
        if lines and lines[0] == "AI RMF 1.0":
            lines = lines[1:]
    if lines and PAGE_ONLY.match(lines[0]):
        lines = lines[1:]
    if lines and PAGE_ONLY.match(lines[-1]):
        lines = lines[:-1]
    return lines


def make_detector(source):
    state = {"last": 0}

    def detect(texts, i):
        line = texts[i]
        if source == "eu_ai_act.pdf":
            m = ARTICLE.match(line)
            if m:
                n, suffix = int(m.group(1)), m.group(2)
                in_order = (not suffix and n == state["last"] + 1) or (
                    suffix and n == state["last"]
                )
                if in_order:
                    state["last"] = n
                    nxt = texts[i + 1] if i + 1 < len(texts) else ""
                    return f"Article {n}{suffix} - {nxt}"
            if ANNEX.match(line):
                nxt = texts[i + 1] if i + 1 < len(texts) else ""
                return f"{line} - {nxt}"
            return None
        if SUBCAT.match(line):
            return line[:90]
        if DOTTED.match(line):
            return line
        return None

    return detect


def build_sections(source, lines):
    detect = make_detector(source)
    texts = [t for _, t in lines]
    sections = []
    current = {"title": "Front matter", "items": []}
    for i, item in enumerate(lines):
        title = detect(texts, i)
        if title:
            sections.append(current)
            current = {"title": title, "items": []}
        current["items"].append(item)
    sections.append(current)
    return sections


def split_items(items):
    groups, buf, count = [], [], 0
    for page, text in items:
        buf.append((page, text))
        count += len(text.split())
        if count >= MAX_WORDS:
            cut, run = len(buf), 0
            best = None
            for j, (_, t) in enumerate(buf):
                run += len(t.split())
                if t.endswith((".", ";", ":")) and run >= MAX_WORDS * 0.5:
                    best = j + 1
            if best:
                cut = best
            groups.append(buf[:cut])
            buf = buf[cut:]
            count = sum(len(t.split()) for _, t in buf)
    if buf:
        groups.append(buf)
    if len(groups) > 1 and sum(len(t.split()) for _, t in groups[-1]) < MIN_TAIL_WORDS:
        groups[-2].extend(groups[-1])
        groups.pop()
    return groups


def chunk_source(source, lines):
    chunks = []
    for si, sec in enumerate(build_sections(source, lines)):
        if sum(len(t.split()) for _, t in sec["items"]) < 8:
            continue
        for gi, group in enumerate(split_items(sec["items"])):
            body = " ".join(t for _, t in group)
            text = body if gi == 0 else f"{sec['title']} (cont.)\n{body}"
            chunks.append(
                {
                    "chunk_id": f"{source}::s{si}::c{gi}",
                    "source": source,
                    "page": group[0][0],
                    "page_end": group[-1][0],
                    "section": sec["title"],
                    "text": text,
                }
            )
    return chunks


def show(chunks, needle):
    for c in chunks:
        if c["section"].startswith(needle):
            print(f"\n[{c['chunk_id']}] pages {c['page']}-{c['page_end']} | {c['section']}")
            print(c["text"][:600])
            return
    print(f"\n(no section starting with '{needle}')")


if __name__ == "__main__":
    pages = json.loads(PAGES_FILE.read_text(encoding="utf-8"))
    by_source = {}
    for p in pages:
        by_source.setdefault(p["source"], []).extend(
            (p["page"], t) for t in page_lines(p["text"])
        )

    all_chunks = []
    for source, lines in by_source.items():
        all_chunks.extend(chunk_source(source, lines))

    OUT_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sizes = [len(c["text"].split()) for c in all_chunks]
    print(f"Total chunks: {len(all_chunks)}")
    for source in by_source:
        n = sum(1 for c in all_chunks if c["source"] == source)
        print(f"  {source}: {n}")
    print(
        f"Words per chunk: min {min(sizes)}, median {int(statistics.median(sizes))}, max {max(sizes)}"
    )

    eu = [c for c in all_chunks if c["source"] == "eu_ai_act.pdf"]
    nums = {
        int(m.group(1))
        for c in eu
        for m in [re.match(r"Article (\d+)", c["section"])]
        if m
    }
    print(f"EU articles found: {len(nums)} (highest {max(nums)})")
    print("EU article numbers missing:", [n for n in range(1, max(nums) + 1) if n not in nums])
    print("EU annex sections:", sorted({c["section"] for c in eu if c["section"].startswith("ANNEX")}))

    show(all_chunks, "Article 99")
    show(all_chunks, "Article 16")
    show(all_chunks, "GOVERN 1.5")