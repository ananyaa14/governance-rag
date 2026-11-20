import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_FILE = ROOT / "data" / "processed" / "pages.json"
CLEAN_FILE = ROOT / "data" / "processed" / "pages_clean.json"

BREAK = re.compile(r"(\w+)-\n(\w+)")                    # word split by hyphen at line end
MARKERS = re.compile(r"[\u25bc\u25ba\u25c4][A-Z]?\d*")  # amendment markers in the EU text
SOFT_HYPHEN = "\u00ad"

stats = {"joined": [], "kept": [], "header_lines": 0}


def normalise(text):
    # NFKC turns ligatures (fi, fl) and special spaces into plain characters
    text = unicodedata.normalize("NFKC", text)
    return text.replace(SOFT_HYPHEN + "\n", "").replace(SOFT_HYPHEN, "")


def build_vocab(pages):
    vocab = Counter()
    for p in pages:
        text = normalise(p["text"])
        text = BREAK.sub(" ", text)  # ignore broken pairs when counting words
        vocab.update(re.findall(r"[\w-]+", text.lower()))
    return vocab


def is_header_shape(norm):
    # a header has a varying number (page no.) or is just a rule line
    return "#" in norm or re.fullmatch(r"[_\-\s]+", norm) is not None


def find_repeated_lines(pages, min_share=0.4):
    by_source = {}
    for p in pages:
        by_source.setdefault(p["source"], []).append(p)
    repeated = {}
    for source, plist in by_source.items():
        counts = Counter()
        for p in plist:
            seen = set()
            for line in p["text"].split("\n"):
                norm = re.sub(r"\d+", "#", line.strip())
                if 12 <= len(norm) < 100 and is_header_shape(norm):
                    seen.add(norm)
            counts.update(seen)
        threshold = max(3, int(len(plist) * min_share))
        repeated[source] = {line for line, c in counts.items() if c >= threshold}
    return repeated


def fix_breaks(text, vocab):
    def repl(m):
        a, b = m.group(1), m.group(2)
        joined = (a + b).lower()
        hyphenated = (a + "-" + b).lower()
        if vocab[joined] > vocab[hyphenated]:
            stats["joined"].append(a + b)
            return a + b
        if vocab[joined] == 0 and vocab[hyphenated] == 0:
            # unseen elsewhere: join only if a fragment is not a real word
            if vocab[a.lower()] == 0 or vocab[b.lower()] == 0:
                stats["joined"].append(a + b)
                return a + b
        stats["kept"].append(a + "-" + b)
        return a + "-" + b

    return BREAK.sub(repl, text)


def clean_page(text, vocab, header_lines):
    text = normalise(text)
    text = MARKERS.sub("", text)
    kept_lines = []
    for line in text.split("\n"):
        norm = re.sub(r"\d+", "#", line.strip())
        if norm in header_lines:
            stats["header_lines"] += 1
            continue
        kept_lines.append(line)
    text = "\n".join(kept_lines)
    text = fix_breaks(text, vocab)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    pages = json.loads(PAGES_FILE.read_text(encoding="utf-8"))
    vocab = build_vocab(pages)
    repeated = find_repeated_lines(pages)

    cleaned = []
    for p in pages:
        new_text = clean_page(p["text"], vocab, repeated.get(p["source"], set()))
        if new_text:
            cleaned.append({**p, "text": new_text})

    CLEAN_FILE.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    before = sum(len(p["text"]) for p in pages)
    after = sum(len(p["text"]) for p in cleaned)
    ligatures = sum(
        1 for p in cleaned for ch in p["text"] if ch in "\ufb00\ufb01\ufb02\ufb03\ufb04"
    )
    print(f"Pages: {len(pages)} -> {len(cleaned)}")
    print(f"Characters: {before} -> {after}")
    print(f"Repeated header/footer lines removed: {stats['header_lines']}")
    print(f"Broken words rejoined: {len(stats['joined'])}  e.g. {stats['joined'][:8]}")
    print(f"Hyphens kept: {len(stats['kept'])}  e.g. {stats['kept'][:12]}")
    print(f"Ligature characters left in cleaned text: {ligatures}")