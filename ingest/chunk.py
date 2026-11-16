import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_FILE = ROOT / "data" / "processed" / "pages.json"
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks_baseline.json"

CHUNK_SIZE = 200  # words per chunk
OVERLAP = 40      # words shared between neighbouring chunks


def chunk_page(page):
    words = page["text"].split()
    step = CHUNK_SIZE - OVERLAP
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_SIZE]
        if len(piece) < 30:  # skip tiny leftovers
            continue
        chunks.append(
            {
                "chunk_id": f"{page['source']}::p{page['page']}::{start}",
                "source": page["source"],
                "page": page["page"],
                "text": " ".join(piece),
            }
        )
    return chunks


if __name__ == "__main__":
    pages = json.loads(PAGES_FILE.read_text(encoding="utf-8"))
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page))

    CHUNKS_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Total chunks: {len(all_chunks)}")
    by_source = {}
    for c in all_chunks:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    for source, count in by_source.items():
        print(f"  {source}: {count}")
    print("--- sample chunk ---")
    print(all_chunks[100]["chunk_id"])
    print(all_chunks[100]["text"])