import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH = 64

if __name__ == "__main__":
    chunks_name = sys.argv[1] if len(sys.argv) > 1 else "chunks_baseline"
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "baseline"

    chunks_file = ROOT / "data" / "processed" / f"{chunks_name}.json"
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)
    print(f"Embedding model reads at most {model.max_seq_length} tokens per chunk")
    embeddings = model.encode(
        texts, batch_size=BATCH, show_progress_bar=True, normalize_embeddings=True
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings[i:i + BATCH].tolist(),
            metadatas=[
                {
                    "source": c["source"],
                    "page": c["page"],
                    "page_end": c.get("page_end", c["page"]),
                    "section": c.get("section", ""),
                }
                for c in batch
            ],
        )

    print(f"Indexed {collection.count()} chunks from '{chunks_name}' into '{collection_name}'")
    print(f"Embedding dimension: {embeddings.shape[1]}")