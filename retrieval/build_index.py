import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks_baseline.json"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "baseline"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH = 64

if __name__ == "__main__":
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts, batch_size=BATCH, show_progress_bar=True, normalize_embeddings=True
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings[i:i + BATCH].tolist(),
            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
        )

    print(f"Indexed {collection.count()} chunks into '{COLLECTION}'")
    print(f"Embedding dimension: {embeddings.shape[1]}")