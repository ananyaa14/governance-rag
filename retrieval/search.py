import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "chroma_db"

if __name__ == "__main__":
    collection_name = sys.argv[1]
    question = " ".join(sys.argv[2:])
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([question], normalize_embeddings=True).tolist()

    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(collection_name)
    results = collection.query(query_embeddings=query_vec, n_results=5)

    print(f"[{collection_name}] {question}\n")
    for rank in range(5):
        meta = results["metadatas"][0][rank]
        dist = results["distances"][0][rank]
        text = results["documents"][0][rank]
        pages = meta["page"]
        if meta.get("page_end", pages) != pages:
            pages = f"{pages}-{meta['page_end']}"
        label = meta.get("section", "")
        print(f"#{rank + 1}  distance={dist:.3f}  {meta['source']} p{pages}  {label}")
        print("    " + text[:200].replace("\n", " "))