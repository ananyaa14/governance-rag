import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "chroma_db"

if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([question], normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection("baseline")
    results = collection.query(query_embeddings=query_vec, n_results=5)

    for rank in range(5):
        meta = results["metadatas"][0][rank]
        dist = results["distances"][0][rank]
        text = results["documents"][0][rank]
        print(f"#{rank + 1}  distance={dist:.3f}  {meta['source']} p{meta['page']}")
        print(text[:300].replace("\n", " "))
        print()