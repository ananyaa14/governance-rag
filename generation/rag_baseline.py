import sys
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "chroma_db"
TOP_K = 5
LLM_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You answer questions about AI governance documents using ONLY the numbered context passages provided.
Rules:
- Base every statement on the passages. Do not use outside knowledge.
- Cite the passages you used like [1] or [2][3] after each claim.
- If the passages do not contain the answer, reply exactly: "I can't find this in the provided documents."
- Be concise."""


def retrieve(question, embed_model, collection, k=TOP_K):
    query_vec = embed_model.encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=query_vec, n_results=k)
    hits = []
    for i in range(len(res["documents"][0])):
        hits.append(
            {
                "text": res["documents"][0][i],
                "source": res["metadatas"][0][i]["source"],
                "page": res["metadatas"][0][i]["page"],
                "distance": res["distances"][0][i],
            }
        )
    return hits


def build_context(hits):
    blocks = []
    for n, h in enumerate(hits, start=1):
        blocks.append(f"[{n}] ({h['source']}, page {h['page']})\n{h['text']}")
    return "\n\n".join(blocks)


def answer(question, embed_model, collection, client):
    hits = retrieve(question, embed_model, collection)
    context = build_context(hits)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    text = "".join(
    block.text for block in response.content if block.type == "text"
    )
    return text, hits, response.usage


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection("baseline")
    client = anthropic.Anthropic()

    text, hits, usage = answer(question, embed_model, collection, client)
    print("ANSWER:\n" + text)
    print("\nSOURCES:")
    for n, h in enumerate(hits, start=1):
        print(f"  [{n}] {h['source']} p{h['page']}  (distance {h['distance']:.3f})")
    print(f"\nTokens: {usage.input_tokens} in, {usage.output_tokens} out")