import json
import re
import sys
import time
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks_structured.json"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "structured"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "claude-sonnet-5-5"

POOL = 20         # candidates taken from each retrieval method
RERANK_POOL = 10  # fused candidates the reranker re-reads
TOP_K = 5         # chunks given to Claude
RRF_K = 60
MAX_EXACT = 4     # most chunks an identifier lookup may add

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")  # keeps numbers like 1.5 together
IDENT = re.compile(
    r"\b(GOVERN|MAP|MEASURE|MANAGE)\s+(\d+\.\d+)"
    r"|\bArticle\s+(\d+[a-z]?)\b"
    r"|\bAnnex\s+([IVXL]+)\b",
    re.I,
)

SYSTEM_PROMPT = """You answer questions about AI governance documents using ONLY the numbered context passages provided.
Rules:
- Base every statement on the passages. Do not use outside knowledge.
- Cite the passages you used like [1] or [2][3] after each claim.
- If the passages do not contain the answer, reply exactly: "I can't find this in the provided documents."
- If the passages only partly answer the question, answer the part they support and say what is missing.
- Be concise."""


def tokens(text):
    return TOKEN.findall(text.lower())


def fuse(*ranked_lists):
    scores = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda cid: -scores[cid])


def pages_label(chunk):
    end = chunk.get("page_end", chunk["page"])
    return f"p{chunk['page']}" if end == chunk["page"] else f"p{chunk['page']}-{end}"


def build_context(hits):
    blocks = []
    for n, h in enumerate(hits, start=1):
        blocks.append(
            f"[{n}] ({h['source']}, {pages_label(h)}, {h['section']})\n{h['text']}"
        )
    return "\n\n".join(blocks)


def identifiers(question):
    labels = []
    for fn, num, art, annex in IDENT.findall(question):
        if fn:
            labels.append(f"{fn.upper()} {num}")
        elif art:
            labels.append(f"Article {art}")
        elif annex:
            labels.append(f"ANNEX {annex.upper()}")
    return labels


class RagPipeline:
    def __init__(self):
        self.chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokens(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
        self.client = anthropic.Anthropic()
        self._warm_up()

    def _warm_up(self):
        # the first Chroma query loads its index from disk and can take a long time
        vec = self.embedder.encode(["warm up"], normalize_embeddings=True).tolist()
        self.collection.query(query_embeddings=vec, n_results=1)
        self.reranker.predict([["warm up", "warm up"]])

    def _bm25_ids(self, question):
        scores = self.bm25.get_scores(tokens(question))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:POOL]
        return [self.chunks[i]["chunk_id"] for i in order]

    def _exact_ids(self, question):
        found = []
        for label in identifiers(question):
            pattern = re.compile(re.escape(label) + r"(?![\w.])", re.I)
            for c in self.chunks:
                if pattern.match(c["section"]) and c["chunk_id"] not in found:
                    found.append(c["chunk_id"])
        return found[:MAX_EXACT]

    def retrieve(self, question, rerank=True, k=TOP_K):
        start = time.perf_counter()
        vec = self.embedder.encode([question], normalize_embeddings=True).tolist()
        vector_ids = self.collection.query(query_embeddings=vec, n_results=POOL)["ids"][0]
        fused = fuse(vector_ids, self._bm25_ids(question))
        exact = self._exact_ids(question)
        fused = exact + [c for c in fused if c not in exact]
        if rerank:
            cand = fused[:RERANK_POOL]
            scores = self.reranker.predict(
                [[question, self.by_id[c]["text"]] for c in cand]
            )
            order = sorted(range(len(cand)), key=lambda j: -scores[j])
            ids = [cand[j] for j in order]
        else:
            ids = fused
        hits = [self.by_id[c] for c in ids[:k]]
        return hits, (time.perf_counter() - start) * 1000

    def answer(self, question, rerank=True):
        hits, retrieval_ms = self.retrieve(question, rerank)
        start = time.perf_counter()
        response = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {question}",
                }
            ],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return {
            "answer": text,
            "hits": hits,
            "retrieval_ms": retrieval_ms,
            "generation_ms": (time.perf_counter() - start) * 1000,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fast"]
    fast = "--fast" in sys.argv
    question = " ".join(args)

    pipeline = RagPipeline()
    result = pipeline.answer(question, rerank=not fast)

    print("ANSWER:\n" + result["answer"])
    print("\nSOURCES:")
    for n, h in enumerate(result["hits"], start=1):
        print(f"  [{n}] {h['source']} {pages_label(h)}  {h['section']}")
    mode = "hybrid" if fast else "hybrid + rerank"
    print(
        f"\nMode: {mode} | retrieval {result['retrieval_ms']:.0f} ms | "
        f"generation {result['generation_ms']:.0f} ms | "
        f"tokens {result['input_tokens']} in, {result['output_tokens']} out"
    )