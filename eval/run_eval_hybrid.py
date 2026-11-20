import json
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
Q_FILE = ROOT / "eval" / (sys.argv[1] if len(sys.argv) > 1 else "questions_synth.json")
CHUNKS = json.loads(
    (ROOT / "data" / "processed" / "chunks_structured.json").read_text(encoding="utf-8")
)
K = 5        # results scored
POOL = 20    # candidates taken from each method before fusing
RRF_K = 60   # standard RRF constant

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")  # keeps numbers like 1.5 together


def tokens(text):
    return TOKEN.findall(text.lower())


BY_ID = {c["chunk_id"]: c for c in CHUNKS}
BM25 = BM25Okapi([tokens(c["text"]) for c in CHUNKS])


def bm25_ids(question):
    scores = BM25.get_scores(tokens(question))
    order = sorted(range(len(CHUNKS)), key=lambda i: -scores[i])[:POOL]
    return [CHUNKS[i]["chunk_id"] for i in order]


def fuse(*ranked_lists):
    scores = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda cid: -scores[cid])


def rank_of(question, ids):
    wanted = set(question["expected_pages"])
    for i, cid in enumerate(ids[:K], start=1):
        c = BY_ID[cid]
        pages = set(range(c["page"], c.get("page_end", c["page"]) + 1))
        if c["source"] == question["source"] and pages & wanted:
            return i
    return None


def summarise(ranks):
    n = len(ranks)
    return {
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        "hit@5": sum(1 for r in ranks if r) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


if __name__ == "__main__":
    questions = [
        q for q in json.loads(Q_FILE.read_text(encoding="utf-8"))
        if q["type"] == "answerable" and q["expected_pages"]
    ]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(
        [q["question"] for q in questions], normalize_embeddings=True
    ).tolist()
    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("structured")
    res = collection.query(query_embeddings=vectors, n_results=POOL)

    methods = {"vector": [], "bm25": [], "hybrid": []}
    for i, q in enumerate(questions):
        v = res["ids"][i]
        b = bm25_ids(q["question"])
        methods["vector"].append(v)
        methods["bm25"].append(b)
        methods["hybrid"].append(fuse(v, b))

    ranks = {m: [rank_of(q, ids) for q, ids in zip(questions, lists)] for m, lists in methods.items()}
    summary = {m: summarise(r) for m, r in ranks.items()}

    print(f"\n{len(questions)} answerable questions from {Q_FILE.name} (structured chunks)\n")
    print(f"{'method':<10}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    for m, s in summary.items():
        print(f"{m:<10}{s['hit@1']:>8.2f}{s['hit@3']:>8.2f}{s['hit@5']:>8.2f}{s['mrr']:>8.3f}")

    print("\nQuestions hybrid missed in its top 5:")
    for i, q in enumerate(questions):
        if ranks["hybrid"][i] is None:
            print(f"  {q['id']} wanted {q['source']} p{q['expected_pages']}: {q['question'][:110]}")

    print("\nQuestions where methods disagree (vector rank / bm25 rank / hybrid rank):")
    for i, q in enumerate(questions):
        r = (ranks["vector"][i], ranks["bm25"][i], ranks["hybrid"][i])
        if len(set(r)) > 1:
            print(f"  {q['id']}: {r[0]} / {r[1]} / {r[2]}   {q['question'][:80]}")

    out = ROOT / "eval" / "results_hybrid.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")