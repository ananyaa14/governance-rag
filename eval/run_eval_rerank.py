import json
import os
import time

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from run_eval_hybrid import (
    BY_ID, POOL, Q_FILE, ROOT, bm25_ids, fuse, rank_of, summarise,
)

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = int(os.environ.get("RERANK_POOL", "20"))

if __name__ == "__main__":
    questions = [
        q for q in json.loads(Q_FILE.read_text(encoding="utf-8"))
        if q["type"] == "answerable" and q["expected_pages"]
    ]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder(RERANK_MODEL)
    vectors = embedder.encode(
        [q["question"] for q in questions], normalize_embeddings=True
    ).tolist()
    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("structured")
    res = collection.query(query_embeddings=vectors, n_results=POOL)

    hybrid_lists, rerank_lists, times = [], [], []
    for i, q in enumerate(questions):
        fused = fuse(res["ids"][i], bm25_ids(q["question"]))[:RERANK_POOL]
        start = time.perf_counter()
        scores = reranker.predict([[q["question"], BY_ID[c]["text"]] for c in fused])
        times.append(time.perf_counter() - start)
        order = sorted(range(len(fused)), key=lambda j: -scores[j])
        hybrid_lists.append(fused)
        rerank_lists.append([fused[j] for j in order])

    r_h = [rank_of(q, ids) for q, ids in zip(questions, hybrid_lists)]
    r_r = [rank_of(q, ids) for q, ids in zip(questions, rerank_lists)]
    summary = {"hybrid": summarise(r_h), "hybrid+rerank": summarise(r_r)}

    print(f"\n{len(questions)} answerable questions, rerank model {RERANK_MODEL}")
    print(f"Reranking the top {RERANK_POOL} hybrid candidates\n")
    print(f"{'method':<16}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    for m, s in summary.items():
        print(f"{m:<16}{s['hit@1']:>8.2f}{s['hit@3']:>8.2f}{s['hit@5']:>8.2f}{s['mrr']:>8.3f}")
    print(f"\nRerank time per question: avg {1000 * sum(times) / len(times):.0f} ms, max {1000 * max(times):.0f} ms")

    print("\nQuestions whose result changed (hybrid rank -> rerank rank):")
    for i, q in enumerate(questions):
        if r_h[i] != r_r[i]:
            print(f"  {q['id']}: {r_h[i]} -> {r_r[i]}   {q['question'][:80]}")

    print("\nStill missed after reranking:")
    for i, q in enumerate(questions):
        if r_r[i] is None:
            print(f"  {q['id']} wanted {q['source']} p{q['expected_pages']}")

    out = ROOT / "eval" / f"results_rerank_pool{RERANK_POOL}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")