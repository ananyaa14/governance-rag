import time

from pipeline import RagPipeline, fuse, pages_label, POOL

QUESTION = "What does GOVERN 1.5 say?"


def show(title, ids, p, n=5):
    print(f"\n{title}")
    for r, cid in enumerate(ids[:n], start=1):
        c = p.by_id[cid]
        print(f"  #{r} {c['source']} {pages_label(c)}  {c['section'][:60]}")


t = time.perf_counter()
p = RagPipeline()
print(f"Load time: {time.perf_counter() - t:.1f} s")

for run in (1, 2, 3):
    t0 = time.perf_counter()
    vec = p.embedder.encode([QUESTION], normalize_embeddings=True).tolist()
    t1 = time.perf_counter()
    vector_ids = p.collection.query(query_embeddings=vec, n_results=POOL)["ids"][0]
    t2 = time.perf_counter()
    bm25_ids = p._bm25_ids(QUESTION)
    t3 = time.perf_counter()
    print(
        f"Run {run}: embed {1000 * (t1 - t0):.0f} ms | "
        f"chroma {1000 * (t2 - t1):.0f} ms | bm25 {1000 * (t3 - t2):.0f} ms"
    )

fused = fuse(vector_ids, bm25_ids)
show("VECTOR top 5", vector_ids, p)
show("BM25 top 5", bm25_ids, p)
show("FUSED top 5", fused, p)

target = [c["chunk_id"] for c in p.chunks if c["section"].startswith("GOVERN 1.5")]
print("\nGOVERN 1.5 chunks:", target)
for cid in target:
    print(
        cid,
        "| vector rank:", vector_ids.index(cid) + 1 if cid in vector_ids else None,
        "| bm25 rank:", bm25_ids.index(cid) + 1 if cid in bm25_ids else None,
        "| fused rank:", fused.index(cid) + 1 if cid in fused else None,
    )

cand = fused[:10]
for run in (1, 2):
    t = time.perf_counter()
    p.reranker.predict([[QUESTION, p.by_id[c]["text"]] for c in cand])
    print(f"Rerank run {run}: {1000 * (time.perf_counter() - t):.0f} ms")