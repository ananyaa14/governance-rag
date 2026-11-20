import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "chroma_db"
Q_FILE = ROOT / "eval" / (sys.argv[1] if len(sys.argv) > 1 else "questions_synth.json")
COLLECTIONS = ["baseline", "fixed_clean", "structured"]
K = 5


def pages_of(meta):
    return set(range(meta["page"], meta.get("page_end", meta["page"]) + 1))


def rank_of(question, metas):
    wanted = set(question["expected_pages"])
    for i, meta in enumerate(metas, start=1):
        if meta["source"] == question["source"] and pages_of(meta) & wanted:
            return i
    return None


def label(meta):
    pages = meta["page"]
    if meta.get("page_end", pages) != pages:
        pages = f"{pages}-{meta['page_end']}"
    return f"{meta['source']} p{pages}"


if __name__ == "__main__":
    questions = [
        q for q in json.loads(Q_FILE.read_text(encoding="utf-8"))
        if q["type"] == "answerable" and q["expected_pages"]
    ]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(
        [q["question"] for q in questions], normalize_embeddings=True
    ).tolist()
    client = chromadb.PersistentClient(path=str(DB_DIR))

    summary, ranks_by_index, metas_by_index = {}, {}, {}
    for name in COLLECTIONS:
        res = client.get_collection(name).query(query_embeddings=vectors, n_results=K)
        metas_by_index[name] = res["metadatas"]
        ranks = [rank_of(q, res["metadatas"][i]) for i, q in enumerate(questions)]
        ranks_by_index[name] = ranks
        n = len(questions)
        summary[name] = {
            "hit@1": sum(1 for r in ranks if r == 1) / n,
            "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
            "hit@5": sum(1 for r in ranks if r) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
        }

    print(f"\n{len(questions)} answerable questions from {Q_FILE.name}\n")
    print(f"{'index':<14}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    for name, s in summary.items():
        print(f"{name:<14}{s['hit@1']:>8.2f}{s['hit@3']:>8.2f}{s['hit@5']:>8.2f}{s['mrr']:>8.3f}")

    print("\nhit@5 by document")
    sources = sorted({q["source"] for q in questions})
    print(f"{'document':<26}" + "".join(f"{n:>13}" for n in COLLECTIONS) + f"{'n':>5}")
    for src in sources:
        idx = [i for i, q in enumerate(questions) if q["source"] == src]
        row = ""
        for name in COLLECTIONS:
            hits = sum(1 for i in idx if ranks_by_index[name][i])
            row += f"{hits / len(idx):>13.2f}"
        print(f"{src:<26}{row}{len(idx):>5}")

    print("\nQuestions the structured index missed in its top 5:")
    for i, q in enumerate(questions):
        if ranks_by_index["structured"][i] is None:
            top = [label(m) for m in metas_by_index["structured"][i][:3]]
            print(f"  {q['id']} wanted {q['source']} p{q['expected_pages']}: {q['question']}")
            print(f"       got: {top}")

    out = ROOT / "eval" / "results_latest.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")