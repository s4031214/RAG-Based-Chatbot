import os, json, sys
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

ROOT = os.path.dirname(os.path.dirname(__file__))
IDXDIR = os.path.join(ROOT, "data", "index")
FAISS_PATH = os.path.join(IDXDIR, "faiss.index")
META_PATH = os.path.join(IDXDIR, "meta.json")
CHUNK_META_PATH = os.path.join(IDXDIR, "chunks_meta.jsonl")

def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_chunk_meta() -> List[dict]:
    rows = []
    with open(CHUNK_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def embed_query(model: SentenceTransformer, q: str, instruction: str) -> np.ndarray:
    q_full = instruction + q
    emb = model.encode([q_full], normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")

def search(query: str, k: int = 5):
    if not os.path.exists(FAISS_PATH):
        print("[ERROR] Missing index. Build it with 04_build_index.py")
        sys.exit(1)

    meta = load_meta()
    rows = load_chunk_meta()

    index = faiss.read_index(FAISS_PATH)
    model = SentenceTransformer(meta["model"])
    qv = embed_query(model, query, meta["query_instruction"])

    D, I = index.search(qv, k)   # cosine sim because we normalized (IP index)
    sims = D[0].tolist()
    idxs = I[0].tolist()

    results = []
    for sim, i in zip(sims, idxs):
        if i < 0: 
            continue
        r = rows[i]
        results.append((sim, r))
    return results

def format_snippet(text: str, n=260) -> str:
    t = " ".join(text.split())
    return t[:n] + ("…" if len(t) > n else "")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/05_search.py \"your question here\" [k]")
        sys.exit(0)
    q = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    hits = search(q, k=k)
    print(f"\nQuery: {q}\nTop-{k} results:\n")
    for rank, (sim, r) in enumerate(hits, 1):
        print(f"{rank}. score={sim:.3f} | {r['title']} [{r.get('role')}/{r.get('category')}]")
        print(f"   doc_id: {r['doc_id']}  chunk_id: {r['chunk_id']}")
        print(f"   url: {r.get('source_url')}")
        print(f"   snippet: {format_snippet(r['text'])}\n")
