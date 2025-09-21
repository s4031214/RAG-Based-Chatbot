import os, json, math, time, sys
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

ROOT = os.path.dirname(os.path.dirname(__file__))
CHUNKS = os.path.join(ROOT, "data", "chunks", "kb_chunks.jsonl")
OUTDIR = os.path.join(ROOT, "data", "index")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384  # bge-small-en output dim

# BGE uses a query instruction for best results
QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "
PASSAGE_INSTRUCTION = "Represent this passage for retrieval: "

BATCH = 256

def read_chunks(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))
    return rows

def embed_passages(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # prepend passage instruction (improves BGE performance)
    texts_inst = [PASSAGE_INSTRUCTION + t for t in texts]
    embs = model.encode(
        texts_inst, 
        batch_size=64, 
        normalize_embeddings=True, 
        show_progress_bar=True
    )
    return np.asarray(embs, dtype="float32")

def main():
    if not os.path.exists(CHUNKS):
        print(f"[ERROR] Missing {CHUNKS}. Run Step 1 first.")
        sys.exit(1)

    print(f"Loading chunks from {CHUNKS} ...")
    rows = read_chunks(CHUNKS)
    if not rows:
        print("[ERROR] No chunks found.")
        sys.exit(1)

    texts = [r["text"] for r in rows]
    print(f"Total chunks: {len(texts)}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding passages...")
    embs = embed_passages(model, texts)  # (N, DIM)

    # Build FAISS index (L2 or IP; we normalized embeddings → use IP)
    index = faiss.IndexFlatIP(DIM)
    index.add(embs)

    # Save index
    faiss_path = os.path.join(OUTDIR, "faiss.index")
    faiss.write_index(index, faiss_path)

    # Save metadata sidecar (parallel lists)
    meta = {
        "model": MODEL_NAME,
        "dim": DIM,
        "normalize": True,
        "query_instruction": QUERY_INSTRUCTION,
        "passage_instruction": PASSAGE_INSTRUCTION,
        "count": len(rows),
        "created_at": int(time.time()),
    }
    with open(os.path.join(OUTDIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save rows (so we can map FAISS ids → chunk metadata)
    with open(os.path.join(OUTDIR, "chunks_meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Index built: {faiss_path}")
    print(f"   Metadata: data/index/meta.json")
    print(f"   Chunk map: data/index/chunks_meta.jsonl")

if __name__ == "__main__":
    main()
