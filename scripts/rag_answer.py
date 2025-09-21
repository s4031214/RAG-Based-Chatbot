import os, json, sys, re, time
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import ollama
import regex as regex

ROOT = os.path.dirname(os.path.dirname(__file__))
IDXDIR = os.path.join(ROOT, "data", "index")
FAISS_PATH = os.path.join(IDXDIR, "faiss.index")
META_PATH = os.path.join(IDXDIR, "meta.json")
CHUNK_META_PATH = os.path.join(IDXDIR, "chunks_meta.jsonl")

# ------- Tunables -------
TOP_K = 8               # retrieve many
CONTEXT_K = 4           # feed the top-k (after rerank / or just first 4)
MODEL_OLLAMA = "llama3.1:8b"  # local LLM in Ollama
MAX_CONTEXT_CHARS = 5500      # keep prompt within model limits
# ------------------------

def load_meta() -> Dict:
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_chunk_meta() -> List[Dict]:
    rows = []
    with open(CHUNK_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def embed_query(q: str, model_name: str, instruction: str) -> np.ndarray:
    st = SentenceTransformer(model_name)
    q_full = instruction + q
    emb = st.encode([q_full], normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")

def search(query: str, k: int = TOP_K) -> List[Tuple[float, Dict]]:
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError("Missing FAISS index. Build it in Step 2.")
    meta = load_meta()
    rows = load_chunk_meta()
    index = faiss.read_index(FAISS_PATH)

    qv = embed_query(query, meta["model"], meta["query_instruction"])
    D, I = index.search(qv, k)
    sims, idxs = D[0].tolist(), I[0].tolist()

    hits = []
    for sim, i in zip(sims, idxs):
        if i < 0: 
            continue
        r = rows[i]
        r["_score"] = sim
        hits.append((sim, r))
    return hits

# Optional micro-rerank (score by a simple heuristic: prefer AU + role coverage + similarity)
def simple_rerank(hits: List[Tuple[float, Dict]], prefer_jurisdiction="AU") -> List[Tuple[float, Dict]]:
    def key(item):
        sim, r = item
        bonus = 0.0
        if r.get("jurisdiction") == prefer_jurisdiction: bonus += 0.02
        # prefer chunks with titles and shorter text (often less noisy)
        tlen = len(r.get("text",""))
        bonus += 0.01 if tlen < 1200 else 0.0
        return sim + bonus
    return sorted(hits, key=key, reverse=True)

SYSTEM_INSTRUCTIONS = """You are a careful support assistant for a food delivery platform in Australia.
You must answer ONLY using the provided context snippets. 
Rules:
- If the context does not contain an answer, say: "I don’t know based on our knowledge base." Then show 2–3 most relevant article titles with links.
- Always include citations in square brackets using chunk_id(s), e.g., [customer-refunds-au--1].
- Keep answers concise and actionable. Call out country-specific details when present (AU).
- Never invent policies, fees, timelines, or phone numbers. Never include personal or payment data.
"""

USER_TEMPLATE = """Question:
{question}

Context snippets (each has id, title, role/category, url, and text):
{context}

Now answer the question using only the context above. 
If multiple snippets disagree, prefer the most jurisdiction-appropriate (e.g., AU). 
After the answer, include a line 'Sources:' listing the source_url(s) and cited chunk_id(s)."""

def format_context(hits: List[Tuple[float, Dict]], limit_chars=MAX_CONTEXT_CHARS) -> List[Dict]:
    used = []
    total = 0
    for sim, r in hits:
        block = (
            f"[{r['chunk_id']}] title={r.get('title')} | role={r.get('role')}/{r.get('category')} | "
            f"url={r.get('source_url')} | jurisdiction={r.get('jurisdiction')} | score={sim:.3f}\n"
            + r.get("text","").strip()
        )
        if total + len(block) > limit_chars:
            break
        used.append({"chunk": block, "row": r})
        total += len(block)
    return used

def build_prompt(question: str, used_ctx: List[Dict]) -> Dict:
    context_str = "\n\n---\n\n".join([u["chunk"] for u in used_ctx])
    user = USER_TEMPLATE.format(question=question, context=context_str)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user}
        ],
        "model": MODEL_OLLAMA,
        "options": {"temperature": 0.2, "num_ctx": 8192}
    }

# Basic PII redaction (very simple; extend later)
RE_EMAIL = regex.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=regex.I)
RE_PHONE = regex.compile(r"\b(?:\+?\d[\s-]?){6,}\b")

def redact_pii(text: str) -> str:
    text = RE_EMAIL.sub("[redacted-email]", text)
    text = RE_PHONE.sub("[redacted-phone]", text)
    return text

def extract_citations(text: str) -> List[str]:
    # citations appear like [doc--idx]; pick all inside brackets that match
    return sorted(set(regex.findall(r"\[([A-Za-z0-9._:-]+--\d+)\]", text)))

def fallback_sources(used_ctx: List[Dict], n=3) -> str:
    uniq = []
    seen = set()
    for u in used_ctx:
        r = u["row"]
        k = (r.get("title"), r.get("source_url"))
        if k not in seen:
            uniq.append(f"- {r.get('title')} — {r.get('source_url')}")
            seen.add(k)
        if len(uniq) >= n:
            break
    return "\n".join(uniq)

def answer(question: str, top_k: int = TOP_K, context_k: int = CONTEXT_K) -> Dict:
    hits = search(question, k=top_k)
    hits = simple_rerank(hits)
    used_ctx = format_context(hits[:context_k])

    prompt = build_prompt(question, used_ctx)
    resp = ollama.chat(**prompt)
    content = resp["message"]["content"].strip()
    content = redact_pii(content)

    # If the model failed to cite or says it doesn't know, force a graceful fallback
    cits = extract_citations(content)
    if not cits or "I don’t know" in content or "I don't know" in content:
        content = ("I don’t know based on our knowledge base.\n\n"
                   "Here are some relevant articles:\n" + fallback_sources(used_ctx))
        cits = []

    # Collect unique source URLs for the cited chunks
    cited_urls = []
    if cits:
        # map chunk_id -> url
        id2url = {u["row"]["chunk_id"]: u["row"]["source_url"] for u in used_ctx}
        for cid in cits:
            url = id2url.get(cid)
            if url and url not in cited_urls:
                cited_urls.append(url)

    return {
        "question": question,
        "answer": content,
        "citations": cits,
        "sources": cited_urls,
        "used_chunks": [u["row"] for u in used_ctx]
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/06_rag_answer.py \"your question here\"")
        sys.exit(0)
    q = sys.argv[1]
    t0 = time.time()
    out = answer(q)
    dt = time.time() - t0

    print("\n================ RAG Answer ================\n")
    print(out["answer"])
    if out["citations"]:
        print("\nCitations:", ", ".join(out["citations"]))
    if out["sources"]:
        print("Sources:")
        for u in out["sources"]:
            print(" -", u)
    print(f"\nLatency: {dt:.2f}s")
    print("\n(Used context chunks)")
    for uc in out["used_chunks"]:
        print(f" - {uc['chunk_id']} | {uc.get('title')} | {uc.get('source_url')}")

if __name__ == "__main__":
    main()
