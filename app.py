import os, json, time
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import ollama
import regex as re
import ollama
import httpx
from openai import OpenAI


# ---- Paths ----
ROOT = os.path.dirname(__file__)
IDXDIR = os.path.join(ROOT, "data", "index")
FAISS_PATH = os.path.join(IDXDIR, "faiss.index")
META_PATH = os.path.join(IDXDIR, "meta.json")
CHUNK_META_PATH = os.path.join(IDXDIR, "chunks_meta.jsonl")
FEEDBACK_PATH = os.path.join(ROOT, "data", "eval", "feedback.csv")
os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

# ---- Prompt & parsing ----
SYSTEM_INSTRUCTIONS = """You are a careful support assistant for a food delivery platform in Australia.
Answer ONLY using the provided context snippets.
- If the context does not contain an answer, say: "I don’t know based on our knowledge base." Then show 2–3 relevant article titles with links.
- Always include citations in square brackets using chunk_id(s), e.g., [customer-refunds-au--1].
- Keep answers concise and actionable. Prefer AU details when present.
- Never invent policies, fees, timelines, contacts, or PII.
"""
USER_TEMPLATE = """Question:
{question}

Context snippets (id | title | role/category | url | jurisdiction | score):
{context}

Answer the question using only the context above. 
Prefer AU details when present. Always include citations in [chunk_id] form and end with a 'Sources:' line listing URLs and chunk_ids.
"""

CIT_RE = re.compile(r"\[([A-Za-z0-9._:-]+--\d+)\]")

# ---- Helpers ----
@st.cache_resource
def load_artifacts():
    if not os.path.exists(FAISS_PATH):
        st.error("Missing FAISS index. Run Step 2.")
        st.stop()
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    rows = []
    with open(CHUNK_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    index = faiss.read_index(FAISS_PATH)
    embedder = SentenceTransformer(meta["model"])
    return meta, rows, index, embedder

def embed_query(st_model, q, instruction):
    q_full = (instruction or "") + q
    v = st_model.encode([q_full], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

def search(query, top_k, meta, rows, index, embedder):
    qv = embed_query(embedder, query, meta.get("query_instruction",""))
    D, I = index.search(qv, top_k)
    sims, idxs = D[0].tolist(), I[0].tolist()
    hits = []
    for sim, i in zip(sims, idxs):
        if i < 0: 
            continue
        r = dict(rows[i])
        r["_score"] = sim
        hits.append((sim, r))
    # tiny heuristic rerank: prefer AU, shorter chunks slightly
    hits.sort(key=lambda sr: sr[0] + (0.02 if sr[1].get("jurisdiction")=="AU" else 0.0) + (0.01 if len(sr[1].get("text",""))<1200 else 0.0), reverse=True)
    return hits

def format_context(hits, context_k, limit_chars=5500):
    used, total, blocks = [], 0, []
    for sim, r in hits[:context_k]:
        block = (f"[{r['chunk_id']}] title={r.get('title')} | role={r.get('role')}/{r.get('category')} | "
                 f"url={r.get('source_url')} | jurisdiction={r.get('jurisdiction')} | score={sim:.3f}\n"
                 + (r.get("text","").strip()))
        if total + len(block) > limit_chars: break
        used.append(r)
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks), used

def call_llm(model_name, question, context):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": USER_TEMPLATE.format(question=question, context=context)}
        ],
        "options": {"temperature": 0.2, "num_ctx": 8192}
    }
    resp = ollama.chat(**payload)
    return resp["message"]["content"].strip()

def extract_citations(answer: str):
    return sorted(set(CIT_RE.findall(answer)))

def record_feedback(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(FEEDBACK_PATH):
        df0 = pd.read_csv(FEEDBACK_PATH)
        df = pd.concat([df0, df], ignore_index=True)
    df.to_csv(FEEDBACK_PATH, index=False)

# ---- UI ----
st.set_page_config(page_title="RAG UberEats Support (Demo)", layout="wide")
st.title("🍔 UberEats RAG Based Chatbot(Demo)")

meta, rows, index, embedder = load_artifacts()

col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    model_name = st.selectbox("Answer model (Ollama)", ["llama3.1:8b","mistral:7b"], index=0)
with col2:
    top_k = st.slider("Retrieval k", 4, 16, 8, step=1)
with col3:
    context_k = st.slider("Context k", 2, 8, 4, step=1)
with col4:
    st.caption(f"Embedder: {meta.get('model')}")

q = st.text_input("Ask a question", placeholder="e.g., How do I get a refund for missing items in Australia?")
go = st.button("Answer")

if "history" not in st.session_state:
    st.session_state.history = []

if go and q.strip():
    t0 = time.time()
    hits = search(q.strip(), top_k, meta, rows, index, embedder)
    ctx, used = format_context(hits, context_k=context_k)
    ans = call_llm(model_name, q.strip(), ctx)
    latency = time.time() - t0
    cits = extract_citations(ans)
    src_urls = []
    id2url = {r["chunk_id"]: r.get("source_url") for r in used}
    for cid in cits:
        u = id2url.get(cid)
        if u and u not in src_urls: src_urls.append(u)

    st.session_state.history.append({
        "q": q.strip(),
        "a": ans,
        "citations": cits,
        "urls": src_urls,
        "used": used,
        "latency": latency,
        "model": model_name,
        "top_k": top_k,
        "context_k": context_k
    })

# Show chat history
for i, turn in enumerate(reversed(st.session_state.history), 1):
    st.markdown(f"**You:** {turn['q']}")
    st.markdown(turn["a"])
    with st.expander("Sources & context"):
        if turn["urls"]:
            st.markdown("**Sources:**")
            for u in turn["urls"]:
                st.markdown(f"- [{u}]({u})")
        st.markdown("**Used chunks:**")
        for r in turn["used"]:
            st.code(f"{r['chunk_id']} | {r.get('title')} | {r.get('source_url')}")
    st.caption(f"Model: {turn['model']} • k={turn['top_k']}/{turn['context_k']} • Latency: {turn['latency']:.2f}s")

    fb_cols = st.columns(3)
    with fb_cols[0]:
        if st.button("👍 Helpful", key=f"up_{i}"):
            record_feedback({
                "ts": time.time(),
                "question": turn["q"],
                "model": turn["model"],
                "top_k": turn["top_k"],
                "context_k": turn["context_k"],
                "latency": turn["latency"],
                "helpful": 1,
                "citations": "|".join(turn["citations"]),
                "urls": "|".join(turn["urls"])
            })
            st.success("Thanks for the feedback!")
    with fb_cols[1]:
        if st.button("👎 Not helpful", key=f"down_{i}"):
            record_feedback({
                "ts": time.time(),
                "question": turn["q"],
                "model": turn["model"],
                "top_k": turn["top_k"],
                "context_k": turn["context_k"],
                "latency": turn["latency"],
                "helpful": 0,
                "citations": "|".join(turn["citations"]),
                "urls": "|".join(turn["urls"])
            })
            st.info("Feedback saved.")
    st.divider()

st.sidebar.header("Quick tips")
st.sidebar.write("- Keep your KB AU-specific to reduce contradictory policies.")
st.sidebar.write("- If answers look thin, raise context_k or increase chunk size (rebuild index).")
st.sidebar.write("- Compare models by switching the selector and asking the same question.")