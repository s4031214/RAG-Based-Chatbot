#!/usr/bin/env python3
"""
Evaluate RAG answer quality across multiple Ollama models.

Usage examples
-------------
# Basic (no judge):
python scripts/evaluate.py \
  --models llama3.1:8b mistral:7b \
  --gold data/eval/questions.jsonl \
  --outprefix baseline --top_k 8 --context_k 4

# With LLM-as-judge (local):
python scripts/evaluate.py \
  --models llama3.1:8b \
  --gold data/eval/questions.jsonl \
  --outprefix judged --top_k 8 --context_k 4 \
  --judge_model mistral:7b
"""
import os, sys, json, time, math, argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import faiss
import regex as re
from sentence_transformers import SentenceTransformer
import ollama

# ---------------- Paths ----------------
ROOT = os.path.dirname(os.path.dirname(__file__))
IDXDIR = os.path.join(ROOT, "data", "index")
FAISS_PATH = os.path.join(IDXDIR, "faiss.index")
META_PATH = os.path.join(IDXDIR, "meta.json")
CHUNK_META_PATH = os.path.join(IDXDIR, "chunks_meta.jsonl")

DEFAULT_GOLD = os.path.join(ROOT, "data", "eval", "questions.jsonl")
OUTDIR = os.path.join(ROOT, "data", "eval", "runs")
os.makedirs(OUTDIR, exist_ok=True)

# ------------- Defaults / knobs -------------
MAX_CONTEXT_CHARS = 5500
ACC_SIM_THRESHOLD = 0.80  # semantic accuracy threshold
UNANSWERED_PHRASES = [
    "I don’t know based on our knowledge base",
    "I don't know based on our knowledge base",
    "I don’t know",
    "I don't know"
]
CIT_RE = re.compile(r"\[([A-Za-z0-9._:-]+--\d+)\]")

SYSTEM_INSTRUCTIONS = """You are a careful support assistant for a food delivery platform in Australia.
Answer ONLY using the provided context snippets.
- If the context does not contain an answer, say: "I don’t know based on our knowledge base." Then show 2–3 relevant article titles with links.
- Always include citations in square brackets using chunk_id(s), e.g., [customer-refunds-au--1].
- Keep answers concise and actionable. Prefer AU details when present.
- Never invent policies, fees, timelines, contacts, or PII.
"""
USER_TEMPLATE = """Question:
{question}

Context snippets:
{context}

Now answer the question using only the context above.
If multiple snippets disagree, prefer the most jurisdiction-appropriate (AU).
After the answer, include a line 'Sources:' listing the source_url(s) and cited chunk_id(s)."""

# ---------------- I/O helpers ----------------
def load_meta() -> Dict:
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_chunk_meta() -> List[Dict]:
    rows = []
    with open(CHUNK_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_gold(path: str) -> List[Dict]:
    exs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                exs.append(json.loads(line))
    return exs

# ---------------- Embedding helpers ----------------
def get_embedder(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)

def embed_query(st: SentenceTransformer, q: str, instruction: str) -> np.ndarray:
    q_full = (instruction or "") + q
    v = st.encode([q_full], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

def embed_passage(st: SentenceTransformer, t: str, passage_instruction: Optional[str]) -> np.ndarray:
    if passage_instruction:
        t = passage_instruction + t
    v = st.encode([t], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

# ---------------- Retrieval ----------------
def search(index, rows, st: SentenceTransformer, meta: Dict, query: str, k: int) -> List[Tuple[float, Dict]]:
    qv = embed_query(st, query, meta.get("query_instruction", ""))
    D, I = index.search(qv, k)
    sims, idxs = D[0].tolist(), I[0].tolist()
    hits = []
    for sim, i in zip(sims, idxs):
        if i < 0: 
            continue
        r = dict(rows[i])
        r["_score"] = sim
        hits.append((sim, r))
    return hits

def simple_rerank(hits: List[Tuple[float, Dict]], prefer_jurisdiction="AU") -> List[Tuple[float, Dict]]:
    def key(item):
        sim, r = item
        bonus = 0.0
        if r.get("jurisdiction") == prefer_jurisdiction: bonus += 0.02
        if len(r.get("text","")) < 1200: bonus += 0.01
        return sim + bonus
    return sorted(hits, key=key, reverse=True)

def format_context(hits: List[Tuple[float, Dict]], limit_chars=MAX_CONTEXT_CHARS, context_k=4) -> Tuple[str, List[Dict]]:
    used = []
    total = 0
    blocks = []
    for sim, r in hits[:context_k]:
        block = (
            f"[{r['chunk_id']}] title={r.get('title')} | role={r.get('role')}/{r.get('category')} | "
            f"url={r.get('source_url')} | jurisdiction={r.get('jurisdiction')} | score={sim:.3f}\n"
            + (r.get("text","").strip())
        )
        if total + len(block) > limit_chars:
            break
        used.append(r)
        blocks.append(block)
        total += len(block)
    ctx = "\n\n---\n\n".join(blocks)
    return ctx, used

# ---------------- LLM call ----------------
def call_llm_answer(model_name: str, question: str, context: str) -> str:
    payload = {
        "model": model_name,
        "messages": [
            {"role":"system","content": SYSTEM_INSTRUCTIONS},
            {"role":"user","content": USER_TEMPLATE.format(question=question, context=context)}
        ],
        "options": {"temperature": 0.2, "num_ctx": 8192}
    }
    resp = ollama.chat(**payload)
    return resp["message"]["content"].strip()

def extract_citations(answer: str) -> List[str]:
    return sorted(set(CIT_RE.findall(answer)))

def is_unanswered(ans: str) -> bool:
    low = ans.lower()
    return any(p.lower() in low for p in UNANSWERED_PHRASES)

# ---------------- LLM-as-judge ----------------
JUDGE_RUBRIC = """You are a strict evaluator. Given a QUESTION, CONTEXT (the only allowed source of truth), and an ANSWER:

Score two dimensions from 1 (poor) to 5 (excellent), integers only:
- FAITHFULNESS: Are the answer's claims supported by the CONTEXT? Penalize any unsupported claim, policy, date, number, or jurisdictional detail.
- COMPLETENESS: Does the answer cover the key points needed to address the QUESTION, given the CONTEXT? Penalize omissions.

Rules:
- Use ONLY the CONTEXT as ground truth. If the answer uses info not in CONTEXT, reduce FAITHFULNESS.
- If the answer honestly says it doesn’t know and points to relevant articles, FAITHFULNESS may be high (4–5), COMPLETENESS medium (2–3) depending on helpfulness.
- Return a single JSON object with keys: faithfulness, completeness, critique.
"""
JUDGE_PROMPT = """QUESTION:
{q}

CONTEXT:
{ctx}

ANSWER:
{ans}

Return JSON only, like:
{{"faithfulness": 4, "completeness": 3, "critique": "short reason"}}
"""
JSON_GRAB = re.compile(r'\{.*\}', re.S)

def call_judge(model: str, question: str, context: str, answer: str) -> Dict:
    payload = {
        "model": model,
        "messages": [
            {"role":"system", "content": JUDGE_RUBRIC},
            {"role":"user", "content": JUDGE_PROMPT.format(q=question, ctx=context, ans=answer)}
        ],
        "options": {"temperature": 0.0, "num_ctx": 8192}
    }
    resp = ollama.chat(**payload)
    txt = resp["message"]["content"].strip()
    m = JSON_GRAB.search(txt)
    if not m:
        return {"faithfulness": None, "completeness": None, "critique": "No JSON from judge."}
    try:
        data = json.loads(m.group(0))
        f = int(data.get("faithfulness")) if data.get("faithfulness") is not None else None
        c = int(data.get("completeness")) if data.get("completeness") is not None else None
        crit = str(data.get("critique") or "")
        return {"faithfulness": f, "completeness": c, "critique": crit[:500]}
    except Exception:
        return {"faithfulness": None, "completeness": None, "critique": "Bad JSON from judge."}

# ---------------- Metrics ----------------
def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / len(A | B)

def evaluate_example(st: SentenceTransformer,
                     meta: Dict,
                     index,
                     rows,
                     ex: Dict,
                     model_name: str,
                     top_k: int,
                     context_k: int,
                     judge_model: Optional[str]) -> Dict:
    # Retrieval
    pre_hits = search(index, rows, st, meta, ex["question"], k=top_k)
    hits = simple_rerank(pre_hits)
    ctx_str, used = format_context(hits, context_k=context_k)

    # LLM answer
    t0 = time.time()
    ans = call_llm_answer(model_name, ex["question"], ctx_str)
    latency = time.time() - t0

    # Basic metrics
    ua = is_unanswered(ans)
    cits = extract_citations(ans)

    # Semantic accuracy vs gold answer
    acc_sim, acc_hit = None, None
    if ex.get("answer_gold"):
        vg = embed_passage(st, ex["answer_gold"], passage_instruction=None)
        vp = embed_passage(st, ans, passage_instruction=None)
        acc_sim = float(np.dot(vg, vp.T).reshape(-1)[0])
        acc_hit = (acc_sim >= ACC_SIM_THRESHOLD)

    # Recall@k: any gold chunk in pre-retrieved set
    recall_hit = None
    if ex.get("gold_chunks"):
        retrieved_ids = [r["chunk_id"] for _, r in pre_hits]
        recall_hit = any(g in retrieved_ids for g in ex["gold_chunks"])

    # Attribution overlap
    attr_overlap, attr_any = None, None
    if ex.get("gold_chunks"):
        attr_overlap = jaccard(cits, ex["gold_chunks"])
        attr_any = (len(set(cits) & set(ex["gold_chunks"])) > 0) if cits else False

    # Hallucination proxy: answered but no overlap with any gold chunk
    hall = False
    if not ua and ex.get("gold_chunks"):
        hall = (attr_any is False)

    # Judge (optional)
    faith, comp, critique = None, None, None
    if judge_model:
        j = call_judge(judge_model, ex["question"], ctx_str, ans)
        faith, comp, critique = j.get("faithfulness"), j.get("completeness"), j.get("critique")

    return {
        "id": ex.get("id"),
        "role": ex.get("role"),
        "intent": ex.get("intent"),
        "model": model_name,
        "unanswered": ua,
        "acc_sim": acc_sim,
        f"acc_hit@{ACC_SIM_THRESHOLD:.2f}": acc_hit,
        f"recall@{top_k}": recall_hit,
        "attr_overlap_jaccard": attr_overlap,
        "attr_any_match": attr_any,
        "hallucination_proxy": hall,
        "latency_s": latency,
        "answer": ans,
        "citations": cits,
        "used_chunk_ids": [r["chunk_id"] for r in used],
        "judge_faithfulness_1to5": faith,
        "judge_completeness_1to5": comp,
        "judge_critique": critique
    }

# ---------------- Aggregation ----------------
def aggregate(df: pd.DataFrame, top_k: int) -> pd.Series:
    out = {}
    out["n"] = len(df)
    if "unanswered" in df:
        out["UA%"] = 100 * df["unanswered"].mean()
    acc_col = f"acc_hit@{ACC_SIM_THRESHOLD:.2f}"
    if acc_col in df:
        out["Acc@Gold%"] = 100 * df[acc_col].fillna(False).mean()
    rec_col = f"recall@{top_k}"
    if rec_col in df:
        out[f"Recall@{top_k}%"] = 100 * df[rec_col].fillna(False).mean()
    if "attr_any_match" in df:
        out["AttributionAny%"] = 100 * df["attr_any_match"].fillna(False).mean()
    if "attr_overlap_jaccard" in df and df["attr_overlap_jaccard"].notna().any():
        out["AttrOverlap(Jaccard)"] = df["attr_overlap_jaccard"].dropna().mean()
    if "hallucination_proxy" in df:
        out["Hallucination%"] = 100 * df["hallucination_proxy"].fillna(False).mean()
    if "judge_faithfulness_1to5" in df and df["judge_faithfulness_1to5"].notna().any():
        out["Faithfulness(mean)"] = df["judge_faithfulness_1to5"].dropna().mean()
    if "judge_completeness_1to5" in df and df["judge_completeness_1to5"].notna().any():
        out["Completeness(mean)"] = df["judge_completeness_1to5"].dropna().mean()
    if "latency_s" in df:
        out["Latency_p50_s"] = df["latency_s"].median()
        out["Latency_p95_s"] = df["latency_s"].quantile(0.95)
    return pd.Series(out)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=DEFAULT_GOLD, help="Path to questions.jsonl")
    ap.add_argument("--models", nargs="+", required=True, help="Ollama model names to compare")
    ap.add_argument("--outprefix", default="eval_run", help="Output run name prefix")
    ap.add_argument("--top_k", type=int, default=8, help="Retrieval top-k")
    ap.add_argument("--context_k", type=int, default=4, help="Context chunks to feed the LLM")
    ap.add_argument("--judge_model", default=None, help="Optional judge model (e.g., mistral:7b)")
    args = ap.parse_args()

    # Check artifacts
    if not os.path.exists(FAISS_PATH):
        print("[ERROR] Missing FAISS index. Build Step 2 first.")
        sys.exit(1)
    meta = load_meta()
    rows = load_chunk_meta()
    index = faiss.read_index(FAISS_PATH)
    gold = load_gold(args.gold)
    if not gold:
        print("[ERROR] Empty gold set.")
        sys.exit(1)

    # Embedder for query/scoring (use the same one as the index was built with)
    st = get_embedder(meta["model"])

    all_rows = []
    for model_name in args.models:
        print(f"\n=== Evaluating model: {model_name} ===")
        for ex in gold:
            row = evaluate_example(
                st, meta, index, rows, ex, model_name,
                top_k=args.top_k,
                context_k=args.context_k,
                judge_model=args.judge_model
            )
            all_rows.append(row)
            # brief progress line
            acc_col = f"acc_hit@{ACC_SIM_THRESHOLD:.2f}"
            print(f"- {ex.get('id')} | UA={row['unanswered']} | acc_hit={row.get(acc_col)} | "
                  f"recall={row.get(f'recall@{args.top_k}')} | t={row['latency_s']:.2f}s")

    df = pd.DataFrame(all_rows)

    # Aggregations
    summary_model = df.groupby("model").apply(lambda d: aggregate(d, args.top_k)).reset_index()
    summary_breakdown = df.groupby(["model","role","intent"]).apply(lambda d: aggregate(d, args.top_k)).reset_index()

    # Save outputs
    run_ts = int(time.time())
    base = os.path.join(OUTDIR, f"{args.outprefix}_{run_ts}")
    os.makedirs(base, exist_ok=True)

    df.to_json(os.path.join(base, "per_question.json"), orient="records", lines=True, force_ascii=False)
    summary_model.to_csv(os.path.join(base, "summary_by_model.csv"), index=False)
    summary_breakdown.to_csv(os.path.join(base, "summary_by_model_role_intent.csv"), index=False)

    print("\n=== Summary by model ===")
    print(summary_model.to_string(index=False))
    print(f"\nSaved:\n- {base}/per_question.json\n- {base}/summary_by_model.csv\n- {base}/summary_by_model_role_intent.csv")

if __name__ == "__main__":
    main()
