import os, json, re, argparse, random
from typing import List, Dict, Tuple
import frontmatter

ROOT = os.path.dirname(os.path.dirname(__file__))
CLEAN_DIR = os.path.join(ROOT, "data", "clean")
CHUNKS_PATH = os.path.join(ROOT, "data", "chunks", "kb_chunks.jsonl")
EVAL_DIR = os.path.join(ROOT, "data", "eval")
OUT_PATH = os.path.join(EVAL_DIR, "questions.jsonl")

# Categories -> question templates
CATEGORY_TEMPLATES = {
    "refunds_charges": "How do refunds/charges work according to “{title}” in {jurisdiction}?",
    "order_change_cancel": "How can a user change or cancel an order per “{title}” in {jurisdiction}?",
    "order_status": "What does “{title}” say about order status in {jurisdiction}?",
    "promos_fees": "What fees or promos are described in “{title}” for {jurisdiction}?",
    "delivery_window": "What are the delivery timing rules in “{title}” ({jurisdiction})?",
    "courier_pay_flow": "What are the components of courier pay per “{title}” in {jurisdiction}?",
    "courier_policies": "What policy applies to couriers in “{title}” ({jurisdiction})?",
    "merchant_onboarding": "How do merchants onboard according to “{title}” in {jurisdiction}?",
    "merchant_payouts": "How do merchant payouts work per “{title}” ({jurisdiction})?",
    "safety_privacy": "What safety or privacy rule is specified in “{title}” ({jurisdiction})?",
    # default fallback:
    "_default": "What is the key rule from “{title}” in {jurisdiction}?"
}

SPLIT_SENT = re.compile(r'(?<=[.!?])\s+')
BAD_STARTS = tuple(["cookie", "©", "© ", "was this helpful", "terms", "privacy", "contact", "support", "learn more"])
MIN_SENT_LEN = 60
MAX_SENT_LEN = 220

KEYWORD_BONUS = [
    "refund", "charge", "fee", "cancel", "cancellation", "delivery", "time", "window",
    "tip", "distance", "base", "fare", "payout", "onboard", "bank", "policy", "must",
    "must not", "cannot", "eligible", "eligibility", "within", "after", "before", "AU",
    "Australia", "support", "evidence", "photo", "order", "merchant", "courier"
]

def load_markdown_docs() -> List[Tuple[Dict, str, str]]:
    """
    Returns list of (meta, content, filename)
    """
    docs = []
    for fn in sorted(os.listdir(CLEAN_DIR)):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(CLEAN_DIR, fn)
        post = frontmatter.load(path)
        meta = post.metadata or {}
        body = (post.content or "").strip()
        # basic required fields; skip if missing
        if not meta.get("id") or not meta.get("title"):
            continue
        docs.append((meta, body, fn))
    return docs

def load_chunks() -> Dict[str, List[Dict]]:
    """
    Map doc_id -> list of {chunk_id, text, ...}
    """
    mapping = {}
    if not os.path.exists(CHUNKS_PATH):
        print(f"[WARN] Missing {CHUNKS_PATH}. You should run Step 1 chunker first.")
        return mapping
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            mapping.setdefault(r["doc_id"], []).append(r)
    return mapping

def pick_informative_sentence(text: str) -> str:
    """
    Choose one sentence that is likely policy-like: not too short/long, contains useful keywords.
    """
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    sents = SPLIT_SENT.split(text)
    best = None
    best_score = -1
    for s in sents:
        s_strip = s.strip()
        if not s_strip:
            continue
        low = s_strip.lower()
        if low.startswith(BAD_STARTS):
            continue
        if len(s_strip) < MIN_SENT_LEN or len(s_strip) > MAX_SENT_LEN:
            continue
        # simple scoring: length and keyword hits
        score = 0
        score += min(len(s_strip) / 80.0, 3)  # prefer medium length
        hits = sum(1 for kw in KEYWORD_BONUS if kw in low)
        score += hits * 1.2
        # sentences with “must/should/can” are often policy statements
        if " must " in low or " should " in low or " can " in low or " cannot " in low:
            score += 1.0
        if score > best_score:
            best_score = score
            best = s_strip
    # fallback: first paragraph if nothing matched
    if not best:
        best = text[:200].strip()
    return best

def make_question(meta: Dict) -> str:
    cat = meta.get("category") or "_default"
    tpl = CATEGORY_TEMPLATES.get(cat, CATEGORY_TEMPLATES["_default"])
    return tpl.format(
        title=meta.get("title", "this article"),
        jurisdiction=meta.get("jurisdiction", "AU")
    )

def find_supporting_chunk(doc_id: str, target_sentence: str, chunks_map: Dict[str, List[Dict]]) -> List[str]:
    """
    Try to locate which chunk(s) contain the chosen answer sentence.
    Return [chunk_id] or fallback to first chunk of the doc.
    """
    chunks = chunks_map.get(doc_id, [])
    target_low = target_sentence.lower().strip()
    # allow a fuzzy-ish containment by removing extra spaces
    target_comp = re.sub(r'\s+', ' ', target_low)
    for ch in chunks:
        text_low = re.sub(r'\s+', ' ', ch.get("text", "").lower())
        if target_comp[:60] in text_low or target_comp in text_low:
            return [ch["chunk_id"]]
    # fallback: first chunk if present
    if chunks:
        return [chunks[0]["chunk_id"]]
    return []

def bootstrap(limit: int = 50, shuffle: bool = True) -> List[Dict]:
    docs = load_markdown_docs()
    chunks_map = load_chunks()
    if shuffle:
        random.shuffle(docs)

    examples = []
    for meta, body, fn in docs:
        # Skip extremely short bodies
        if len(body) < 120:
            continue
        answer_sentence = pick_informative_sentence(body)
        q = make_question(meta)
        gold_chunks = find_supporting_chunk(meta["id"], answer_sentence, chunks_map)
        ex = {
            "id": meta["id"] + "::auto",
            "question": q,
            "answer_gold": answer_sentence,
            "gold_chunks": gold_chunks,
            "role": meta.get("role"),
            "intent": meta.get("category")
        }
        examples.append(ex)
        if len(examples) >= limit:
            break
    return examples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="max number of questions to generate")
    ap.add_argument("--out", default=OUT_PATH, help="output JSONL path")
    args = ap.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)
    examples = bootstrap(limit=args.limit, shuffle=True)
    if not examples:
        print("[ERROR] No examples generated. Check that data/clean/*.md and data/chunks/kb_chunks.jsonl exist.")
        return

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(examples)} examples to {args.out}")
    print("   Sample:")
    print(json.dumps(examples[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
