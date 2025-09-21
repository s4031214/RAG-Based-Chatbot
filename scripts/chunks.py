import os, json, re, frontmatter
from datetime import datetime
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
CLEAN_DIR = os.path.join(ROOT, "data", "clean")
CHUNK_DIR = os.path.join(ROOT, "data", "chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

CHUNK_SIZE = 900    # characters (tune: 600–1200)
OVERLAP = 120       # characters

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n) 
        chunk = text[i:j]
        # try to avoid breaking mid-sentence
        if j < n:
            k = chunk.rfind(". ")
            if k > size * 0.6:
                j = i + k + 1
                chunk = text[i:j]
        yield chunk.strip()
        i = max(j - overlap, j)

def main():
    out_path = os.path.join(CHUNK_DIR, "kb_chunks.jsonl")
    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for fn in tqdm(os.listdir(CLEAN_DIR), desc="Chunking"):
            if not fn.endswith(".md"):
                continue
            post = frontmatter.load(os.path.join(CLEAN_DIR, fn))
            meta = post.metadata
            body = post.content
            # strip very short boilerplate
            body = re.sub(r"\n{3,}", "\n\n", body).strip()
            for idx, ch in enumerate(chunk_text(body)):
                rec = {
                    "doc_id": meta["id"],
                    "chunk_id": f"{meta['id']}--{idx}",
                    "title": meta["title"],
                    "text": ch,
                    "source_url": meta.get("source_url"),
                    "published_at": meta.get("published_at"),
                    "jurisdiction": meta.get("jurisdiction"),
                    "role": meta.get("role"),
                    "category": meta.get("category"),
                    "version": meta.get("version"),
                    "retrieved_at": meta.get("retrieved_at")
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    print(f"Wrote {count} chunks to {out_path}")

if __name__ == "__main__":
    main()
