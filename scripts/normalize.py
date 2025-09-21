import os, json, re, hashlib
from datetime import datetime
from requests import post
from slugify import slugify
import frontmatter
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pdfminer.high_level import extract_text

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CLEAN_DIR = os.path.join(ROOT, "data", "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)

def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()

def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove nav/footer/script/style
    for tag in soup(["script","style","nav","footer","noscript"]):
        tag.decompose()
    # Remove irrelevant headers like "Was this helpful?"
    for tag in soup.find_all(lambda t: t.name in ["aside","form"]): 
        tag.decompose()
    return md(str(soup), heading_style="ATX", strip=["img"])

def guess_title(md_text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", md_text, flags=re.MULTILINE)
    return m.group(1).strip() if m else fallback

def normalize_one(raw_path: str):
    meta_path = raw_path + ".json"
    if not os.path.exists(meta_path): 
        return
    with open(meta_path, "r") as f:
        m = json.load(f)

    ext = os.path.splitext(raw_path)[1].lower()
    text_md = ""
    if ext == ".html":
        with open(raw_path, "rb") as f:
            html = f.read().decode("utf-8", "ignore")
        text_md = html_to_markdown(html)
    elif ext == ".pdf":
        text = extract_text(raw_path) or ""
        # Convert simple PDF paragraphs to Markdown paragraphs
        text_md = re.sub(r"\n{2,}", "\n\n", text.strip())
    else:
        # unknown: treat as text
        with open(raw_path, "rb") as f:
            text_md = f.read().decode("utf-8", "ignore")

    text_md = text_md.strip()
    if not text_md:
        return

    title = m.get("title") or guess_title(text_md, "Untitled")
    doc_id = slugify(f"{m.get('role','gen')}-{m.get('category','general')}-{title}")[:80]

    post = frontmatter.Post(text_md)
    post.metadata = {
        "id": doc_id,
        "title": title,
        "source_url": m.get("url"),
        "published_at": None,          # fill manually if known
        "jurisdiction": m.get("jurisdiction", "AU"),
        "role": m.get("role","general"),
        "category": m.get("category","general"),
        "version": "1.0",
        "retrieved_at": m.get("fetched_at"),
        "license": "© Owner; educational use; no redistribution",
        "checksum": sha256_text(text_md)
    }

    out_path = os.path.join(CLEAN_DIR, f"{doc_id}.md")
    fm = frontmatter.dumps(post)   # returns a str
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(fm)
    return out_path

def main():
    for fn in os.listdir(RAW_DIR):
        if fn.endswith(".json"): 
            continue
        normalize_one(os.path.join(RAW_DIR, fn))

if __name__ == "__main__":
    main()
