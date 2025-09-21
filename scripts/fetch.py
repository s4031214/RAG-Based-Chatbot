import hashlib, os, time, re, json, sys
from urllib.parse import urlparse
import requests, yaml
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CFG = os.path.join(ROOT, "config", "urls.yaml")

os.makedirs(RAW_DIR, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def allowed_by_robots(base, path, ua):
    # Minimal robots.txt check 
    try:
        rob = requests.get(f"{base}/robots.txt", timeout=10)
        if rob.status_code != 200:
            return True
        disallows = []
        user_block = False
        for line in rob.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("user-agent:"):
                agent = line.split(":",1)[1].strip()
                user_block = (agent == "*" or agent.lower() in ua.lower())
            elif user_block and line.lower().startswith("disallow:"):
                rule = line.split(":",1)[1].strip()
                disallows.append(rule)
        for d in disallows:
            if path.startswith(d):
                return False
        return True
    except Exception:
        return True  # be conservative but not blocking educational fetch

def sanitize_filename(url):
    safe = re.sub(r'[^a-zA-Z0-9_.-]+','-', url)
    return safe[:180]

def main():
    with open(CFG, "r") as f:
        cfg = yaml.safe_load(f)
    urls = cfg["urls"]
    rate = cfg.get("rate_limit_seconds", 2)
    ua = cfg.get("user_agent", "RMIT-RAG-StudyBot/1.0")

    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept": "*/*"})

    for item in tqdm(urls, desc="Fetching"):
        url = item["url"]
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if not allowed_by_robots(base, parsed.path, ua):
            print(f"[SKIP robots] {url}")
            continue
        try:
            r = s.get(url, timeout=20)
            if r.status_code != 200:
                print(f"[WARN] {url} -> {r.status_code}")
                continue
            content = r.content
            ext = ".html"
            if parsed.path.lower().endswith(".pdf") or r.headers.get("Content-Type","").lower().startswith("application/pdf"):
                ext = ".pdf"
            name = sanitize_filename(url) + ext
            path = os.path.join(RAW_DIR, name)
            with open(path, "wb") as f:
                f.write(content)
            meta = {
                "url": url,
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "status": r.status_code,
                "headers": dict(r.headers),
                "sha256": sha256_bytes(content),
                **{k:v for k,v in item.items() if k not in ["url"]}
            }
            with open(path + ".json", "w") as f:
                json.dump(meta, f, indent=2)
            time.sleep(rate)
        except Exception as e:
            print(f"[ERROR] {url}: {e}")

if __name__ == "__main__":
    main()
