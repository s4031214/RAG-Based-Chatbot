import os, frontmatter, sys
REQ = {"id","title","source_url","jurisdiction","role","category","version","retrieved_at"}
ROOT = os.path.dirname(os.path.dirname(__file__))
CLEAN = os.path.join(ROOT,"data","clean")
ok = True
for fn in sorted(os.listdir(CLEAN)):
    if not fn.endswith(".md"): continue
    post = frontmatter.load(os.path.join(CLEAN, fn))
    missing = REQ - set(post.metadata.keys())
    if missing:
        ok = False
        print(f"[MISSING] {fn}: {sorted(missing)}")
if ok:
    print("✅ All front matter keys present.")
 