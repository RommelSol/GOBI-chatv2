
import os, re, unicodedata
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .document_reader import load_text_from_path
from .config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MAX_WORDS, PUBLIC_DOC_BASE_URL

class Chunk:
    def __init__(self, text: str, source: str, meta: Dict[str, Any]):
        self.text = text
        self.source = source
        self.meta = meta

def _norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s).encode("ascii","ignore").decode("utf-8")
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s

def _chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    chunks, i = [], 0
    while i < len(s):
        chunks.append(s[i:i+size])
        i += (size - overlap)
    return chunks

def infer_paths_from_dir(root: str = DOCS_DIR):
    if not os.path.isdir(root):
        return []
    out = []
    for f in os.listdir(root):
        if f.lower().endswith((".pdf", ".txt")):
            out.append(os.path.join(root, f))
    return sorted(out)

def build_index(paths: List[str]):
    chunks = []
    for p in paths:
        try:
            txt = load_text_from_path(p)
        except Exception as e:
            print(f"[WARN] No pude leer {p}: {e}")
            continue
        if not txt.strip():
            print(f"[INFO] Sin texto util: {p}")
            continue
        for idx, c in enumerate(_chunk_text(txt)):
            chunks.append(Chunk(text=_norm(c), source=p, meta={"chunk_id": idx}))

    corpus = [c.text for c in chunks]
    if not corpus:
        vec = TfidfVectorizer().fit([""])
        return {"vectorizer": vec, "matrix": None, "chunks": []}

    vectorizer = TfidfVectorizer(strip_accents="unicode", ngram_range=(1,2), max_df=0.9, min_df=1)
    X = vectorizer.fit_transform(corpus)
    return {"vectorizer": vectorizer, "matrix": X, "chunks": chunks}

def retrieve(query: str, index, k: int = TOP_K):
    if not index or not index.get("chunks"):
        return []
    vec = index["vectorizer"].transform([_norm(query)])
    sims = cosine_similarity(vec, index["matrix"]).ravel()
    top_idx = sims.argsort()[::-1][:k]
    return [index["chunks"][i] for i in top_idx]

def compose_answer(query: str, hits: List[Chunk]):
    if not hits:
        return {"answer": "", "sources": []}
    combined = " ".join([h.text for h in hits]).strip()
    words = combined.split()
    short = " ".join(words[:MAX_WORDS]) + ("..." if len(words) > MAX_WORDS else "")

    sources = []
    seen = set()
    for h in hits:
        name = os.path.basename(h.source)
        if name in seen:
            continue
        seen.add(name)
        if PUBLIC_DOC_BASE_URL:
            url = f"{PUBLIC_DOC_BASE_URL}/{name}"
        else:
            url = h.source
        sources.append({"name": name, "path": url})
    return {"answer": short, "sources": sources}
