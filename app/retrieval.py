
# app/retrieval.py
import os, re, unicodedata
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import (
    DOCS_DIR, KB_CSV, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, MAX_WORDS, PUBLIC_DOC_BASE_URL
)
from app.document_reader import load_text_from_path

# ---------------------- utils ----------------------
def _norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s).encode("ascii","ignore").decode("utf-8")
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s

def _chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    s = re.sub(r"\s+", " ", s).strip()
    out, i = [], 0
    while i < len(s):
        out.append(s[i:i+size])
        i += max(1, size - overlap)
    return out

def _cap_words(text: str, maxw: int = MAX_WORDS) -> str:
    w = (text or "").split()
    return " ".join(w[:maxw]) + ("..." if len(w) > maxw else "")

# ---------------------- KB ----------------------
def _load_kb_df() -> pd.DataFrame:
    if not KB_CSV or not os.path.isfile(KB_CSV):
        return pd.DataFrame(columns=["pregunta","respuesta","link"])
    df = pd.read_csv(KB_CSV)
    for col in ["pregunta","respuesta","link"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    return df

def _build_kb_index(df: pd.DataFrame):
    if df.empty:
        return {"vectorizer": None, "X": None, "df": df}
    corpus = [_norm(f"{r.pregunta} || {r.respuesta}") for r in df.itertuples()]
    vect = TfidfVectorizer(strip_accents="unicode", ngram_range=(1,2),
                           max_df=0.9, min_df=1, token_pattern=r"(?u)\b\w+\b")
    try:
        X = vect.fit_transform(corpus)
    except ValueError:
        return {"vectorizer": None, "X": None, "df": df}
    return {"vectorizer": vect, "X": X, "df": df}

def _query_kb(query: str, kb_index, top_n: int = 1):
    if not kb_index or kb_index["vectorizer"] is None:
        return None
    v = kb_index["vectorizer"].transform([_norm(query)])
    sims = cosine_similarity(v, kb_index["X"]).ravel()
    i = sims.argsort()[::-1][:top_n][0]
    row = kb_index["df"].iloc[i]
    return {
        "pregunta": row.get("pregunta", ""),
        "respuesta": row.get("respuesta", ""),
        "link": row.get("link", "")
    }

# ---------------------- DOCS ----------------------
def _infer_doc_paths(root=DOCS_DIR) -> List[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for f in os.listdir(root):
        if f.lower().endswith((".pdf",".docx",".doc",".txt")):
            out.append(os.path.join(root, f))
    return sorted(out)

def _build_doc_index(paths: List[str]):
    chunks, metas = [], []
    for p in paths:
        txt = load_text_from_path(p)
        if not txt or not txt.strip():
            print(f"[INFO] Sin texto útil: {p}")
            continue
        for k, ch in enumerate(_chunk_text(txt)):
            chunks.append(_norm(ch))
            metas.append({"name": os.path.basename(p), "path": p, "chunk_id": k})

    if not chunks:
        # índice vacío pero válido
        return {"vectorizer": None, "X": None, "chunks": [], "metas": []}

    vect = TfidfVectorizer(strip_accents="unicode", ngram_range=(1,2),
                           max_df=0.9, min_df=1, token_pattern=r"(?u)\b\w+\b")
    try:
        X = vect.fit_transform(chunks)
    except ValueError:
        return {"vectorizer": None, "X": None, "chunks": [], "metas": []}
    return {"vectorizer": vect, "X": X, "chunks": chunks, "metas": metas}

def _retrieve_docs(query: str, doc_index, k: int = TOP_K) -> Tuple[str, List[Dict[str,Any]]]:
    if (not doc_index) or (not doc_index.get("chunks")) or (doc_index.get("vectorizer") is None):
        return "", []
    v = doc_index["vectorizer"].transform([_norm(query)])
    sims = cosine_similarity(v, doc_index["X"]).ravel()
    order = sims.argsort()[::-1][:k]

    joined = " ".join([doc_index["chunks"][i] for i in order[:2]])  # resumen con 2 top
    sources, seen = [], set()
    for i in order:
        m = doc_index["metas"][i]
        name = m["name"]
        if name in seen: 
            continue
        seen.add(name)
        url = f"{PUBLIC_DOC_BASE_URL}/{name}" if PUBLIC_DOC_BASE_URL else m["path"]
        sources.append({"name": name, "path": url})
    return joined, sources

# ---------------------- API pública del módulo ----------------------
_DOC_INDEX = None
_KB_INDEX  = None

def init_indexes():
    """Construye índices (tolerante a vacíos) y también los deja en variables globales.
       Devuelve (_DOC_INDEX, _KB_INDEX) para quien quiera usarlos explícitamente."""
    global _DOC_INDEX, _KB_INDEX
    doc_paths = _infer_doc_paths()
    _DOC_INDEX = _build_doc_index(doc_paths)

    kb_df = _load_kb_df()
    _KB_INDEX = _build_kb_index(kb_df)

    return _DOC_INDEX, _KB_INDEX

def answer_with_sources(query: str,
                        DOC_INDEX: Optional[dict]=None,
                        KB_INDEX: Optional[dict]=None,
                        max_words: int = MAX_WORDS) -> Tuple[str, List[Dict[str,Any]]]:
    """Puede usarse de dos formas:
       - answer_with_sources(q, DOC_INDEX, KB_INDEX)
       - answer_with_sources(q)  # usa los índices globales creados por init_indexes()"""
    d_index = DOC_INDEX if DOC_INDEX is not None else _DOC_INDEX
    k_index = KB_INDEX  if KB_INDEX  is not None else _KB_INDEX

    kb_best = _query_kb(query, k_index)
    doc_text, doc_sources = _retrieve_docs(query, d_index)

    kb_block = kb_best["respuesta"] if kb_best else ""
    combined = (kb_block + ("\n\nResumen documental: " + doc_text if doc_text else "")).strip()
    final = _cap_words(combined, max_words)

    sources = []
    if kb_best and kb_best.get("link"):
        sources.append({"name": "KB", "path": kb_best["link"]})
    sources.extend(doc_sources)

    if not final.strip():  # corpus vacío
        return ("Por el momento no cuento con la información necesaria para responderte, ¿tienes alguna otra duda?"
                " si no, pregúnta algo sobre mi"), []
    return final, sources
