
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata, re

def _norm(s:str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s).encode("ascii","ignore").decode("utf-8")
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s

def load_kb(csv_path: str):
    if not os.path.isfile(csv_path):
        return pd.DataFrame(columns=["pregunta","respuesta","link"])
    df = pd.read_csv(csv_path)
    for col in ["pregunta","respuesta","link"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    return df

def build_kb_index(df: "pd.DataFrame"):
    if df.empty:
        return {"vectorizer": None, "matrix": None, "df": df}
    corpus = [_norm(f"{r.pregunta} || {r.respuesta}") for r in df.itertuples()]
    vect = TfidfVectorizer(strip_accents="unicode", ngram_range=(1,2), max_df=0.9, min_df=1)
    X = vect.fit_transform(corpus)
    return {"vectorizer": vect, "matrix": X, "df": df}

def query_kb(q: str, kb_index, top_n: int = 3):
    if not kb_index or kb_index["vectorizer"] is None:
        return []
    vec = kb_index["vectorizer"].transform([_norm(q)])
    sims = cosine_similarity(vec, kb_index["matrix"]).ravel()
    order = sims.argsort()[::-1][:top_n]
    rows = []
    for i in order:
        row = kb_index["df"].iloc[i]
        rows.append({"pregunta": row.get("pregunta",""), "respuesta": row.get("respuesta",""), "link": row.get("link",""), "score": float(sims[i])})
    return rows
