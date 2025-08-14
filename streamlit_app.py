
import os
import streamlit as st

from app.config import KB_CSV, PUBLIC_DOC_BASE_URL
from app.knowledge import load_kb, build_kb_index, query_kb
from app.gobi_core import build_index, infer_paths_from_dir, retrieve, compose_answer
from app.emotion import detect_emotion, empathetic_prefix

st.set_page_config(page_title="GOBI", page_icon="🤖", layout="wide")
st.title("🤖 GOBI – Chatbot documental (Streamlit)")

st.sidebar.header("📚 Documentos")
st.sidebar.write("Sube PDFs con texto digital o archivos .txt. Para PDFs escaneados, usa .txt preextraído.")
uploaded = st.sidebar.file_uploader("Subir archivos", type=["pdf","docx","doc","txt"], accept_multiple_files=True)

if uploaded:
    os.makedirs("data/docs", exist_ok=True)
    for up in uploaded:
        path = os.path.join("data/docs", up.name)
        with open(path, "wb") as f:
            f.write(up.read())
    st.sidebar.success("Archivos guardados en data/docs. Recarga si no aparecen.")

# ------- cached indices -------
@st.cache_resource(show_spinner=True)
def _build_doc_index():
    return build_index(infer_paths_from_dir())

@st.cache_resource(show_spinner=True)
def _build_kb_index():
    df = load_kb(KB_CSV)
    return build_kb_index(df)

doc_index = _build_doc_index()
kb_index = _build_kb_index()

if "history" not in st.session_state:
    st.session_state["history"] = []
    st.chat_message("assistant").write("¡Hola! Soy GOBI. Puedo usar tu KB (CSV) y tus documentos (PDF/TXT). ¿En qué te ayudo?")

with st.form("chat"):
    q = st.text_input("Escribe tu consulta:")
    submitted = st.form_submit_button("Enviar")

if submitted and q.strip():
    emo = detect_emotion(q)
    kb_hits = query_kb(q, kb_index, top_n=3)
    # Recuperación documental
    hits = retrieve(q, doc_index)
    doc_result = compose_answer(q, hits) if hits else {"answer":"", "sources":[]}

    # Ensamble + límite (se hace dentro de compose_answer para lo documental; KB se concatena)
    kb_block = kb_hits[0]["respuesta"] if kb_hits else ""
    combined = (kb_block + ("\n\nResumen documental: " + doc_result["answer"] if doc_result["answer"] else "")).strip()
    prefix = empathetic_prefix(emo)
    final = prefix + combined if combined else prefix + "No encontré evidencia suficiente en la documentación disponible."

    # guardar historial
    st.session_state["history"].append(("Tú", q))
    st.session_state["history"].append(("Gobi", final))

    st.markdown("**Fuentes:**")
    sources = []
    if kb_hits and kb_hits[0].get("link"):
        sources.append({"name": "KB", "path": kb_hits[0]["link"]})
    if doc_result.get("sources"):
        sources.extend(doc_result["sources"])

    if sources:
        for s in sources:
            if PUBLIC_DOC_BASE_URL and s['name'] != "KB":
                # ya vienen con URL absoluta; garantizamos que apunten a RAW si procede
                pass
            st.markdown(f"- [{s['name']}]({s['path']})")
    else:
        st.write("—")

for who, msg in st.session_state["history"]:
    st.chat_message("user" if who == "Tú" else "assistant").write(msg)

st.caption("Modo demo: si no subes archivos ni CSV, GOBI responderá con información limitada.")
