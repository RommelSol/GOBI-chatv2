
import streamlit as st
import re
import random
from time import time

from app.config import MAX_WORDS
from app.retrieval import init_indexes, answer_with_sources
from app.smalltalk import smalltalk_reply
from app.emotion_ml import detect_emotion, empathetic_prefix
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="GOBI · Chatbot Documental", page_icon="🤖", layout="wide")

# ============================
# ===== UI: basic styles =====
# ============================
st.markdown(
    """
<style>
.chat-container {max-width: 850px; margin: 0 auto;}
.msg {padding: 10px 14px; border-radius: 16px; margin: 8px 0; display: inline-block; max-width: 90%;}
.user {background: #DCF8C6; align-self: flex-end;}
.bot  {background: #F1F0F0;}
.bubble {display:flex;}
.bubble.user {justify-content: flex-end;}
.sources {font-size: 0.9rem; color:#555; margin-top: 8px;}
.title {text-align: center; margin-bottom: 0;}
.subtitle {text-align:center; color:#666; margin-top: 0;}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("<h1 class='title'>🤖 GOBI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Chatbot documental · Respuestas hasta {mw} palabras + enlaces a fuentes</p>".format(
        mw=MAX_WORDS
    ),
    unsafe_allow_html=True,
)

# ==================================
# ===== Helpers (unificados) =======
# ==================================

# --- Split de oraciones ---
_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+")

def split_sentences(text: str) -> list[str]:
    s = re.sub(r"\s+", " ", text or "").strip()
    return [t.strip() for t in _SENT_SPLIT.split(s) if t.strip()]

# --- Resumen extractivo (TF-IDF) ---
def summarize_text(text: str, max_sent: int = 5) -> str:
    sents = split_sentences(text)
    if len(sents) <= max_sent:
        return text
    vect = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=1, max_df=0.9)
    X = vect.fit_transform(sents)
    scores = (X.power(2).sum(axis=1)).A1
    top = sorted(scores.argsort()[::-1][:max_sent])  # conserva orden original
    return " ".join(sents[i] for i in top)

def summarize_to_words(text: str, max_words: int = 300, max_sent: int = 5) -> str:
    s = summarize_text(text, max_sent=max_sent)
    w = s.split()
    return s if len(w) <= max_words else " ".join(w[:max_words]) + "…"

# --- Stopwords ES (única) ---
STOP_ES = {
    "el","la","los","las","un","una","unos","unas","de","del","al","a","en","y","o","u",
    "que","con","por","para","como","es","son","ser","estar","haber","no","sí","si","se",
    "mi","mis","tu","tus","su","sus","lo"
}

# --- Parafraseo ligero (fusión de reglas) ---
_REWRITE = [
    (r"\bpor lo tanto\b", "en consecuencia"),
    (r"\bes decir\b", "o sea"),
    (r"\bpor ejemplo\b", "a modo de ejemplo"),
    (r"\bmediante\b", "a través de"),
    (r"\bbarra de herramientas\b", "barra superior"),
    (r"\bverificaci[oó]n de documentos electr[oó]nicos firmados digitalmente\b", "validación de documentos firmados"),
    (r"\bse (debe|deben)\b", "debe"),
    (r"\bcon el cual\b", "con lo que"),
    (r"\bmediante el cual\b", "con lo que"),
]

def light_rephrase_es(text: str) -> str:
    out = text
    for pat, rep in _REWRITE:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out

# --- Texto a pasos/bullets ---
def to_steps(text: str, max_items: int = 6, keywords: tuple[str, ...] = ("paso","clic","seleccione","ingrese","verifique","enviar","derivar","mesa de partes","glosa")) -> str:
    sents = split_sentences(text)
    kept = []
    for s in sents:
        if any(k in s.lower() for k in keywords):
            kept.append(s)
    if not kept:
        kept = sents[:max_items]
    bullets = []
    for s in kept[:max_items]:
        s2 = re.sub(r"^\s*(Paso\s*\d+[:\-]?)\s*", "", s, flags=re.I)
        s2 = re.sub(r"\bse debe\b", "debe", s2, flags=re.I)
        s2 = re.sub(r"\busted\b", "", s2, flags=re.I)
        bullets.append("• " + s2.strip().capitalize())
    return "\n".join(bullets)

# --- Confirmación corta ---
CONFIRM_PAT = re.compile(
    r"^\s*(eso|esa|ese|este|esta)?\s*(es|era|sería)\s*(el|la)?\s*(paso|sección|apartado)\s*\d*\s*\?\s*$",
    re.IGNORECASE,
)

def is_confirmation(q: str) -> bool:
    return bool(CONFIRM_PAT.search((q or "").strip()))

# --- Fallback empático/creativo ---
def _keywords_es(text: str, limit: int = 6) -> list[str]:
    t = re.sub(r"[^\wáéíóúüñÁÉÍÓÚÜÑ ]", " ", (text or "").lower())
    toks = [w for w in t.split() if w not in STOP_ES and len(w) > 2]
    return toks[:limit]

FALLBACK_BANK = {
    "neutral": [
        "Entiendo tu situación. Ahora mismo no tengo información suficiente para responder con precisión.",
        "Gracias por contarlo. Por el momento no cuento con datos concretos para dar una respuesta segura.",
        "Quisiera ayudarte mejor, pero no tengo información disponible en este instante.",
    ],
    "positivo": [
        "Gracias por la confianza. Aún no tengo datos concretos para responder con precisión.",
        "Me alegra apoyar; de momento no tengo la información exacta a mano.",
    ],
    "triste": [
        "Siento que estés pasando por esto. Ahora mismo no tengo la información necesaria para darte una respuesta certera.",
        "Lamento la dificultad; no cuento con datos suficientes en este momento.",
    ],
    "enojado": [
        "Entiendo la frustración. Por ahora no tengo información disponible para resolverlo con certeza.",
        "Veo lo molesto que es; en este instante no dispongo de los datos para responder con precisión.",
    ],
    "ansioso": [
        "Tranquilo, vamos paso a paso. Por ahora no tengo la información necesaria.",
        "Voy a ayudarte; de momento no cuento con datos suficientes para una respuesta segura.",
    ],
}

CLARIFY_BANK = [
    "¿Podrías especificar el trámite o módulo involucrado?",
    "¿Qué error exacto ves o en qué paso te quedas?",
    "¿Desde cuándo te ocurre y en qué entorno (web, móvil, intranet)?",
    "Si tuvieras un número de expediente o área, ¿cuál sería?",
]

def creative_fallback(user_text: str, emotion_es: str = "neutral") -> str:
    base_pool = FALLBACK_BANK.get(emotion_es, FALLBACK_BANK["neutral"])
    base = random.choice(base_pool)
    kws = _keywords_es(user_text)
    if kws:
        base += f" He recogido esto de tu mensaje: {', '.join(kws[:3])}."
    followups = " ".join(random.sample(CLARIFY_BANK, k=2))
    return f"{base} {followups}"

# --- “Respuesta humana” (resumen + parafraseo o pasos) ---
def make_human_answer(raw_text: str, mode: str = "auto", max_words: int = 300) -> str:
    if not raw_text or not raw_text.strip():
        return ""
    if mode == "steps":
        core = to_steps(raw_text)
    else:
        core = summarize_to_words(raw_text, max_words=max_words)
        core = light_rephrase_es(core)
    return core

def compress_rules(text: str) -> str:
    """Pequeña limpieza + recorte de oraciones demasiado largas para evitar eco."""
    repl = [
        (r"\bpor lo tanto\b", "en consecuencia"),
        (r"\bes decir\b", "o sea"),
        (r"\bmediante\b", "a través de"),
        (r"\bbarra de herramientas\b", "barra superior"),
        (r"\s{2,}", " "),
    ]
    out = text
    for pat, rep in repl:
        out = re.sub(pat, rep, out, flags=re.I)
    pieces = []
    for s in split_sentences(out):
        pieces.append(s if len(s) < 240 else s[:240].rstrip() + "…")
    return " ".join(pieces)

# ==================================
# ===== Índices (cache) ============
# ==================================
@st.cache_resource(show_spinner=True)
def _init():
    return init_indexes()

DOC_INDEX, KB_INDEX = _init()

# ==================================
# ===== Estado inicial =============
# ==================================
if "history" not in st.session_state:
    st.session_state["history"] = [
        ("bot", "¡Hola! Soy GOBI. Tu asistente virtual para guiarte paso a paso en tus procesos. ¿En qué puedo ayudarte?")
    ]
    st.session_state["last_sources"] = []
    st.session_state["last_emotion"] = ("neutral", {"model": "start", "score": 1.0})

if "last_route" not in st.session_state:
    st.session_state["last_route"] = ""

# Sidebar
with st.sidebar:
    st.header("📁 Datos")
    st.write("Para esta demo, se estará cargando los PDFs directamente en el chatbot para enfocarnos en sus respuestas.")
    st.info("Si no hay documentos compatibles aún, GOBI responderá con mensajes genéricos.")
    st.write("**Smalltalk**")
    st.write("¿Quién eres?")
    st.write("¿Qué puedes hacer?")
    st.write("¿Cómo puedes ayudarme?")
    st.markdown("---")
    st.write("**Mejoras para la versión final:**")
    st.write("- Desplegar en un API de Telegram (aún no probado para tener disponible la versión gratuita).")
    st.write("- Definir `PUBLIC_DOC_BASE_URL` para enlaces públicos a documentos (Principalmente los tutoriales).")

# Precalentamiento (airbag)
if "emotion_warmed" not in st.session_state:
    try:
        _ = detect_emotion("warmup")
        st.session_state["emotion_warmed"] = True
    except Exception:
        st.session_state["emotion_warmed"] = False

# ============================
# ===== Chat rendering =======
# ============================
for role, msg in st.session_state["history"]:
    st.chat_message("user" if role == "user" else "assistant").write(msg)

with st.form("chat", clear_on_submit=True):
    q = st.text_input("Escribe tu consulta", "")
    submitted = st.form_submit_button("Enviar")

if "last_q" not in st.session_state:
    st.session_state["last_q"] = None
    st.session_state["last_q_ts"] = 0.0

now = time()

# ============================
# ====== Main handler ========
# ============================
if submitted and q.strip():
    # 1) Smalltalk primero (rápido)
    st_reply = smalltalk_reply(q)
    if st_reply:
        st.session_state["history"].append(("user", q))
        st.session_state["history"].append(("bot", st_reply))
        st.session_state["last_sources"] = []
        st.session_state["last_emotion"] = ("neutral", {"model": "smalltalk", "score": 1.0})
        st.rerun()

    # 2) Emoción
    try:
        label, info = detect_emotion(q)
    except Exception:
        label, info = "neutral", {"model": "fallback", "score": 0.0}
    st.session_state["last_emotion"] = (label, info)

    # 3) Motor documental
    try:
        raw_text, sources = answer_with_sources(q, DOC_INDEX, KB_INDEX, max_words=MAX_WORDS)
    except Exception:
        raw_text, sources = "", []

    # 4) Composición (concisa)
    if is_confirmation(q):
        answer = empathetic_prefix(label) + "Sí: corresponde a esa sección/paso descrito arriba. ¿Quieres que lo resuma en 3 puntos o que pase al paso siguiente?"
    else:
        if raw_text.strip():
            looks_steps = any(k in q.lower() for k in ["paso", "procedimiento", "cómo hago", "instrucción"])
            mode = "steps" if looks_steps else "auto"
            concise = make_human_answer(raw_text, mode=mode, max_words=300)
            answer = empathetic_prefix(label) + concise
        else:
            answer = creative_fallback(q, emotion_es=label)
            sources = []

    # 5) Guardar y refrescar
    st.session_state["history"].append(("user", q))
    st.session_state["history"].append(("bot", answer))
    st.session_state["last_sources"] = sources
    st.rerun()

# ============================
# ====== Extras (UI) =========
# ============================
label, info = st.session_state.get("last_emotion", ("neutral", {"model": "none", "score": 0.0}))
with st.expander("🔎 Emoción detectada", expanded=False):
    st.write(f"Modelo: {info.get('model','?')} | Etiqueta: {label} | Confianza: {info.get('score',0):.2f}")

# Fuentes del último turno
if st.session_state["history"]:
    bot_msgs = [i for i, (w, _) in enumerate(st.session_state["history"]) if w == "bot"]
    if bot_msgs:
        st.markdown("### Fuentes")
        for s in st.session_state.get("last_sources", []):
            st.markdown(f"- [{s['name']}]({s['path']})")
        if not st.session_state.get("last_sources"):
            st.caption("Se mostrarán cuando existan documentos o KB con enlaces.")
