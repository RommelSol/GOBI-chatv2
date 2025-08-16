# app/emotion_ml.py
from typing import Tuple
import re, unicodedata

def _lazy_load():
    from pysentimiento import create_analyzer
    emo = create_analyzer(task="emotion", lang="es")      # joy, sadness, anger, fear, disgust, surprise, others
    sent = create_analyzer(task="sentiment", lang="es")   # POS, NEG, NEU
    return emo, sent

_EMO_ANALYZER = None
_SENT_ANALYZER = None

def _get_analyzers():
    global _EMO_ANALYZER, _SENT_ANALYZER
    if _EMO_ANALYZER is None or _SENT_ANALYZER is None:
        _EMO_ANALYZER, _SENT_ANALYZER = _lazy_load()
    return _EMO_ANALYZER, _SENT_ANALYZER

def _norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s).encode("ascii","ignore").decode("utf-8")
    return re.sub(r"\s+", " ", s.lower()).strip()

_EMO_MAP = {
    "joy": "positivo", "sadness": "triste", "anger": "enojado",
    "fear": "ansioso", "disgust": "disgusto", "surprise": "sorprendido",
    "others": "neutral",
}
_SENT_MAP = {"POS": "positivo", "NEG": "negativo", "NEU": "neutral"}

def detect_emotion(text: str) -> Tuple[str, dict]:
    t = _norm(text)
    if not t:
        return "neutral", {"model": "none", "score": 0.0}
    emo_an, sent_an = _get_analyzers()
    emo_res = emo_an.predict(t)
    emo_label = str(emo_res.output)
    emo_score = float(max(emo_res.probas.values()) if emo_res.probas else 0.0)
    emo_es = _EMO_MAP.get(emo_label, "neutral")
    if emo_score < 0.55:  # umbral simple
        sent_res = sent_an.predict(t)
        sent_label = str(sent_res.output)
        sent_score = float(max(sent_res.probas.values()) if sent_res.probas else 0.0)
        sent_es = _SENT_MAP.get(sent_label, "neutral")
        if emo_label == "others" or sent_score > emo_score:
            return sent_es, {"model": "sentiment", "label": sent_label, "score": sent_score}
    return emo_es, {"model": "emotion", "label": emo_label, "score": emo_score}

def empathetic_prefix(emotion_es: str) -> str:
    if emotion_es == "enojado":  return "Entiendo la frustración; vamos a solucionarlo. "
    if emotion_es == "triste":   return "Siento que te sientas así; haré lo posible por ayudarte. "
    if emotion_es == "ansioso":  return "Tranquilo, te acompaño paso a paso. "
    return ""