import re
from typing import Optional

# Patrones estrictos: anclados y pensados para frases cortas
SMALLTALK_PATTERNS = [
    (re.compile(r"^\s*(hola|buenas|buenos dias|buenas tardes|buenas noches)\s*$", re.I),
     "¡Hola! Soy GOBI. ¿En qué te ayudo?"),
    (re.compile(r"^\s*(gracias|muchas gracias|te agradezco|perfecto|genial)\s*$", re.I),
     "¡Con gusto! ¿Hay algo más en lo que pueda ayudarte?"),
    (re.compile(r"^\s*(adios|hasta luego|nos vemos|chau|chao)\s*$", re.I),
     "¡Hasta luego! 👋"),
    (re.compile(r"^\s*(quien eres|quién eres|que eres|qué eres)\s*$", re.I),
     "Soy GOBI, un asistente que responde dudas frecuentes sobre tus procedimientos y te guía paso a paso."),
    (re.compile(r"^\s*(que puedes hacer|qué puedes hacer|como me ayudas|cómo me ayudas|como puedes ayudarme|cómo puedes ayudarme)\s*$", re.I),
     "Puedo buscar en tus archivos y darte un resumen (máx. 300 palabras) con enlaces a la fuente."),
    (re.compile(r"^\s*(ayuda|ayúdame|no entiendo|no se|no sé)\s*$", re.I),
     "Puedo ayudarte si me das un poco más de contexto: ¿qué trámite o módulo estás buscando?"),
]

# Si aparecen estas palabras, asumimos que NO es smalltalk sino tema “serio” (o sea, para los PDFs)
DOMAIN_HINTS = {
    "documento","pdf","docx","tramite","trámite","expediente","mesa de partes","recepción",
    "referencias","glosa","derivar","firma","sgd","contraloría","formulario","página","paso",
    "perfil","instructivo","consulta","módulo","modulo"
}

def _looks_domain(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in DOMAIN_HINTS)

def smalltalk_reply(text: str, emo_label: str = "neutral", emo_score: float = 0.0) -> Optional[str]:
    """
    Devuelve respuesta de smalltalk o None si NO debe activarse.

    Reglas:
      1) Emoción fuerte (enojo/ansiedad/tristeza/positivo alto) -> NO smalltalk
      2) Frases largas (>8 palabras) -> NO smalltalk
      3) Palabras de dominio -> NO smalltalk
      4) Si pasa filtros, aplica patrones anclados
    """
    t = (text or "").strip()
    if not t:
        return None

    # 1) Prioriza emoción
    NEG = {"enojado","ansioso","triste","disgusto"}
    POS = {"positivo","sorprendido"}
    if (emo_label in NEG and emo_score >= 0.65) or (emo_label in POS and emo_score >= 0.70):
        return None

    # 2) Longitud
    if len(t.split()) > 8:
        return None

    # 3) Dominio
    if _looks_domain(t):
        return None

    # 4) Patrones
    for pat, resp in SMALLTALK_PATTERNS:
        if pat.search(t):
            return resp
    return None